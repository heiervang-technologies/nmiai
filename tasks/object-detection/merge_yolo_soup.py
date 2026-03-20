"""Model Soup: Average weights of two YOLOv8x checkpoints.

Loads two YOLOv8x .pt files (same architecture, different training data),
averages their EMA weights, re-estimates BatchNorm running stats on a
calibration set, and exports the merged model to ONNX.

Usage:
    python merge_yolo_soup.py \
        --model1 yolo-approach/best.pt \
        --model2 yolo-approach/runs/yolov8x_v3_1280/weights/best.pt \
        --calib-dir data-creation/data/images \
        --output souped_yolov8x.onnx \
        --weights 0.5 0.5
"""
import argparse
import copy
import pathlib
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn


def load_ema_model(path: str):
    """Load the EMA model from a YOLO .pt checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ema = ckpt.get("ema") or ckpt.get("model")
    if ema is None:
        raise ValueError(f"No model or ema found in {path}")
    return ema.float()


def average_state_dicts(sd1: dict, sd2: dict, w1: float = 0.5, w2: float = 0.5):
    """Weighted average of two state dicts (same keys/shapes required)."""
    merged = {}
    for key in sd1:
        if sd1[key].dtype.is_floating_point:
            merged[key] = w1 * sd1[key] + w2 * sd2[key]
        else:
            # Non-float (e.g., num_batches_tracked) — take from model1
            merged[key] = sd1[key].clone()
    return merged


def reset_bn_stats(model: nn.Module):
    """Reset all BatchNorm running statistics to prepare for re-estimation."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()


def estimate_bn_stats(model: nn.Module, calib_dir: str, num_batches: int = 50,
                      input_size: int = 1280):
    """Re-estimate BN running stats using calibration images."""
    model.train()  # BN updates running stats in train mode
    device = next(model.parameters()).device

    calib_path = pathlib.Path(calib_dir)
    image_files = sorted(
        list(calib_path.glob("*.jpg")) + list(calib_path.glob("*.jpeg"))
        + list(calib_path.glob("*.png"))
    )
    if not image_files:
        print(f"WARNING: No calibration images found in {calib_dir}")
        print("Skipping BN re-estimation — using averaged BN stats (may be slightly worse)")
        return

    print(f"Re-estimating BatchNorm stats on {min(num_batches, len(image_files))} images...")

    with torch.no_grad():
        for i, img_path in enumerate(image_files[:num_batches]):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            # Letterbox resize
            h, w = img.shape[:2]
            ratio = input_size / max(h, w)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img_resized = cv2.resize(img, (new_w, new_h))
            dw = (input_size - new_w) // 2
            dh = (input_size - new_h) // 2
            padded = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
            padded[dh:dh + new_h, dw:dw + new_w] = img_resized

            blob = padded[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
            tensor = torch.from_numpy(blob).unsqueeze(0).to(device)
            model(tensor)

            if (i + 1) % 10 == 0:
                print(f"  BN calibration: {i + 1}/{min(num_batches, len(image_files))}")

    model.eval()
    print("BN re-estimation complete.")


def export_onnx_ultralytics(ema_model, ckpt_path: str, merged_sd: dict,
                             output_path: str, input_size: int = 1280):
    """Export merged model to ONNX using ultralytics' own export pipeline.

    We modify the checkpoint in-place, save a temp .pt, then use YOLO.export().
    """
    from ultralytics import YOLO

    # Save merged weights as a temp YOLO checkpoint
    tmp_pt = pathlib.Path(output_path).with_suffix(".pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Replace EMA weights with merged weights
    ckpt["ema"].load_state_dict(merged_sd)
    # Also set model to the merged weights (some export paths use 'model')
    if ckpt.get("model") is not None:
        ckpt["model"].load_state_dict(merged_sd)
    torch.save(ckpt, str(tmp_pt))

    # Export using ultralytics
    yolo = YOLO(str(tmp_pt))
    yolo.export(format="onnx", imgsz=input_size, simplify=True, opset=12, half=True)

    # Ultralytics saves next to the .pt file
    exported = tmp_pt.with_suffix(".onnx")
    if exported.exists():
        shutil.move(str(exported), output_path)
        size_mb = pathlib.Path(output_path).stat().st_size / (1024 * 1024)
        print(f"Exported ONNX: {output_path} ({size_mb:.1f} MB)")
    else:
        print(f"ERROR: Expected {exported} but not found")
        # List what was created
        for f in tmp_pt.parent.glob(tmp_pt.stem + "*"):
            print(f"  Found: {f} ({f.stat().st_size / 1024 / 1024:.1f} MB)")

    # Clean up temp .pt
    if tmp_pt.exists():
        tmp_pt.unlink()


def main():
    parser = argparse.ArgumentParser(description="Model Soup: merge two YOLOv8x checkpoints")
    parser.add_argument("--model1", required=True, help="Path to first .pt checkpoint")
    parser.add_argument("--model2", required=True, help="Path to second .pt checkpoint")
    parser.add_argument("--calib-dir", default=None,
                        help="Directory with calibration images for BN re-estimation")
    parser.add_argument("--output", default="souped_yolov8x.onnx", help="Output ONNX path")
    parser.add_argument("--weights", nargs=2, type=float, default=[0.5, 0.5],
                        help="Weights for averaging (default: 0.5 0.5)")
    parser.add_argument("--input-size", type=int, default=1280, help="Input resolution")
    parser.add_argument("--num-calib", type=int, default=50, help="Number of calibration images")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    w1, w2 = args.weights
    total = w1 + w2
    w1, w2 = w1 / total, w2 / total
    print(f"Merging with weights: model1={w1:.2f}, model2={w2:.2f}")

    # Load both models
    print(f"Loading model1: {args.model1}")
    model1 = load_ema_model(args.model1)
    print(f"Loading model2: {args.model2}")
    model2 = load_ema_model(args.model2)

    sd1 = model1.state_dict()
    sd2 = model2.state_dict()

    # Verify compatibility
    if set(sd1.keys()) != set(sd2.keys()):
        raise ValueError("Models have different architectures (key mismatch)")
    for key in sd1:
        if sd1[key].shape != sd2[key].shape:
            raise ValueError(f"Shape mismatch at {key}: {sd1[key].shape} vs {sd2[key].shape}")

    print(f"Architecture match confirmed: {len(sd1)} layers, "
          f"{sum(p.numel() for p in sd1.values()):,} parameters")

    # Average weights
    print("Averaging weights...")
    merged_sd = average_state_dicts(sd1, sd2, w1, w2)

    # Load merged weights into a fresh model copy
    merged_model = copy.deepcopy(model1)
    merged_model.load_state_dict(merged_sd)
    merged_model.to(args.device)

    # Re-estimate BatchNorm stats
    if args.calib_dir:
        reset_bn_stats(merged_model)
        estimate_bn_stats(merged_model, args.calib_dir, args.num_calib, args.input_size)
    else:
        print("No calibration dir provided — using averaged BN stats (may be suboptimal)")
        print("Tip: pass --calib-dir with training images for better results")

    # Export to ONNX via ultralytics
    export_onnx_ultralytics(model1, args.model1, merged_sd, args.output, args.input_size)
    print("Done! Model soup complete.")


if __name__ == "__main__":
    main()

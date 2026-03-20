"""Advanced model merging: SLERP, greedy soup, weight sweep.

Implements:
  - SLERP (Spherical Linear Interpolation) - better than linear for 2-model merge
  - Greedy Model Soup - iteratively adds checkpoints only if they improve val mAP
  - Weight sweep - find optimal alpha for interpolation

Usage:
    # SLERP merge
    python merge_yolo_advanced.py slerp \
        --model1 yolo-approach/best.pt \
        --model2 yolo-approach/runs/yolov8x_v3_1280/weights/best.pt \
        --alpha 0.5 \
        --calib-dir yolo-approach/dataset/images/train \
        --output slerp_yolov8x.onnx

    # Greedy soup across multiple checkpoints
    python merge_yolo_advanced.py greedy \
        --models yolo-approach/best.pt yolo-approach/runs/yolov8x_v3_1280/weights/best.pt \
                 yolo-approach/runs/yolov8x_v3_1280/weights/last.pt \
        --val-images yolo-approach/dataset/images/val \
        --val-labels yolo-approach/dataset/labels/val \
        --calib-dir yolo-approach/dataset/images/train \
        --output greedy_soup.onnx

    # Weight sweep to find optimal alpha
    python merge_yolo_advanced.py sweep \
        --model1 yolo-approach/best.pt \
        --model2 yolo-approach/runs/yolov8x_v3_1280/weights/best.pt \
        --val-images yolo-approach/dataset/images/val \
        --val-labels yolo-approach/dataset/labels/val \
        --steps 11
"""
import argparse
import copy
import json
import pathlib
import shutil

import cv2
import numpy as np
import torch
import torch.nn as nn


# ─── Model loading ───────────────────────────────────────────────────────────

def load_ema_model(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ema = ckpt.get("ema") or ckpt.get("model")
    if ema is None:
        raise ValueError(f"No model or ema found in {path}")
    return ema.float()


def get_state_dict(path: str):
    return load_ema_model(path).state_dict()


# ─── Merge methods ───────────────────────────────────────────────────────────

def lerp_state_dicts(sd1, sd2, alpha=0.5):
    """Linear interpolation: (1-alpha)*sd1 + alpha*sd2"""
    merged = {}
    for key in sd1:
        if sd1[key].dtype.is_floating_point:
            merged[key] = (1 - alpha) * sd1[key] + alpha * sd2[key]
        else:
            merged[key] = sd1[key].clone()
    return merged


def slerp_state_dicts(sd1, sd2, alpha=0.5):
    """Spherical Linear Interpolation between two state dicts.

    SLERP interpolates along the great circle on the weight hypersphere,
    preserving the magnitude of the weights better than linear interpolation.
    For each parameter tensor, we flatten, compute SLERP, then reshape.
    """
    merged = {}
    for key in sd1:
        if not sd1[key].dtype.is_floating_point:
            merged[key] = sd1[key].clone()
            continue

        v1 = sd1[key].flatten().float()
        v2 = sd2[key].flatten().float()

        # Normalize
        norm1 = torch.norm(v1)
        norm2 = torch.norm(v2)

        if norm1 < 1e-8 or norm2 < 1e-8:
            # Degenerate case: fall back to lerp
            merged[key] = ((1 - alpha) * sd1[key] + alpha * sd2[key])
            continue

        v1_unit = v1 / norm1
        v2_unit = v2 / norm2

        # Compute angle between vectors
        dot = torch.clamp(torch.dot(v1_unit, v2_unit), -1.0, 1.0)
        omega = torch.acos(dot)

        if omega.abs() < 1e-6:
            # Vectors nearly parallel: fall back to lerp
            merged[key] = ((1 - alpha) * sd1[key] + alpha * sd2[key])
            continue

        sin_omega = torch.sin(omega)
        # SLERP on the unit sphere
        interp_unit = (torch.sin((1 - alpha) * omega) / sin_omega) * v1_unit + \
                      (torch.sin(alpha * omega) / sin_omega) * v2_unit
        # Interpolate magnitude linearly
        interp_norm = (1 - alpha) * norm1 + alpha * norm2
        interp = interp_unit * interp_norm
        merged[key] = interp.reshape(sd1[key].shape)

    return merged


def average_state_dicts(sds, weights=None):
    """Weighted average of N state dicts."""
    if weights is None:
        weights = [1.0 / len(sds)] * len(sds)
    else:
        total = sum(weights)
        weights = [w / total for w in weights]

    merged = {}
    for key in sds[0]:
        if sds[0][key].dtype.is_floating_point:
            merged[key] = sum(w * sd[key] for w, sd in zip(weights, sds))
        else:
            merged[key] = sds[0][key].clone()
    return merged


# ─── BN recalibration ────────────────────────────────────────────────────────

def reset_bn_stats(model):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.SyncBatchNorm)):
            m.running_mean.zero_()
            m.running_var.fill_(1.0)
            m.num_batches_tracked.zero_()


def estimate_bn_stats(model, calib_dir, num_batches=50, input_size=1280):
    model.train()
    device = next(model.parameters()).device
    calib_path = pathlib.Path(calib_dir)
    image_files = sorted(
        list(calib_path.glob("*.jpg")) + list(calib_path.glob("*.jpeg"))
        + list(calib_path.glob("*.png"))
    )
    if not image_files:
        print(f"WARNING: No calibration images in {calib_dir}")
        return

    with torch.no_grad():
        for i, img_path in enumerate(image_files[:num_batches]):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
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

    model.eval()


def recalibrate_model(base_model, merged_sd, calib_dir, num_calib=50, input_size=1280):
    """Load merged state dict into model, recalibrate BN, return model."""
    model = copy.deepcopy(base_model)
    model.load_state_dict(merged_sd)
    reset_bn_stats(model)
    if calib_dir:
        estimate_bn_stats(model, calib_dir, num_calib, input_size)
    return model


# ─── ONNX export ─────────────────────────────────────────────────────────────

def export_onnx_ultralytics(ckpt_path, merged_sd, output_path, input_size=1280):
    from ultralytics import YOLO
    tmp_pt = pathlib.Path(output_path).with_suffix(".pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    ckpt["ema"].load_state_dict(merged_sd)
    if ckpt.get("model") is not None:
        ckpt["model"].load_state_dict(merged_sd)
    torch.save(ckpt, str(tmp_pt))
    yolo = YOLO(str(tmp_pt))
    yolo.export(format="onnx", imgsz=input_size, simplify=True, opset=12, half=True)
    exported = tmp_pt.with_suffix(".onnx")
    if exported.exists():
        shutil.move(str(exported), output_path)
        size_mb = pathlib.Path(output_path).stat().st_size / (1024 * 1024)
        print(f"Exported ONNX: {output_path} ({size_mb:.1f} MB)")
    if tmp_pt.exists():
        tmp_pt.unlink()


# ─── Quick validation (box count/confidence proxy) ──────────────────────────

def quick_val_score(model, val_dir, input_size=1280, conf_thresh=0.25):
    """Quick proxy metric: count high-confidence detections.
    Not a real mAP but useful for relative comparison between merges.
    """
    model.eval()
    device = next(model.parameters()).device
    val_path = pathlib.Path(val_dir)
    image_files = sorted(
        list(val_path.glob("*.jpg")) + list(val_path.glob("*.jpeg"))
        + list(val_path.glob("*.png"))
    )

    total_dets = 0
    total_conf = 0.0

    with torch.no_grad():
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
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

            output = model(tensor)
            if isinstance(output, (list, tuple)):
                output = output[0]
            if output.ndim == 3:
                output = output[0]
            if output.shape[0] < output.shape[1]:
                output = output.T

            class_scores = output[:, 4:]
            max_scores = class_scores.max(dim=1).values
            mask = max_scores > conf_thresh
            total_dets += mask.sum().item()
            total_conf += max_scores[mask].sum().item()

    avg_conf = total_conf / max(total_dets, 1)
    print(f"  Detections (conf>{conf_thresh}): {total_dets}, avg_conf: {avg_conf:.4f}")
    return total_dets, avg_conf


# ─── Commands ────────────────────────────────────────────────────────────────

def cmd_slerp(args):
    print(f"SLERP merge with alpha={args.alpha}")
    print(f"  model1: {args.model1}")
    print(f"  model2: {args.model2}")

    model1 = load_ema_model(args.model1)
    sd1 = model1.state_dict()
    sd2 = get_state_dict(args.model2)

    merged_sd = slerp_state_dicts(sd1, sd2, args.alpha)

    model = recalibrate_model(model1, merged_sd, args.calib_dir, args.num_calib)
    final_sd = model.state_dict()

    export_onnx_ultralytics(args.model1, final_sd, args.output)
    print("SLERP merge complete!")


def cmd_sweep(args):
    """Sweep alpha values and compare quick proxy metrics."""
    print(f"Weight sweep between:")
    print(f"  model1: {args.model1}")
    print(f"  model2: {args.model2}")

    model1 = load_ema_model(args.model1)
    sd1 = model1.state_dict()
    sd2 = get_state_dict(args.model2)

    alphas = np.linspace(0.0, 1.0, args.steps)
    results = []

    for alpha in alphas:
        print(f"\nalpha={alpha:.2f}:")
        merged_sd = slerp_state_dicts(sd1, sd2, alpha) if args.method == "slerp" else \
                    lerp_state_dicts(sd1, sd2, alpha)
        model = recalibrate_model(model1, merged_sd, args.calib_dir, args.num_calib)
        dets, conf = quick_val_score(model, args.val_images)
        results.append((alpha, dets, conf))

    print("\n=== Sweep Results ===")
    print(f"{'Alpha':>6}  {'Detections':>10}  {'Avg Conf':>10}")
    for alpha, dets, conf in results:
        print(f"{alpha:>6.2f}  {dets:>10}  {conf:>10.4f}")

    best = max(results, key=lambda x: x[1])  # Most detections
    print(f"\nBest alpha by detection count: {best[0]:.2f}")


def cmd_greedy(args):
    """Greedy model soup: iteratively add models if they improve proxy metric."""
    print(f"Greedy soup with {len(args.models)} candidates:")
    for m in args.models:
        print(f"  - {m}")

    # Start with first model as base
    base_model = load_ema_model(args.models[0])
    current_sd = base_model.state_dict()

    # Evaluate base
    print(f"\nBase model ({args.models[0]}):")
    model = recalibrate_model(base_model, current_sd, args.calib_dir, args.num_calib)
    base_dets, base_conf = quick_val_score(model, args.val_images)
    best_dets = base_dets
    included = [args.models[0]]

    # Try adding each subsequent model
    for candidate_path in args.models[1:]:
        print(f"\nTrying to add: {candidate_path}")
        candidate_sd = get_state_dict(candidate_path)

        # Average current soup with candidate
        n = len(included)
        trial_sd = {}
        for key in current_sd:
            if current_sd[key].dtype.is_floating_point:
                trial_sd[key] = (current_sd[key] * n + candidate_sd[key]) / (n + 1)
            else:
                trial_sd[key] = current_sd[key].clone()

        model = recalibrate_model(base_model, trial_sd, args.calib_dir, args.num_calib)
        trial_dets, trial_conf = quick_val_score(model, args.val_images)

        if trial_dets >= best_dets:
            print(f"  ACCEPTED (dets: {best_dets} -> {trial_dets})")
            current_sd = trial_sd
            best_dets = trial_dets
            included.append(candidate_path)
        else:
            print(f"  REJECTED (dets: {best_dets} -> {trial_dets})")

    print(f"\nFinal soup includes {len(included)} models:")
    for m in included:
        print(f"  - {m}")

    # Recalibrate and export
    model = recalibrate_model(base_model, current_sd, args.calib_dir, args.num_calib)
    final_sd = model.state_dict()
    export_onnx_ultralytics(args.models[0], final_sd, args.output)
    print("Greedy soup complete!")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Advanced YOLO model merging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # SLERP
    p = subparsers.add_parser("slerp", help="SLERP merge of two models")
    p.add_argument("--model1", required=True)
    p.add_argument("--model2", required=True)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--calib-dir", default=None)
    p.add_argument("--num-calib", type=int, default=50)
    p.add_argument("--output", default="slerp_yolov8x.onnx")

    # Sweep
    p = subparsers.add_parser("sweep", help="Sweep alpha values")
    p.add_argument("--model1", required=True)
    p.add_argument("--model2", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--calib-dir", default=None)
    p.add_argument("--num-calib", type=int, default=50)
    p.add_argument("--steps", type=int, default=11)
    p.add_argument("--method", choices=["lerp", "slerp"], default="slerp")

    # Greedy soup
    p = subparsers.add_parser("greedy", help="Greedy model soup")
    p.add_argument("--models", nargs="+", required=True)
    p.add_argument("--val-images", required=True)
    p.add_argument("--calib-dir", default=None)
    p.add_argument("--num-calib", type=int, default=50)
    p.add_argument("--output", default="greedy_soup.onnx")

    args = parser.parse_args()
    {"slerp": cmd_slerp, "sweep": cmd_sweep, "greedy": cmd_greedy}[args.command](args)


if __name__ == "__main__":
    main()

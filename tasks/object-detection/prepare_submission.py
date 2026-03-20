"""Prepare submission-ensemble directory with all required files.

Creates:
  submission-ensemble/
    run.py          - already written
    yolo_a.onnx     - primary detector (souped or best)
    yolo_b.onnx     - secondary detector
    dino_with_probe.pth - combined DINOv2 + linear probe weights

Usage:
    # Option A: Use souped model + yolo11x
    python prepare_submission.py \
        --yolo-a souped_yolov8x.onnx \
        --yolo-b titan-models/yolo11x_v3.onnx

    # Option B: Use original best + yolo11x (no soup)
    python prepare_submission.py \
        --yolo-a yolo-approach/best.onnx \
        --yolo-b titan-models/yolo11x_v3.onnx
"""
import argparse
import shutil
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-a", required=True, help="Primary YOLO ONNX model")
    parser.add_argument("--yolo-b", required=True, help="Secondary YOLO ONNX model")
    parser.add_argument("--dino", default="submission-single-model/dinov2_vits14.pth")
    parser.add_argument("--probe", default="submission-single-model/linear_probe.pth")
    parser.add_argument("--out-dir", default="submission-ensemble")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(exist_ok=True)

    # Copy YOLO models
    for src, dst_name in [(args.yolo_a, "yolo_a.onnx"), (args.yolo_b, "yolo_b.onnx")]:
        dst = out / dst_name
        print(f"Copying {src} -> {dst}")
        shutil.copy2(src, dst)

    # Combine DINOv2 + linear probe
    print(f"Combining {args.dino} + {args.probe} -> dino_with_probe.pth")
    dino_state = torch.load(args.dino, map_location="cpu", weights_only=True)
    probe_state = torch.load(args.probe, map_location="cpu", weights_only=True)

    combined = dict(dino_state)
    for k, v in probe_state.items():
        combined[f"probe.{k}"] = v

    torch.save(combined, out / "dino_with_probe.pth")

    # Summary
    total_size = 0
    weight_files = 0
    py_files = 0
    for f in sorted(out.iterdir()):
        size = f.stat().st_size / (1024 * 1024)
        total_size += size
        ext = f.suffix
        if ext in (".onnx", ".pth", ".pt"):
            weight_files += 1
        elif ext == ".py":
            py_files += 1
        print(f"  {f.name:30s} {size:8.1f} MB")

    print(f"\nTotal: {total_size:.1f} MB ({weight_files} weight files, {py_files} Python files)")
    print(f"Budget: 420 MB / 3 weight files / 10 Python files")
    ok = total_size <= 420 and weight_files <= 3 and py_files <= 10
    print(f"Status: {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()

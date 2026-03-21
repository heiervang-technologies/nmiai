"""Package V5 YOLO submission with DINOv2 classification.

Creates a submission ZIP with:
  - run.py (proven DINOv2+YOLO pipeline from submission-single-model)
  - best.onnx (V5 exported model)
  - dinov2_vits14.pth (DINOv2 backbone weights)
  - linear_probe.pth (classification head)

Usage:
  # After V5 ONNX export:
  python package_v5_submission.py

  # Or with custom paths:
  python package_v5_submission.py --onnx /path/to/v5_best.onnx
"""

import argparse
import zipfile
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
SUBMISSION_TEMPLATE = TASK_DIR / "submission-single-model"
DEFAULT_V5_ONNX = Path("/home/me/ht/nmiai/runs/detect/yolov8x_v5_b2/weights/best.onnx")
OUTPUT_ZIP = TASK_DIR / "submission_v5_dino.zip"

REQUIRED_FILES = {
    "run.py": SUBMISSION_TEMPLATE / "run.py",
    "dinov2_vits14.pth": SUBMISSION_TEMPLATE / "dinov2_vits14.pth",
    "linear_probe.pth": SUBMISSION_TEMPLATE / "linear_probe.pth",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_V5_ONNX,
                        help="Path to V5 best.onnx")
    parser.add_argument("--output", type=Path, default=OUTPUT_ZIP,
                        help="Output ZIP path")
    args = parser.parse_args()

    onnx_path = args.onnx.resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"V5 ONNX not found: {onnx_path}")

    # Validate all required files exist
    for name, path in REQUIRED_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    # Package ZIP
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        # V5 ONNX as best.onnx
        zf.write(onnx_path, "best.onnx")
        print(f"  best.onnx <- {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
        # Template files
        for arcname, src in REQUIRED_FILES.items():
            zf.write(src, arcname)
            print(f"  {arcname} <- {src} ({src.stat().st_size / 1e6:.1f} MB)")

    zip_size = args.output.stat().st_size / 1e6
    print(f"\nZIP: {args.output} ({zip_size:.1f} MB)")
    if zip_size > 420:
        print("WARNING: ZIP exceeds 420 MB limit!")
    else:
        print(f"OK: {420 - zip_size:.1f} MB under limit")


if __name__ == "__main__":
    main()

"""Package V5 YOLO + MarkusNet vision submission.

Creates a submission ZIP with:
  - run.py (MarkusNet vision pipeline from submission-markusnet-vision)
  - best.onnx (V5 YOLO detector)
  - markusnet_vision_int8.onnx (MarkusNet vision encoder)
  - markusnet_probe.onnx (classification head)
"""

import argparse
import zipfile
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
MARKUSNET_TEMPLATE = TASK_DIR / "submission-markusnet-vision"
DEFAULT_V5_ONNX = Path("/home/me/ht/nmiai/runs/detect/yolov8x_v5_b2/weights/best.onnx")
OUTPUT_ZIP = TASK_DIR / "submission_v5_markusnet.zip"

REQUIRED_FILES = {
    "run.py": MARKUSNET_TEMPLATE / "run.py",
    "markusnet_vision_int8.onnx": MARKUSNET_TEMPLATE / "markusnet_vision_int8.onnx",
    "markusnet_probe.onnx": MARKUSNET_TEMPLATE / "markusnet_probe.onnx",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=Path, default=DEFAULT_V5_ONNX)
    parser.add_argument("--output", type=Path, default=OUTPUT_ZIP)
    args = parser.parse_args()

    onnx_path = args.onnx.resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"V5 ONNX not found: {onnx_path}")

    for name, path in REQUIRED_FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(onnx_path, "best.onnx")
        print(f"  best.onnx <- {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")
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

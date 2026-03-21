"""Package V5+YOLO11x ensemble + DINOv2 submission."""
import argparse
import zipfile
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
ENSEMBLE_TEMPLATE = TASK_DIR / "submission-ensemble"
V5_ONNX = Path("/home/me/ht/nmiai/runs/detect/yolov8x_v5_b2/weights/best.onnx")
YOLO11X_ONNX = TASK_DIR / "titan-models" / "yolo11x_v3.onnx"
OUTPUT_ZIP = TASK_DIR / "submission_v5_ensemble.zip"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ZIP)
    args = parser.parse_args()

    files = {
        "run.py": ENSEMBLE_TEMPLATE / "run.py",
        "yolo_a.onnx": V5_ONNX,
        "yolo_b.onnx": YOLO11X_ONNX,
        "dino_with_probe.pth": ENSEMBLE_TEMPLATE / "dino_with_probe.pth",
    }

    for name, path in files.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")

    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, src in files.items():
            zf.write(src, arcname)
            print(f"  {arcname} <- {src} ({src.stat().st_size / 1e6:.1f} MB)")

    zip_size = args.output.stat().st_size / 1e6
    print(f"\nZIP: {args.output} ({zip_size:.1f} MB)")
    print(f"{'OK' if zip_size <= 420 else 'WARNING'}: {420 - zip_size:.1f} MB {'under' if zip_size <= 420 else 'OVER'} limit")

if __name__ == "__main__":
    main()

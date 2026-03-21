"""Export the active V5 YOLO checkpoint to ONNX and package a submission ZIP.

This is intended to be the one-command handoff when training finishes:

    python export_v5_submission.py

It exports the latest `best.pt` from the V5 run to `best.onnx`, swaps that
artifact into the YOLO ONNX submission template, and writes a versioned zip.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
YOLO_DIR = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = ROOT / "runs" / "detect" / "yolov8x_v5_b2"
DEFAULT_WEIGHTS = DEFAULT_RUN_DIR / "weights" / "best.pt"
DEFAULT_ONNX = DEFAULT_RUN_DIR / "weights" / "best.onnx"
DEFAULT_OUTPUT_ZIP = YOLO_DIR / "nmiai_yolov8x_v5_b2_onnx.zip"
DEFAULT_MANIFEST = DEFAULT_RUN_DIR / "submission_export_manifest.json"


def export_onnx(weights_path: Path, onnx_path: Path, imgsz: int, half: bool, simplify: bool):
    model = YOLO(str(weights_path))
    export_result = model.export(format="onnx", imgsz=imgsz, half=half, simplify=simplify)
    exported_path = Path(str(export_result))
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    if exported_path.resolve() != onnx_path.resolve():
        shutil.copy2(exported_path, onnx_path)
    return onnx_path


def package_submission(onnx_path: Path, output_zip: Path):
    run_py = YOLO_DIR / "run_onnx.py"
    with tempfile.TemporaryDirectory(prefix="yolo_v5_pkg_") as tmpdir:
        stage = Path(tmpdir)
        shutil.copy2(run_py, stage / "run.py")
        shutil.copy2(onnx_path, stage / "best.onnx")
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(stage / "run.py", "run.py")
            zf.write(stage / "best.onnx", "best.onnx")
    return output_zip


def write_manifest(manifest_path: Path, payload: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--onnx", type=Path, default=DEFAULT_ONNX)
    parser.add_argument("--output-zip", type=Path, default=DEFAULT_OUTPUT_ZIP)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--no-half", action="store_true")
    parser.add_argument("--no-simplify", action="store_true")
    args = parser.parse_args()

    weights_path = args.weights.resolve()
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing weights: {weights_path}")

    onnx_path = export_onnx(
        weights_path=weights_path,
        onnx_path=args.onnx.resolve(),
        imgsz=args.imgsz,
        half=not args.no_half,
        simplify=not args.no_simplify,
    )
    output_zip = package_submission(onnx_path=onnx_path, output_zip=args.output_zip.resolve())

    payload = {
        "run_dir": str(args.run_dir.resolve()),
        "weights": str(weights_path),
        "onnx": str(onnx_path),
        "output_zip": str(output_zip),
        "weights_size_mb": round(weights_path.stat().st_size / 1e6, 2),
        "onnx_size_mb": round(onnx_path.stat().st_size / 1e6, 2),
        "zip_size_mb": round(output_zip.stat().st_size / 1e6, 2),
        "imgsz": args.imgsz,
        "half": not args.no_half,
        "simplify": not args.no_simplify,
    }
    write_manifest(args.manifest.resolve(), payload)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

"""
Export trained YOLO26x to ONNX FP16 for sandbox submission.

Usage:
    uv run python export_yolo26_onnx.py --weights best.pt --imgsz 640 --output best_fp16.onnx
"""

import argparse


def export_onnx(weights_path, imgsz, output_path):
    from ultralytics import YOLO

    model = YOLO(weights_path)

    # Export to ONNX with FP16
    model.export(
        format="onnx",
        imgsz=imgsz,
        half=True,
        simplify=True,
        opset=17,
        dynamic=False,
        batch=1,
    )

    # The export creates a file next to the weights
    import pathlib

    default_onnx = pathlib.Path(weights_path).with_suffix(".onnx")
    if default_onnx.exists() and str(default_onnx) != output_path:
        default_onnx.rename(output_path)
        print(f"Moved to: {output_path}")

    # Print size
    onnx_path = pathlib.Path(output_path)
    if onnx_path.exists():
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"ONNX model size: {size_mb:.1f} MB")
    else:
        print(f"Warning: expected output at {output_path} not found")
        print(f"Check: {default_onnx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to .pt weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--output", default="best_fp16.onnx")
    args = parser.parse_args()
    export_onnx(args.weights, args.imgsz, args.output)

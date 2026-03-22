#!/usr/bin/env python3
"""Evaluate pretrained model baselines on the 248-image val set.

Establishes transfer learning floor (COCO pretrained, zero training)
and dishonest ceiling (trained on competition data).

Usage: python eval_pretrained_baselines.py
"""
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_SCRIPT = SCRIPT_DIR / "eval_honest.py"


def run_eval(model_path, experiment_id, description, unique_images=0):
    """Run eval_honest.py on a model."""
    cmd = [
        "python", str(EVAL_SCRIPT),
        str(model_path),
        "--experiment-id", experiment_id,
        "--description", description,
        "--unique-images", str(unique_images),
        "--data-sources", "pretrained",
    ]
    print(f"\n{'=' * 60}")
    print(f"Evaluating: {description}")
    print(f"Model: {model_path}")
    print(f"{'=' * 60}")
    subprocess.run(cmd)


def main():
    baselines = []

    # 1. COCO pretrained YOLOv8x (zero-shot detection baseline)
    # This downloads automatically if not present
    baselines.append({
        "model": "yolov8x.pt",
        "id": "baseline_coco_yolov8x",
        "desc": "YOLOv8x COCO pretrained (zero-shot)",
        "unique": 0,
    })

    # 2. Our best trained ONNX model (trained on competition data = dishonest ceiling)
    onnx_candidates = [
        SCRIPT_DIR / "yolo-approach" / "runs" / "detect" / "yolov8x_v6_clean" / "weights" / "best.onnx",
        SCRIPT_DIR / "submission-single-model" / "best.onnx",
        SCRIPT_DIR / "submission-ensemble" / "yolo_a.onnx",
    ]
    for onnx in onnx_candidates:
        if onnx.exists():
            baselines.append({
                "model": str(onnx),
                "id": f"baseline_trained_{onnx.parent.parent.name}",
                "desc": f"Trained on competition data (dishonest ceiling) - {onnx.parent.parent.name}",
                "unique": 248,
            })

    # 3. Any .pt models in yolo runs
    runs_dir = SCRIPT_DIR / "yolo-approach" / "runs" / "detect"
    if runs_dir.exists():
        for run_dir in sorted(runs_dir.iterdir()):
            best_pt = run_dir / "weights" / "best.pt"
            if best_pt.exists() and "v6" not in run_dir.name:  # Skip v6, already covered by ONNX
                baselines.append({
                    "model": str(best_pt),
                    "id": f"baseline_{run_dir.name}",
                    "desc": f"Existing trained model - {run_dir.name}",
                    "unique": 0,
                })

    if not baselines:
        print("No baseline models found!")
        return

    print(f"Found {len(baselines)} baselines to evaluate")
    for b in baselines:
        print(f"  - {b['desc']}")

    for b in baselines:
        model_path = Path(b["model"])
        if not model_path.exists() and not b["model"].endswith(".pt"):
            print(f"Skipping missing: {b['model']}")
            continue
        run_eval(b["model"], b["id"], b["desc"], b["unique"])

    # Plot progress
    plot_script = SCRIPT_DIR / "plot_progress.py"
    if plot_script.exists():
        subprocess.run(["python", str(plot_script)])

    print("\nBaseline evaluation complete. Check data_experiment_results.tsv")


if __name__ == "__main__":
    main()

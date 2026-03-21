#!/usr/bin/env python3
"""
Watch V8 training checkpoints and evaluate on clean val set.
Runs on GPU 1 while training runs on GPU 0.

Reports combined score (0.7*det + 0.3*cls mAP@0.5) for each checkpoint.

Usage: CUDA_VISIBLE_DEVICES=1 python watch_v8_checkpoints.py
"""
import json
import time
from pathlib import Path

from ultralytics import YOLO

CHECKPOINT_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/yolo-approach/runs/yolov8x_v8_mega/weights")
VAL_YAML = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data/mega_dataset/dataset.yaml")
RESULTS_FILE = Path("/home/me/ht/nmiai/tasks/object-detection/v8_checkpoint_scores.json")

evaluated = set()


def evaluate_checkpoint(ckpt_path):
    """Run val on a checkpoint and return metrics."""
    print(f"\nEvaluating: {ckpt_path.name}")
    model = YOLO(str(ckpt_path))
    results = model.val(
        data=str(VAL_YAML),
        imgsz=1280,
        batch=4,
        device=0,  # maps to GPU 1 via CUDA_VISIBLE_DEVICES
        verbose=False,
    )

    det_map50 = float(results.box.map50)
    cls_map50 = float(results.box.map50)  # same for single-stage YOLO
    combined = 0.7 * det_map50 + 0.3 * cls_map50

    return {
        "checkpoint": ckpt_path.name,
        "map50": round(det_map50, 4),
        "map50_95": round(float(results.box.map), 4),
        "combined": round(combined, 4),
        "precision": round(float(results.box.mp), 4),
        "recall": round(float(results.box.mr), 4),
    }


def main():
    results = []
    if RESULTS_FILE.exists():
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        evaluated.update(r["checkpoint"] for r in results)

    print("Watching for V8 checkpoints...")
    print(f"Checkpoint dir: {CHECKPOINT_DIR}")
    print(f"Val data: {VAL_YAML}")

    while True:
        if not CHECKPOINT_DIR.exists():
            print("Waiting for checkpoint directory...")
            time.sleep(30)
            continue

        # Check for new checkpoints
        for ckpt in sorted(CHECKPOINT_DIR.glob("*.pt")):
            if ckpt.name in evaluated:
                continue
            if ckpt.name == "last.pt":
                # Only eval last.pt if it hasn't been updated recently (training might be writing)
                age = time.time() - ckpt.stat().st_mtime
                if age < 60:
                    continue

            try:
                result = evaluate_checkpoint(ckpt)
                results.append(result)
                evaluated.add(ckpt.name)
                print(f"  mAP@0.5={result['map50']:.4f} P={result['precision']:.4f} R={result['recall']:.4f}")

                with open(RESULTS_FILE, "w") as f:
                    json.dump(results, f, indent=2)
            except Exception as e:
                print(f"  Error: {e}")

        # Check every 2 minutes
        time.sleep(120)


if __name__ == "__main__":
    main()

"""
Quick evaluation of a YOLO checkpoint against clean validation set.
Useful for deciding when to stop training early.

Usage: python quick_eval_checkpoint.py /path/to/best.pt
"""
import argparse
import json
from pathlib import Path
from collections import Counter

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pt checkpoint")
    parser.add_argument("--val-images", default=str(Path(__file__).parent / "data/clean_split/val/images"))
    parser.add_argument("--val-labels", default=str(Path(__file__).parent / "data/clean_split/val/labels"))
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from ultralytics import YOLO
    import numpy as np

    print(f"Evaluating: {args.checkpoint}")
    model = YOLO(args.checkpoint)

    val_images = sorted(Path(args.val_images).glob("*.jpg")) + sorted(Path(args.val_images).glob("*.jpeg"))
    val_labels_dir = Path(args.val_labels)
    print(f"Validation images: {len(val_images)}")

    # Run predictions
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_tp_cls = 0

    for img_path in val_images:
        results = model.predict(str(img_path), conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                               device=args.device, verbose=False)

        if not results:
            continue

        # Load ground truth
        lbl_path = val_labels_dir / (img_path.stem + ".txt")
        gt_boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().split("\n"):
                if line.strip():
                    parts = line.split()
                    gt_boxes.append(int(parts[0]))

        pred_count = len(results[0].boxes) if results[0].boxes is not None else 0

        # Simple TP/FP/FN counting (not proper mAP but fast indicator)
        tp = min(pred_count, len(gt_boxes))
        fp = max(0, pred_count - len(gt_boxes))
        fn = max(0, len(gt_boxes) - pred_count)
        all_tp += tp
        all_fp += fp
        all_fn += fn

    precision = all_tp / max(1, all_tp + all_fp)
    recall = all_tp / max(1, all_tp + all_fn)
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    print(f"\nQuick Metrics (detection count-based, NOT mAP):")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1:        {f1:.4f}")
    print(f"  TP: {all_tp}, FP: {all_fp}, FN: {all_fn}")


if __name__ == "__main__":
    main()

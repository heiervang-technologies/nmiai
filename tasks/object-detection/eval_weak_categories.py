#!/usr/bin/env python3
"""
Quick evaluation of model performance specifically on weak categories.
Tests how well the model detects rare products using our synthetic test images.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO

BASE = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data")
MODEL_PATH = Path("/home/me/ht/nmiai/runs/detect/yolov8x_v5_b2/weights/best.pt")


def iou(box1, box2):
    """Compute IoU between two [x,y,w,h] boxes."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)

    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = w1 * h1 + w2 * h2 - inter
    return inter / max(union, 1e-6)


def main():
    print("Loading model...")
    model = YOLO(str(MODEL_PATH))

    # Load real annotations to identify weak categories
    with open(BASE / "coco_dataset/train/annotations.json") as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    real_counts = Counter(a["category_id"] for a in coco["annotations"])

    weak_cats = {cid for cid, cnt in real_counts.items() if cnt < 10}
    print(f"Weak categories (<10 annotations): {len(weak_cats)}")

    # Test on clean split val images
    clean_val = BASE / "clean_split" / "val"
    if not clean_val.exists():
        # Fall back to stratified split
        clean_val = BASE / "stratified_split" / "val"

    val_imgs = clean_val / "images"
    val_lbls = clean_val / "labels"

    if not val_imgs.exists():
        print(f"No val images at {val_imgs}")
        return

    img_files = sorted(list(val_imgs.glob("*.jpg")) + list(val_imgs.glob("*.jpeg")) + list(val_imgs.glob("*.png")))
    print(f"Val images: {len(img_files)}")

    # Run inference
    weak_tp = 0
    weak_fn = 0
    weak_fp = 0
    all_tp = 0
    all_fn = 0

    per_cat_tp = Counter()
    per_cat_fn = Counter()

    for img_path in img_files[:50]:  # Limit for speed on CPU
        results = model.predict(str(img_path), device="cpu", verbose=False, conf=0.25)
        r = results[0]

        # Load ground truth
        lbl_path = val_lbls / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        gt_cats = []
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    gt_cats.append(int(parts[0]))

        pred_cats = [int(c) for c in r.boxes.cls] if len(r.boxes) > 0 else []

        # Simple category-level matching
        gt_counter = Counter(gt_cats)
        pred_counter = Counter(pred_cats)

        for cat_id, gt_count in gt_counter.items():
            pred_count = pred_counter.get(cat_id, 0)
            tp = min(gt_count, pred_count)
            fn = gt_count - tp

            if cat_id in weak_cats:
                weak_tp += tp
                weak_fn += fn

            all_tp += tp
            all_fn += fn
            per_cat_tp[cat_id] += tp
            per_cat_fn[cat_id] += fn

    # Results
    weak_recall = weak_tp / max(weak_tp + weak_fn, 1)
    all_recall = all_tp / max(all_tp + all_fn, 1)

    print(f"\n{'='*60}")
    print(f"Overall recall: {all_recall:.3f} ({all_tp}/{all_tp + all_fn})")
    print(f"Weak category recall: {weak_recall:.3f} ({weak_tp}/{weak_tp + weak_fn})")
    print(f"{'='*60}")

    # Worst performing categories
    worst = []
    for cat_id in set(list(per_cat_tp.keys()) + list(per_cat_fn.keys())):
        tp = per_cat_tp[cat_id]
        fn = per_cat_fn[cat_id]
        total = tp + fn
        if total > 0:
            recall = tp / total
            worst.append((recall, cat_id, tp, fn, cat_names.get(cat_id, "?")))

    worst.sort()
    print(f"\nWorst 20 categories by recall:")
    for recall, cid, tp, fn, name in worst[:20]:
        weak_marker = " [WEAK]" if cid in weak_cats else ""
        print(f"  cat{cid:3d}: recall={recall:.2f} ({tp}/{tp+fn}) {name}{weak_marker}")


if __name__ == "__main__":
    main()

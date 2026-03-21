"""
Fast pseudo-labeling using ultralytics YOLO directly on GPU.
Much faster than ONNX on CPU.
"""
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

DATA_DIR = Path(__file__).parent / "data"
STORE_PHOTOS = DATA_DIR / "store_photos"
OUTPUT_DIR = DATA_DIR / "pseudo_labeled_stores"
OUT_IMAGES = OUTPUT_DIR / "images"
OUT_LABELS = OUTPUT_DIR / "labels"

YOLO_PT = Path(__file__).parent.parent / "yolo-approach" / "best.pt"
DET_CONF = 0.25
NMS_IOU = 0.45
MAX_DET = 300
IMG_SIZE = 640  # smaller for CPU speed


def main():
    print("=" * 50)
    print("FAST PSEUDO-LABELING (ultralytics + GPU)")
    print("=" * 50)

    from ultralytics import YOLO

    # Force CPU - GPU is occupied by training
    device = "cpu"
    print(f"Device: {device} (GPU occupied, using CPU)")

    # Load model
    print(f"Loading YOLO model: {YOLO_PT}")
    model = YOLO(str(YOLO_PT))

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    # Get store photos
    store_images = sorted(STORE_PHOTOS.glob("*.jpg")) + sorted(STORE_PHOTOS.glob("*.jpeg")) + sorted(STORE_PHOTOS.glob("*.png"))
    print(f"\nProcessing {len(store_images)} store photos...")

    total_images = 0
    total_annotations = 0

    for img_path in store_images:
        # Run inference
        results = model.predict(
            str(img_path),
            conf=DET_CONF,
            iou=NMS_IOU,
            max_det=MAX_DET,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            print(f"  {img_path.name}: 0 detections")
            continue

        result = results[0]
        boxes = result.boxes

        # Get image dimensions
        img = cv2.imread(str(img_path))
        h_img, w_img = img.shape[:2]

        labels = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls_id = int(box.cls[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())

            # Convert to YOLO format (normalized cx cy w h)
            cx = ((x1 + x2) / 2) / w_img
            cy = ((y1 + y2) / 2) / h_img
            bw = (x2 - x1) / w_img
            bh = (y2 - y1) / h_img

            # Clip
            cx = np.clip(cx, 0, 1)
            cy = np.clip(cy, 0, 1)
            bw = np.clip(bw, 0, 1)
            bh = np.clip(bh, 0, 1)

            # Skip tiny detections
            if bw < 0.005 or bh < 0.005:
                continue

            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        if labels:
            fname = f"pseudo_{img_path.stem}"
            shutil.copy2(str(img_path), str(OUT_IMAGES / f"{fname}.jpg"))
            with open(OUT_LABELS / f"{fname}.txt", "w") as f:
                f.write("\n".join(labels) + "\n")

            total_images += 1
            total_annotations += len(labels)
            print(f"  {img_path.name}: {len(labels)} detections")

    print(f"\n{'=' * 50}")
    print(f"PSEUDO-LABELING COMPLETE")
    print(f"{'=' * 50}")
    print(f"Images labeled: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"Output: {OUTPUT_DIR}")

    # Category distribution
    cat_counts = Counter()
    for f in OUT_LABELS.glob("*.txt"):
        for line in f.read_text().strip().split("\n"):
            if line.strip():
                cat_counts[int(line.split()[0])] += 1

    print(f"\nTop 20 categories in pseudo-labels:")
    for cat_id, count in cat_counts.most_common(20):
        print(f"  cat {cat_id}: {count}")


if __name__ == "__main__":
    main()

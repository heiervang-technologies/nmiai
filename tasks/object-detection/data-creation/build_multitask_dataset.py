"""
Build multitask dataset: combines detection (YOLO boxes) + classification (crops).
For each detection image, also extract crops for classifier training.
Output structure:
  data/multitask_dataset/
    detection/  -> symlink to mega_dataset (YOLO format)
    classification/  -> ImageFolder with crops
    metadata.json -> links crops to detections
"""
import json
import random
from pathlib import Path
from collections import Counter
import cv2

DATA_DIR = Path(__file__).parent / "data"
MEGA_DIR = DATA_DIR / "mega_dataset"
OUTPUT_DIR = DATA_DIR / "multitask_dataset"
SEED = 42

random.seed(SEED)


def extract_crops_from_detection(img_path, label_path, output_dir, max_per_image=20):
    """Extract classification crops from a detection image + labels."""
    img = cv2.imread(str(img_path))
    if img is None:
        return []

    h, w = img.shape[:2]
    crops = []

    lines = label_path.read_text().strip().split("\n") if label_path.exists() else []
    for i, line in enumerate(lines[:max_per_image]):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

        # Convert normalized to pixel
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))

        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        # Add small padding
        pad = max(5, int(min(x2 - x1, y2 - y1) * 0.05))
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = img[y1:y2, x1:x2]

        cat_dir = output_dir / str(cls_id)
        cat_dir.mkdir(exist_ok=True)
        crop_name = f"{img_path.stem}_crop_{i}.jpg"
        crop_path = cat_dir / crop_name

        if not crop_path.exists():
            cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])

        crops.append({
            "crop_path": str(crop_path),
            "source_image": img_path.name,
            "category_id": cls_id,
            "bbox_norm": [cx, cy, bw, bh],
        })

    return crops


def main():
    print("Building multitask dataset...")

    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Symlink detection to mega
    det_link = OUTPUT_DIR / "detection"
    if not det_link.exists():
        det_link.symlink_to(MEGA_DIR.resolve())
    print(f"Detection: -> {MEGA_DIR}")

    # Extract crops from mega dataset images
    cls_dir = OUTPUT_DIR / "classification"
    for split in ["train", "val"]:
        (cls_dir / split).mkdir(parents=True, exist_ok=True)

    all_crops = []
    img_dir = MEGA_DIR / "train" / "images"
    lbl_dir = MEGA_DIR / "train" / "labels"

    # Process a sample of images (not all - would be too many crops)
    images = sorted(img_dir.iterdir())
    # Take every Nth image to get good coverage
    step = max(1, len(images) // 2000)
    selected = images[::step]

    print(f"Extracting crops from {len(selected)} of {len(images)} training images...")

    for i, img_path in enumerate(selected):
        if not img_path.name.endswith((".jpg", ".jpeg", ".png")):
            continue

        real = img_path.resolve() if img_path.is_symlink() else img_path
        lbl_path = lbl_dir / (img_path.stem + ".txt")

        crops = extract_crops_from_detection(real, lbl_path, cls_dir / "train")
        all_crops.extend(crops)

        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(selected)} images, {len(all_crops)} crops")

    # Val crops
    val_crops = []
    for img_path in sorted((MEGA_DIR / "val" / "images").iterdir()):
        if not img_path.name.endswith((".jpg", ".jpeg", ".png")):
            continue
        real = img_path.resolve() if img_path.is_symlink() else img_path
        lbl_path = MEGA_DIR / "val" / "labels" / (img_path.stem + ".txt")
        crops = extract_crops_from_detection(real, lbl_path, cls_dir / "val")
        val_crops.extend(crops)

    # Save metadata
    metadata = {
        "train_crops": len(all_crops),
        "val_crops": len(val_crops),
        "detection_path": str(MEGA_DIR),
        "classification_path": str(cls_dir),
    }
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Count categories
    cat_counts = Counter()
    for c in all_crops:
        cat_counts[c["category_id"]] += 1

    print(f"\nMULTITASK DATASET COMPLETE:")
    print(f"  Detection: {len(images)} train / 80 val")
    print(f"  Classification crops: {len(all_crops)} train / {len(val_crops)} val")
    print(f"  Categories covered: {len(cat_counts)}")
    print(f"  Path: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

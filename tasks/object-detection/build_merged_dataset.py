#!/usr/bin/env python3
"""
Build a merged YOLO dataset combining real + synthetic training data.
Output is a single YOLO-format dataset ready for training.
"""

import json
import shutil
from pathlib import Path
from collections import Counter

BASE_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data")
SYNTHETIC_DIR = BASE_DIR / "synthetic_dataset"
SYNTHETIC_ANNOT = SYNTHETIC_DIR / "annotations.json"
SYNTHETIC_IMAGES = SYNTHETIC_DIR / "images"

# Find existing YOLO dataset (check common locations)
EXISTING_YOLO_CANDIDATES = [
    BASE_DIR / "mega_dataset",
    BASE_DIR / "yolo_v5_dataset",
    BASE_DIR / "yolo_v4_dataset",
    BASE_DIR / "yolo_dataset",
    BASE_DIR / "augmented",
]

# Output
MERGED_DIR = BASE_DIR / "merged_with_synthetic"
MERGED_IMAGES_TRAIN = MERGED_DIR / "images" / "train"
MERGED_LABELS_TRAIN = MERGED_DIR / "labels" / "train"
MERGED_IMAGES_VAL = MERGED_DIR / "images" / "val"
MERGED_LABELS_VAL = MERGED_DIR / "labels" / "val"


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """Convert COCO bbox [x, y, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = max(0, min(1, (x + w / 2) / img_w))
    cy = max(0, min(1, (y + h / 2) / img_h))
    nw = max(0, min(1, w / img_w))
    nh = max(0, min(1, h / img_h))
    return cx, cy, nw, nh


def find_existing_yolo():
    """Find the best existing YOLO dataset to merge with.

    Handles both structures:
      - standard: images/train/, labels/train/
      - mega: train/images/, train/labels/
    """
    for candidate in EXISTING_YOLO_CANDIDATES:
        if not candidate.exists():
            continue
        # Try mega structure first: train/images/
        for img_dir, lbl_dir in [
            (candidate / "train" / "images", candidate / "train" / "labels"),
            (candidate / "images" / "train", candidate / "labels" / "train"),
        ]:
            if img_dir.exists() and lbl_dir.exists():
                n_imgs = len(list(img_dir.glob("*.jp*")) + list(img_dir.glob("*.png")))
                if n_imgs > 0:
                    print(f"Found existing YOLO dataset: {candidate} ({n_imgs} train images)")
                    return candidate, img_dir, lbl_dir
    return None, None, None


def convert_synthetic_coco_to_yolo():
    """Convert synthetic COCO annotations to YOLO label files."""
    with open(SYNTHETIC_ANNOT) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    results = {}  # filename -> list of yolo lines
    for img_id, img_info in images.items():
        fname = img_info["file_name"]
        if fname.startswith("bg_"):
            continue

        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue

        lines = []
        for ann in anns:
            cx, cy, nw, nh = coco_to_yolo_bbox(
                ann["bbox"], img_info["width"], img_info["height"]
            )
            if nw < 0.005 or nh < 0.005:
                continue
            lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if lines:
            results[fname] = lines

    return results


def main():
    print("=" * 60)
    print("Building merged real + synthetic YOLO dataset")
    print("=" * 60)

    # Create output dirs
    for d in [MERGED_IMAGES_TRAIN, MERGED_LABELS_TRAIN, MERGED_IMAGES_VAL, MERGED_LABELS_VAL]:
        d.mkdir(parents=True, exist_ok=True)

    real_count = 0
    synth_count = 0

    # Step 1: Symlink existing YOLO data (faster than copy for 10k+ images)
    existing, train_img_dir, train_lbl_dir = find_existing_yolo()
    if existing:
        # Symlink train images and labels
        for img_path in list(train_img_dir.glob("*.jp*")) + list(train_img_dir.glob("*.png")):
            dst = MERGED_IMAGES_TRAIN / img_path.name
            if not dst.exists():
                dst.symlink_to(img_path.resolve())
            label_path = train_lbl_dir / (img_path.stem + ".txt")
            if label_path.exists():
                dst_lbl = MERGED_LABELS_TRAIN / label_path.name
                if not dst_lbl.exists():
                    dst_lbl.symlink_to(label_path.resolve())
            real_count += 1

        # Symlink val (check both structures)
        for val_dir in [existing / "val", existing / "images" / "val"]:
            val_img = val_dir / "images" if (val_dir / "images").exists() else val_dir
            val_lbl_parent = val_dir / "labels" if (val_dir / "labels").exists() else existing / "labels" / "val"
            if val_img.exists():
                for img_path in list(val_img.glob("*.jp*")) + list(val_img.glob("*.png")):
                    dst = MERGED_IMAGES_VAL / img_path.name
                    if not dst.exists():
                        dst.symlink_to(img_path.resolve())
                    label_path = val_lbl_parent / (img_path.stem + ".txt")
                    if label_path.exists():
                        dst_lbl = MERGED_LABELS_VAL / label_path.name
                        if not dst_lbl.exists():
                            dst_lbl.symlink_to(label_path.resolve())
                break

        print(f"Linked {real_count} real training images")
    else:
        print("WARNING: No existing YOLO dataset found! Synthetic-only dataset.")

    # Step 2: Convert and add synthetic data
    print("Converting synthetic annotations to YOLO format...")
    synth_labels = convert_synthetic_coco_to_yolo()

    for fname, lines in synth_labels.items():
        src_img = SYNTHETIC_IMAGES / fname
        if not src_img.exists():
            continue

        # Prefix synthetic filenames to avoid collision
        dst_name = f"synth_{fname}" if not fname.startswith("synth_") else fname
        shutil.copy2(src_img, MERGED_IMAGES_TRAIN / dst_name)

        label_name = Path(dst_name).stem + ".txt"
        with open(MERGED_LABELS_TRAIN / label_name, "w") as f:
            f.write("\n".join(lines))

        synth_count += 1

    # Step 3: Write YAML config
    yaml_path = MERGED_DIR / "data.yaml"

    # Read class names from existing config
    class_yaml = BASE_DIR / "yolo_all_data.yaml"
    class_section = ""
    if class_yaml.exists():
        with open(class_yaml) as f:
            class_section = f.read()

    with open(yaml_path, "w") as f:
        f.write(f"path: {MERGED_DIR}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("\n")
        f.write(class_section)

    # Stats
    total_train = real_count + synth_count
    print(f"\n{'=' * 60}")
    print(f"Merged dataset ready at: {MERGED_DIR}")
    print(f"  Real training images: {real_count}")
    print(f"  Synthetic training images: {synth_count}")
    print(f"  Total training images: {total_train}")
    print(f"  YAML config: {yaml_path}")

    # Category distribution in synthetic
    cat_counts = Counter()
    for lines in synth_labels.values():
        for line in lines:
            cat_id = int(line.split()[0])
            cat_counts[cat_id] += 1

    print(f"  Synthetic annotations: {sum(cat_counts.values())}")
    print(f"  Categories in synthetic: {len(cat_counts)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

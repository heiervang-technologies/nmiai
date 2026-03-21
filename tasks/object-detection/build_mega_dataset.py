#!/usr/bin/env python3
"""
Build MEGA DATASET for V8 training.

Sources (in priority order):
1. silver_augmented_dataset (6,440 images) - base + silver copy-paste + boosted + augmented
2. Remaining 760 video frames (pseudo-labeled when GPU available)
3. Remaining 18 store photos (pseudo-labeled when GPU available)

Val: 80-image large_clean_split val (NEVER touches training)

This script:
- Starts with silver_augmented_dataset as base (already a superset of large_clean_split)
- Adds pseudo-labeled video frames and store photos as they become available
- Verifies ZERO overlap with val set
- Produces dataset.yaml for YOLO training
- Reports per-category annotation counts

Usage:
  python build_mega_dataset.py                    # Build from existing silver
  python build_mega_dataset.py --add-pseudolabels # Also add pseudo-labeled frames
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

BASE = Path(__file__).parent / "data-creation" / "data"
SILVER = BASE / "silver_augmented_dataset"
LARGE_VAL = BASE / "large_clean_split" / "val"
VAL_LIST = BASE / "large_clean_split" / "val_images.txt"
MEGA_OUT = BASE / "mega_dataset"
PSEUDO_LABELS_DIR = BASE / "pseudo_labels"  # Where pseudo-labeling outputs go
VIDEO_FRAMES = BASE / "store_photos" / "video_frames"
STORE_PHOTOS = BASE / "store_photos"

DATA_YAML_TEMPLATE = """path: {path}
train: train/images
val: val/images

nc: 356
names:
"""


def load_val_images():
    """Load val image filenames to prevent overlap."""
    val_images = set()
    if VAL_LIST.exists():
        with open(VAL_LIST) as f:
            for line in f:
                val_images.add(line.strip())
    # Also get from directory
    val_dir = LARGE_VAL / "images"
    if val_dir.exists():
        for p in val_dir.iterdir():
            val_images.add(p.name)
    return val_images


def load_class_names():
    yaml_path = Path(__file__).parent / "yolo-approach" / "dataset" / "data.yaml"
    names = {}
    with open(yaml_path) as f:
        in_names = False
        for line in f:
            line = line.rstrip()
            if line.startswith("names:"):
                in_names = True
                continue
            if in_names and ":" in line:
                parts = line.strip().split(":", 1)
                try:
                    idx = int(parts[0].strip())
                    name = parts[1].strip().strip("'\"")
                    names[idx] = name
                except ValueError:
                    pass
    return names


def count_labels(label_dir):
    """Count annotations per class from YOLO label files."""
    counts = Counter()
    total = 0
    for f in label_dir.iterdir():
        if f.suffix != '.txt':
            continue
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split()
                if parts:
                    try:
                        counts[int(parts[0])] += 1
                        total += 1
                    except ValueError:
                        pass
    return counts, total


def copy_with_dedup(src_img_dir, src_label_dir, dst_img_dir, dst_label_dir,
                     val_images, prefix="", existing_files=None):
    """Copy images and labels, skipping val images and duplicates."""
    if existing_files is None:
        existing_files = set()

    added = 0
    skipped_val = 0
    skipped_dup = 0

    for img_path in sorted(src_img_dir.iterdir()):
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue

        # Check val overlap
        base_name = img_path.name
        if base_name in val_images:
            skipped_val += 1
            continue

        # Generate target name
        if prefix:
            target_name = f"{prefix}_{base_name}"
        else:
            target_name = base_name

        # Check duplicates
        if target_name in existing_files:
            skipped_dup += 1
            continue

        # Copy image
        dst_img = dst_img_dir / target_name
        shutil.copy2(img_path, dst_img)

        # Copy label if exists
        label_name = img_path.stem + ".txt"
        src_label = src_label_dir / label_name
        if src_label.exists():
            if prefix:
                dst_label_name = f"{prefix}_{label_name}"
            else:
                dst_label_name = label_name
            shutil.copy2(src_label, dst_label_dir / dst_label_name)

        existing_files.add(target_name)
        added += 1

    return added, skipped_val, skipped_dup


def build_mega(add_pseudolabels=False):
    val_images = load_val_images()
    class_names = load_class_names()

    print(f"Val images to exclude: {len(val_images)}")

    # Create output dirs
    mega_train_img = MEGA_OUT / "train" / "images"
    mega_train_lbl = MEGA_OUT / "train" / "labels"
    mega_val_img = MEGA_OUT / "val" / "images"
    mega_val_lbl = MEGA_OUT / "val" / "labels"

    for d in [mega_train_img, mega_train_lbl, mega_val_img, mega_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    existing_files = set()

    # ============================================================
    # STEP 1: Copy silver_augmented_dataset as base
    # ============================================================
    print(f"\n=== STEP 1: Silver augmented dataset ===")
    silver_img = SILVER / "train" / "images"
    silver_lbl = SILVER / "train" / "labels"

    if silver_img.exists():
        added, sv, sd = copy_with_dedup(
            silver_img, silver_lbl, mega_train_img, mega_train_lbl,
            val_images, prefix="", existing_files=existing_files
        )
        print(f"  Added: {added}, skipped_val: {sv}, skipped_dup: {sd}")
    else:
        print("  WARNING: Silver dataset not found!")

    # ============================================================
    # STEP 2: Add pseudo-labeled data (if available and requested)
    # ============================================================
    if add_pseudolabels:
        print(f"\n=== STEP 2: Pseudo-labeled data ===")

        # Check for pseudo-labeled video frames
        pl_vframes_img = PSEUDO_LABELS_DIR / "video_frames" / "images"
        pl_vframes_lbl = PSEUDO_LABELS_DIR / "video_frames" / "labels"
        if pl_vframes_img.exists():
            added, sv, sd = copy_with_dedup(
                pl_vframes_img, pl_vframes_lbl, mega_train_img, mega_train_lbl,
                val_images, prefix="plvf", existing_files=existing_files
            )
            print(f"  Video frames: added={added}, skipped_val={sv}, skipped_dup={sd}")
        else:
            print(f"  Video frames pseudo-labels not yet available at {pl_vframes_img}")

        # Check for pseudo-labeled store photos
        pl_store_img = PSEUDO_LABELS_DIR / "store_photos" / "images"
        pl_store_lbl = PSEUDO_LABELS_DIR / "store_photos" / "labels"
        if pl_store_img.exists():
            added, sv, sd = copy_with_dedup(
                pl_store_img, pl_store_lbl, mega_train_img, mega_train_lbl,
                val_images, prefix="plsp", existing_files=existing_files
            )
            print(f"  Store photos: added={added}, skipped_val={sv}, skipped_dup={sd}")
        else:
            print(f"  Store photo pseudo-labels not yet available at {pl_store_img}")
    else:
        print(f"\n=== STEP 2: Skipping pseudo-labels (use --add-pseudolabels) ===")

    # ============================================================
    # STEP 3: Copy val set (from large_clean_split)
    # ============================================================
    print(f"\n=== STEP 3: Val set (large_clean_split) ===")
    val_src_img = LARGE_VAL / "images"
    val_src_lbl = LARGE_VAL / "labels"
    if val_src_img.exists():
        val_count = 0
        for img_path in sorted(val_src_img.iterdir()):
            if img_path.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                shutil.copy2(img_path, mega_val_img / img_path.name)
                lbl = val_src_lbl / (img_path.stem + ".txt")
                if lbl.exists():
                    shutil.copy2(lbl, mega_val_lbl / lbl.name)
                val_count += 1
        print(f"  Val images: {val_count}")

    # ============================================================
    # STEP 4: Verify zero overlap
    # ============================================================
    print(f"\n=== STEP 4: Overlap verification ===")
    train_files = set(p.name for p in mega_train_img.iterdir())
    val_files = set(p.name for p in mega_val_img.iterdir())
    overlap = train_files & val_files
    if overlap:
        print(f"  !!! OVERLAP DETECTED: {len(overlap)} files !!!")
        for f in sorted(overlap)[:10]:
            print(f"    {f}")
        # Remove overlapping files from train
        for f in overlap:
            (mega_train_img / f).unlink()
            lbl = mega_train_lbl / (Path(f).stem + ".txt")
            if lbl.exists():
                lbl.unlink()
        print(f"  Removed {len(overlap)} overlapping files from train")
    else:
        print(f"  CLEAN: Zero overlap between train and val")

    # ============================================================
    # STEP 5: Generate dataset.yaml
    # ============================================================
    print(f"\n=== STEP 5: Dataset YAML ===")
    yaml_content = DATA_YAML_TEMPLATE.format(path=str(MEGA_OUT))
    for i in range(356):
        name = class_names.get(i, f"class_{i}")
        yaml_content += f'  {i}: "{name}"\n'

    yaml_path = MEGA_OUT / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"  Written: {yaml_path}")

    # ============================================================
    # STEP 6: Final statistics
    # ============================================================
    print(f"\n=== FINAL STATISTICS ===")
    final_train = len(list(mega_train_img.iterdir()))
    final_val = len(list(mega_val_img.iterdir()))
    print(f"Train images: {final_train}")
    print(f"Val images: {final_val}")

    train_counts, total_anns = count_labels(mega_train_lbl)
    print(f"Total train annotations: {total_anns}")

    # Per-category stats
    zero_cats = [i for i in range(356) if train_counts.get(i, 0) == 0]
    low_cats = [i for i in range(356) if 0 < train_counts.get(i, 0) < 50]
    ok_cats = [i for i in range(356) if 50 <= train_counts.get(i, 0) < 200]
    good_cats = [i for i in range(356) if train_counts.get(i, 0) >= 200]

    print(f"\nCategory coverage:")
    print(f"  Zero annotations: {len(zero_cats)}")
    print(f"  < 50 annotations: {len(low_cats)}")
    print(f"  50-199 annotations: {len(ok_cats)}")
    print(f"  >= 200 annotations: {len(good_cats)}")

    if zero_cats:
        print(f"\n  Categories with ZERO annotations:")
        for c in zero_cats:
            print(f"    [{c}] {class_names.get(c, '?')}")

    # Bottom 20 categories
    sorted_cats = sorted(range(356), key=lambda c: train_counts.get(c, 0))
    print(f"\n  Bottom 20 categories:")
    for c in sorted_cats[:20]:
        print(f"    [{c}] {train_counts.get(c, 0):>5} annotations - {class_names.get(c, '?')}")

    # Save manifest
    manifest = {
        "train_images": final_train,
        "val_images": final_val,
        "total_annotations": total_anns,
        "category_counts": {str(k): v for k, v in train_counts.items()},
        "zero_categories": zero_cats,
        "dataset_yaml": str(yaml_path),
        "overlap_check": "CLEAN" if not overlap else f"REMOVED {len(overlap)}",
    }
    manifest_path = MEGA_OUT / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}")
    print(f"Dataset YAML: {yaml_path}")
    print(f"\nReady for training:")
    print(f"  yolo train data={yaml_path} model=yolov8x.pt epochs=100 imgsz=1280 batch=4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--add-pseudolabels", action="store_true",
                        help="Include pseudo-labeled video frames and store photos")
    args = parser.parse_args()
    build_mega(add_pseudolabels=args.add_pseudolabels)

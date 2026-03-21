"""
Build MEGA unified dataset combining ALL data sources.
Zero data leakage - uses large_clean_split val (80 images).

Sources combined into training:
1. Original COCO train images (from large_clean_split)
2. All augmented data (boosted_weak, edge_case, mosaic, mixup, scale)
3. Silver copy-paste synthetic data
4. V7 targeted augmentations for weak categories
5. Pseudo-labeled store photos (when available)

Val: 80 stratified images from large_clean_split (NEVER touched)
"""
import json
import shutil
from pathlib import Path
from collections import Counter
import yaml

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "mega_dataset"
VAL_SOURCE = DATA_DIR / "large_clean_split"


def main():
    print("Building MEGA unified dataset...")

    # Setup output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Load val stems for leakage prevention
    val_stems = set()
    for img in (VAL_SOURCE / "val" / "images").iterdir():
        val_stems.add(img.stem)
    print(f"Val holdout: {len(val_stems)} images (protected)")

    # Copy val as-is
    val_count = 0
    for img in (VAL_SOURCE / "val" / "images").iterdir():
        dst = OUTPUT_DIR / "val" / "images" / img.name
        if not dst.exists():
            real = img.resolve() if img.is_symlink() else img
            dst.symlink_to(real)
        lsrc = VAL_SOURCE / "val" / "labels" / (img.stem + ".txt")
        ldst = OUTPUT_DIR / "val" / "labels" / (img.stem + ".txt")
        if lsrc.exists() and not ldst.exists():
            shutil.copy2(lsrc, ldst)
        val_count += 1
    print(f"Val: {val_count} images")

    # Gather all training sources
    sources = []

    # 1. Original COCO train from large_clean_split
    sources.append(("large_clean_split", VAL_SOURCE / "train"))

    # 2. Silver copy-paste
    silver_dir = DATA_DIR / "silver_copypaste"
    if (silver_dir / "images").exists():
        sources.append(("silver_copypaste", silver_dir))

    # 3. V7 augmentations (only the v7aug_ files, originals already in large_clean_split)
    v7_dir = DATA_DIR / "yolo_v7_balanced" / "train"
    if (v7_dir / "images").exists():
        sources.append(("v7_balanced_aug", v7_dir))

    # 4. Additional augmented sources not already in large_clean_split
    for aug_name in ["boosted_weak", "edge_case_augment", "mosaic_augment", "mixup_augment", "scale_augment"]:
        aug_dir = DATA_DIR / aug_name
        if aug_dir.exists():
            # These have flat structure (images directly in dir or images/ subdir)
            if (aug_dir / "images").exists():
                sources.append((aug_name, aug_dir))

    # 5. Pseudo-labeled store data (if available)
    ext_val = DATA_DIR / "external_val"
    if (ext_val / "val" / "images").exists():
        sources.append(("external_pseudo", ext_val / "val"))

    # Track what we add
    added_stems = set()
    train_count = 0
    source_counts = Counter()

    for source_name, source_dir in sources:
        img_dir = source_dir / "images"
        lbl_dir = source_dir / "labels"

        if not img_dir.exists():
            print(f"  Skip {source_name}: no images dir")
            continue

        count = 0
        for img_path in img_dir.iterdir():
            if not img_path.name.endswith((".jpg", ".jpeg", ".png")):
                continue

            stem = img_path.stem

            # Skip val images (leakage prevention)
            if stem in val_stems:
                continue

            # For v7_balanced_aug, only take the augmented files
            if source_name == "v7_balanced_aug" and not stem.startswith("v7aug_"):
                continue

            # Skip duplicates
            if stem in added_stems:
                continue

            # Check label exists
            label_path = lbl_dir / (stem + ".txt")
            if not label_path.exists():
                continue

            # Add image
            dst_img = OUTPUT_DIR / "train" / "images" / img_path.name
            if not dst_img.exists():
                real = img_path.resolve() if img_path.is_symlink() else img_path
                try:
                    dst_img.symlink_to(real)
                except (OSError, FileExistsError):
                    continue

            # Add label
            dst_lbl = OUTPUT_DIR / "train" / "labels" / (stem + ".txt")
            if not dst_lbl.exists():
                shutil.copy2(label_path, dst_lbl)

            added_stems.add(stem)
            count += 1
            train_count += 1

        source_counts[source_name] = count
        print(f"  {source_name}: +{count} images")

    # Write dataset.yaml
    ann = json.load(open(DATA_DIR / "coco_dataset" / "train" / "annotations.json"))
    names = {c["id"]: c["name"] for c in ann["categories"]}

    yaml_data = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names,
    }
    with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
        f.write("# MEGA unified dataset - all sources combined\n")
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Stats
    final_counts = Counter()
    for lf in (OUTPUT_DIR / "train" / "labels").glob("*.txt"):
        for line in open(lf):
            parts = line.strip().split()
            if parts:
                final_counts[int(parts[0])] += 1

    vals = sorted(final_counts.values())
    print(f"\nMEGA DATASET COMPLETE:")
    print(f"  Train: {train_count} images")
    print(f"  Val: {val_count} images")
    print(f"  Total annotations: {sum(vals)}")
    print(f"  Categories: {len(final_counts)}")
    print(f"  Min ann/cat: {vals[0]}")
    print(f"  Median ann/cat: {vals[len(vals)//2]}")
    print(f"  Max ann/cat: {vals[-1]}")
    print(f"  dataset.yaml: {OUTPUT_DIR / 'dataset.yaml'}")

    print(f"\nSource breakdown:")
    for name, count in source_counts.most_common():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()

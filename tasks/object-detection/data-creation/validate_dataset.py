"""
Validate any YOLO dataset for quality issues before training.
Usage: python validate_dataset.py <dataset_yaml>
"""
import json
import sys
from pathlib import Path
from collections import Counter
import yaml


def validate(yaml_path):
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    base_dir = Path(config["path"])
    train_dir = base_dir / config["train"]
    val_dir = base_dir / config["val"]
    nc = config["nc"]
    names = config["names"]

    issues = []
    warnings = []

    print(f"Validating: {yaml_path}")
    print(f"  Base: {base_dir}")
    print(f"  Classes: {nc}")

    # Check directories exist
    if not train_dir.exists():
        issues.append(f"Train dir missing: {train_dir}")
    if not val_dir.exists():
        issues.append(f"Val dir missing: {val_dir}")

    # Check images
    train_imgs = list(train_dir.glob("*")) if train_dir.exists() else []
    val_imgs = list(val_dir.glob("*")) if val_dir.exists() else []
    train_imgs = [f for f in train_imgs if f.suffix in (".jpg", ".jpeg", ".png")]
    val_imgs = [f for f in val_imgs if f.suffix in (".jpg", ".jpeg", ".png")]

    print(f"  Train images: {len(train_imgs)}")
    print(f"  Val images: {len(val_imgs)}")

    # Check labels
    train_label_dir = base_dir / config["train"].replace("images", "labels")
    val_label_dir = base_dir / config["val"].replace("images", "labels")

    # Validate labels
    for split_name, img_list, lbl_dir in [("train", train_imgs, train_label_dir), ("val", val_imgs, val_label_dir)]:
        counts = Counter()
        missing_labels = 0
        bad_labels = 0
        broken_symlinks = 0
        oob_boxes = 0

        for img_path in img_list:
            # Check broken symlinks
            if img_path.is_symlink() and not img_path.resolve().exists():
                broken_symlinks += 1
                continue

            label_path = lbl_dir / (img_path.stem + ".txt")
            if not label_path.exists():
                missing_labels += 1
                continue

            for line_no, line in enumerate(open(label_path), 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    bad_labels += 1
                    continue

                try:
                    cls = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                except ValueError:
                    bad_labels += 1
                    continue

                counts[cls] += 1

                # Check bounds
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    oob_boxes += 1

        print(f"\n  {split_name}:")
        print(f"    Broken symlinks: {broken_symlinks}")
        print(f"    Missing labels: {missing_labels}")
        print(f"    Bad label format: {bad_labels}")
        print(f"    Out-of-bounds boxes: {oob_boxes}")
        print(f"    Categories found: {len(counts)}/{nc}")
        print(f"    Total annotations: {sum(counts.values())}")

        if broken_symlinks:
            issues.append(f"{split_name}: {broken_symlinks} broken symlinks")
        if missing_labels > len(img_list) * 0.1:
            warnings.append(f"{split_name}: {missing_labels} missing labels ({missing_labels/len(img_list)*100:.0f}%)")
        if bad_labels:
            issues.append(f"{split_name}: {bad_labels} bad label lines")

        # Check for leakage
        if split_name == "train" and val_imgs:
            val_stems = {p.stem for p in val_imgs}
            train_stems = {p.stem for p in img_list}
            leaked = val_stems & train_stems
            if leaked:
                issues.append(f"DATA LEAKAGE: {len(leaked)} val stems in train!")

        # Class distribution
        vals = sorted(counts.values()) if counts else [0]
        if vals:
            ratio = vals[-1] / vals[0] if vals[0] > 0 else float("inf")
            print(f"    Min ann/cat: {vals[0]}")
            print(f"    Max ann/cat: {vals[-1]}")
            print(f"    Balance ratio: {ratio:.1f}x")
            if ratio > 50:
                warnings.append(f"{split_name}: high imbalance {ratio:.0f}x")

        missing_cats = set(range(nc)) - set(counts.keys())
        if missing_cats and split_name == "val":
            warnings.append(f"val missing {len(missing_cats)} categories")

    # Summary
    print(f"\n{'='*40}")
    if issues:
        print(f"ISSUES ({len(issues)}):")
        for i in issues:
            print(f"  [!] {i}")
    if warnings:
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [~] {w}")
    if not issues and not warnings:
        print("ALL CHECKS PASSED!")

    return len(issues) == 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_dataset.py <dataset.yaml>")
        sys.exit(1)
    ok = validate(sys.argv[1])
    sys.exit(0 if ok else 1)

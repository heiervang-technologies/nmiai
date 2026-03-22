#!/usr/bin/env python3
"""Build a 1000-sample pre-training subset from each external dataset.

Creates a unified YOLO-format dataset for fast pre-training experiments.
All categories mapped to single "product" class (detection-only transfer).

Usage: python build_pretrain_subset.py
"""
import json
import random
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "data-creation" / "data"
OUTPUT = BASE / "pretrain_subset"
COCO_ANN = BASE / "coco_dataset" / "train" / "annotations.json"
SAMPLES_PER_DATASET = 1000


def coco_to_yolo_subset(name, ann_path, img_dir, out_images, out_labels, n=SAMPLES_PER_DATASET):
    """Convert n random images from a COCO dataset to YOLO format (single class)."""
    with open(ann_path) as f:
        coco = json.load(f)

    img_lookup = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    from collections import defaultdict
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    # Only images that have annotations AND exist on disk
    valid_ids = []
    for img_id, img_info in img_lookup.items():
        if img_id not in anns_by_img:
            continue
        fname = img_info["file_name"]
        # Check multiple extensions
        img_path = img_dir / fname
        if not img_path.exists():
            # Try without extension changes
            for ext in [".jpg", ".jpeg", ".png"]:
                alt = img_dir / (Path(fname).stem + ext)
                if alt.exists():
                    img_path = alt
                    break
        if img_path.exists():
            valid_ids.append((img_id, img_path))

    random.shuffle(valid_ids)
    selected = valid_ids[:n]

    count_imgs = 0
    count_anns = 0

    for img_id, img_path in selected:
        img_info = img_lookup[img_id]
        w, h = img_info["width"], img_info["height"]
        anns = anns_by_img[img_id]

        # Convert to YOLO format (single class = 0)
        yolo_lines = []
        for ann in anns:
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0.001, min(1, nw))
            nh = max(0.001, min(1, nh))
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            count_anns += 1

        if not yolo_lines:
            continue

        # Use prefix to avoid name collisions
        dst_name = f"{name}_{img_path.name}"
        shutil.copy2(img_path, out_images / dst_name)

        label_name = f"{name}_{img_path.stem}.txt"
        with open(out_labels / label_name, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")
        count_imgs += 1

    print(f"  {name}: {count_imgs} images, {count_anns} annotations (from {len(valid_ids)} available)")
    return count_imgs, count_anns


def build_val_from_competition(out_images, out_labels):
    """Convert all 248 competition images as val set."""
    with open(COCO_ANN) as f:
        coco = json.load(f)

    # Load category names for the YAML
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    img_lookup = {img["id"]: img for img in coco["images"]}
    from collections import defaultdict
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    img_dir = BASE / "coco_dataset" / "train" / "images"
    count = 0
    for img_id, img_info in img_lookup.items():
        img_path = img_dir / img_info["file_name"]
        if not img_path.exists():
            continue
        w, h = img_info["width"], img_info["height"]

        # For val, keep original categories (not single-class)
        yolo_lines = []
        for ann in anns_by_img.get(img_id, []):
            bx, by, bw, bh = ann["bbox"]
            cat = ann["category_id"]
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            yolo_lines.append(f"{cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if yolo_lines:
            shutil.copy2(img_path, out_images / img_info["file_name"])
            with open(out_labels / (Path(img_info["file_name"]).stem + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines) + "\n")
            count += 1

    print(f"  Val: {count} images (all competition data)")
    return count, cat_names


def main():
    random.seed(42)

    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)

    train_imgs = OUTPUT / "train" / "images"
    train_lbls = OUTPUT / "train" / "labels"
    val_imgs = OUTPUT / "val" / "images"
    val_lbls = OUTPUT / "val" / "labels"

    for d in [train_imgs, train_lbls, val_imgs, val_lbls]:
        d.mkdir(parents=True)

    print("Building pre-training subset...")
    total_imgs = 0
    total_anns = 0

    # External datasets
    sources = [
        {
            "name": "polish",
            "annotations": BASE / "external" / "skus_on_shelves_pl" / "extracted" / "annotations.json",
            "images": BASE / "external" / "skus_on_shelves_pl" / "extracted",
        },
        {
            "name": "grocery",
            "annotations": BASE / "external" / "grocery_shelves" / "coco_annotations.json",
            "images": BASE / "external" / "grocery_shelves" / "Supermarket shelves" / "Supermarket shelves" / "images",
        },
    ]

    # SKU-110K (extracted)
    sku_extracted = BASE / "external" / "sku110k_extracted"
    if (sku_extracted / "annotations.json").exists():
        sources.append({
            "name": "sku110k",
            "annotations": sku_extracted / "annotations.json",
            "images": sku_extracted / "images",
        })

    for src in sources:
        if not Path(src["annotations"]).exists():
            print(f"  Skipping {src['name']} (annotations not found)")
            continue
        ni, na = coco_to_yolo_subset(src["name"], src["annotations"], src["images"],
                                      train_imgs, train_lbls, n=SAMPLES_PER_DATASET)
        total_imgs += ni
        total_anns += na

    # Build val set
    val_count, cat_names = build_val_from_competition(val_imgs, val_lbls)

    # Write dataset.yaml - single class for training (detection only)
    yaml_content = f"path: {OUTPUT}\ntrain: train/images\nval: val/images\n\n"
    yaml_content += "# Training uses single class (detection transfer)\n"
    yaml_content += "# Val uses all 356 categories for honest evaluation\n"
    yaml_content += "nc: 1\nnames:\n  0: product\n"

    with open(OUTPUT / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    # Also write a 356-class val yaml for proper evaluation
    val_yaml = f"path: {OUTPUT}\ntrain: train/images\nval: val/images\n\nnc: 356\nnames:\n"
    for i in range(356):
        name = cat_names.get(i, f"class_{i}")
        val_yaml += f'  {i}: "{name}"\n'
    with open(OUTPUT / "dataset_356cls.yaml", "w") as f:
        f.write(val_yaml)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Pre-training subset built:")
    print(f"  Train: {total_imgs} images, {total_anns} annotations (single class)")
    print(f"  Val: {val_count} images (356 classes, all competition)")
    print(f"  Output: {OUTPUT}")
    print(f"  YAML: {OUTPUT / 'dataset.yaml'}")
    print(f"{'=' * 60}")

    # Verify no leakage
    train_files = {p.name for p in train_imgs.glob("*")}
    val_files = {p.name for p in val_imgs.glob("*")}
    overlap = train_files & val_files
    if overlap:
        print(f"*** LEAKAGE: {len(overlap)} overlapping files! ***")
    else:
        print("Leakage check: ZERO overlap (PASS)")


if __name__ == "__main__":
    main()

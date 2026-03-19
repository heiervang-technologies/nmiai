"""
Build yolo_augmented_v3: our augmented v2 data + matched PL shelf images.

- Our v2 train (1698 images) stays as-is
- PL images whose categories map to our 356 categories get added to train
- Val stays unchanged (50 original images)
- Uses yaml.dump() for proper YAML escaping
"""
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

DATA_DIR = Path(__file__).parent / "data"
V2_DIR = DATA_DIR / "yolo_augmented_v2"
PL_DIR = DATA_DIR / "external" / "skus_on_shelves_pl" / "extracted"
PL_ANN = PL_DIR / "annotations.json"
MAPPING_FILE = Path(__file__).parent / "outputs" / "pl_to_ng_simple_mapping.json"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
V3_DIR = DATA_DIR / "yolo_augmented_v3"


def coco_to_yolo_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = max(0, min(1, (x + w / 2) / img_w))
    cy = max(0, min(1, (y + h / 2) / img_h))
    nw = max(0, min(1, w / img_w))
    nh = max(0, min(1, h / img_h))
    return cx, cy, nw, nh


def main():
    # Load PL -> NG category mapping
    with open(MAPPING_FILE) as f:
        pl_to_ng = json.load(f)  # str(pl_cat_id) -> ng_cat_id
    print(f"PL->NG mapping: {len(pl_to_ng)} categories")

    # Load PL annotations
    print("Loading PL annotations (213 MB)...")
    with open(PL_ANN) as f:
        pl_coco = json.load(f)

    pl_img_map = {img["id"]: img for img in pl_coco["images"]}

    # Group PL annotations by image, filtering to mapped categories only
    pl_img_anns = defaultdict(list)
    mapped_ann_count = 0
    for ann in pl_coco["annotations"]:
        pl_cat_str = str(ann["category_id"])
        if pl_cat_str in pl_to_ng:
            ng_cat_id = pl_to_ng[pl_cat_str]
            pl_img_anns[ann["image_id"]].append({
                "category_id": ng_cat_id,
                "bbox": ann["bbox"],
            })
            mapped_ann_count += 1

    # Cap PL annotations per NG category to avoid domination
    # Max 500 PL annotations per our category (on top of our existing data)
    MAX_PL_ANNS_PER_CAT = 500
    cat_pl_counts = Counter()
    for img_id, anns in pl_img_anns.items():
        for ann in anns:
            cat_pl_counts[ann["category_id"]] += 1

    # Skip unknown_product (cat 355) from PL entirely - it's noise
    print(f"\nPL annotation counts before capping (top 10):")
    for cat_id, cnt in cat_pl_counts.most_common(10):
        print(f"  cat {cat_id}: {cnt}")

    # Build set of images to include, respecting per-category caps
    cat_budget = {cat_id: MAX_PL_ANNS_PER_CAT for cat_id in range(356)}
    cat_budget[355] = 0  # Skip unknown_product from PL

    eligible_images = set()
    cat_used = Counter()
    # Shuffle images for diversity
    import random
    random.seed(42)
    shuffled_img_ids = list(pl_img_anns.keys())
    random.shuffle(shuffled_img_ids)

    for img_id in shuffled_img_ids:
        anns = pl_img_anns[img_id]
        # Check if any annotation in this image is still under budget
        dominated_cat = max(anns, key=lambda a: cat_pl_counts[a["category_id"]])["category_id"]
        if cat_used[dominated_cat] < cat_budget.get(dominated_cat, MAX_PL_ANNS_PER_CAT):
            eligible_images.add(img_id)
            for ann in anns:
                cat_used[ann["category_id"]] += 1

    print(f"\nPL images after capping ({MAX_PL_ANNS_PER_CAT}/cat): {len(eligible_images)}")
    print(f"Total capped annotations: {sum(cat_used.values())}")

    # Check which PL images actually exist on disk
    available_images = set()
    for img_id in eligible_images:
        img_info = pl_img_map.get(img_id)
        if not img_info:
            continue
        # PL images are just numbered .jpeg files in the extracted dir
        img_path = PL_DIR / img_info["file_name"]
        if img_path.exists():
            available_images.add(img_id)

    print(f"PL images available on disk: {len(available_images)}")

    # Create v3 output structure
    for split in ["train", "val"]:
        (V3_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V3_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Step 1: Symlink all v2 train and val data
    print("\nLinking v2 train data...")
    v2_train_count = 0
    for img_path in (V2_DIR / "train" / "images").iterdir():
        dst = V3_DIR / "train" / "images" / img_path.name
        if not dst.exists():
            # Resolve symlinks from v2
            real_path = img_path.resolve()
            dst.symlink_to(real_path)
        # Copy label
        label_src = V2_DIR / "train" / "labels" / (img_path.stem + ".txt")
        label_dst = V3_DIR / "train" / "labels" / (img_path.stem + ".txt")
        if label_src.exists() and not label_dst.exists():
            shutil.copy2(label_src, label_dst)
        v2_train_count += 1

    print(f"  Linked {v2_train_count} v2 train images")

    print("Linking v2 val data...")
    v2_val_count = 0
    for img_path in (V2_DIR / "val" / "images").iterdir():
        dst = V3_DIR / "val" / "images" / img_path.name
        if not dst.exists():
            real_path = img_path.resolve()
            dst.symlink_to(real_path)
        label_src = V2_DIR / "val" / "labels" / (img_path.stem + ".txt")
        label_dst = V3_DIR / "val" / "labels" / (img_path.stem + ".txt")
        if label_src.exists() and not label_dst.exists():
            shutil.copy2(label_src, label_dst)
        v2_val_count += 1

    print(f"  Linked {v2_val_count} v2 val images")

    # Step 2: Add PL images to train
    print("\nAdding PL images to train...")
    pl_added = 0
    pl_anns_added = 0
    for img_id in sorted(available_images):
        img_info = pl_img_map[img_id]
        img_w = img_info["width"]
        img_h = img_info["height"]
        filename = img_info["file_name"]

        # Create unique filename with pl_ prefix to avoid collisions
        pl_filename = f"pl_{filename}"
        stem = Path(pl_filename).stem

        # Symlink image
        src = (PL_DIR / filename).resolve()
        dst = V3_DIR / "train" / "images" / pl_filename
        if not dst.exists():
            dst.symlink_to(src)

        # Write YOLO labels
        anns = pl_img_anns[img_id]
        label_path = V3_DIR / "train" / "labels" / f"{stem}.txt"
        with open(label_path, "w") as f:
            for ann in anns:
                cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                f.write(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        pl_added += 1
        pl_anns_added += len(anns)

    print(f"  Added {pl_added} PL images, {pl_anns_added} annotations")

    # Step 3: Write dataset.yaml with yaml.dump()
    with open(COCO_ANN) as f:
        our_coco = json.load(f)
    names = {c["id"]: c["name"] for c in our_coco["categories"]}

    yaml_data = {
        "path": str(V3_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names,
    }
    yaml_path = V3_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write("# NorgesGruppen Object Detection - Augmented v3 (SAM3 + PL Shelf Data)\n")
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Summary
    total_train = v2_train_count + pl_added
    total_train_imgs = len(list((V3_DIR / "train" / "images").iterdir()))
    total_val_imgs = len(list((V3_DIR / "val" / "images").iterdir()))

    # Count annotations per category in combined set
    cat_counts = Counter()
    for label_file in (V3_DIR / "train" / "labels").glob("*.txt"):
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    cat_counts[int(parts[0])] += 1

    covered_cats = len(cat_counts)
    counts = sorted(cat_counts.values()) if cat_counts else [0]

    print(f"\n=== YOLO AUGMENTED V3 COMPLETE ===")
    print(f"Output: {V3_DIR}")
    print(f"Train: {total_train_imgs} images")
    print(f"  - From v2: {v2_train_count}")
    print(f"  - From PL: {pl_added} ({pl_anns_added} mapped annotations)")
    print(f"Val: {total_val_imgs} images (unchanged)")
    print(f"Categories covered in train: {covered_cats}/356")
    print(f"Train annotations per class: min={counts[0]}, median={counts[len(counts)//2]}, max={counts[-1]}")
    print(f"dataset.yaml: {yaml_path}")


if __name__ == "__main__":
    main()

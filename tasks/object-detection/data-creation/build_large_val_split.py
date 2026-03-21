"""
Build a LARGER stratified validation split (80-100 images) for reliable model selection.

Strategy:
1. Load all 248 original COCO images and their annotations
2. Stratified split: ensure every category appears in val
3. Target ~80 val images (32% of 248)
4. Remaining ~168 original images + ALL augmented data = training set
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
import shutil
import yaml

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
OUTPUT_DIR = DATA_DIR / "large_clean_split"

TARGET_VAL = 80
SEED = 42


def main():
    with open(COCO_ANN) as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    # For each image, track which categories it contains
    img_cats = {}
    for img_id, anns in img_anns.items():
        img_cats[img_id] = set(ann["category_id"] for ann in anns)

    all_img_ids = list(img_map.keys())
    print(f"Total original images: {len(all_img_ids)}")

    # Stratified selection: ensure every category appears in val
    random.seed(SEED)

    # Start with images that cover rare categories
    cat_counts = Counter()
    for anns in img_anns.values():
        for ann in anns:
            cat_counts[ann["category_id"]] += 1

    # Sort categories by rarity (rarest first)
    sorted_cats = sorted(cat_counts.keys(), key=lambda c: cat_counts[c])

    val_ids = set()
    remaining = set(all_img_ids)

    # Phase 1: For each category (rarest first), ensure at least one image in val
    cats_covered = set()
    for cat_id in sorted_cats:
        if cat_id in cats_covered:
            continue
        # Find images containing this category that aren't yet in val
        candidates = [img_id for img_id in remaining if cat_id in img_cats.get(img_id, set())]
        if not candidates:
            continue
        # Pick the image that covers the most uncovered categories
        best = max(candidates, key=lambda img_id: len(img_cats[img_id] - cats_covered))
        val_ids.add(best)
        remaining.discard(best)
        cats_covered.update(img_cats[best])

    print(f"Phase 1 (category coverage): {len(val_ids)} val images, {len(cats_covered)}/{len(cat_names)} cats covered")

    # Phase 2: Add more images to reach target, preferring diverse ones
    if len(val_ids) < TARGET_VAL:
        extra_needed = TARGET_VAL - len(val_ids)
        extra = random.sample(list(remaining), min(extra_needed, len(remaining)))
        val_ids.update(extra)
        remaining -= set(extra)

    train_ids = remaining

    print(f"Final split: {len(train_ids)} train, {len(val_ids)} val")

    # Verify coverage
    val_cats = set()
    for img_id in val_ids:
        val_cats.update(img_cats.get(img_id, set()))
    print(f"Val category coverage: {len(val_cats)}/{len(cat_names)}")

    train_cats = set()
    for img_id in train_ids:
        train_cats.update(img_cats.get(img_id, set()))
    print(f"Train category coverage: {len(train_cats)}/{len(cat_names)}")

    # Save split
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    val_filenames = sorted([img_map[img_id]["file_name"] for img_id in val_ids])
    train_filenames = sorted([img_map[img_id]["file_name"] for img_id in train_ids])

    with open(OUTPUT_DIR / "val_images.txt", "w") as f:
        f.write("\n".join(val_filenames) + "\n")
    with open(OUTPUT_DIR / "train_images.txt", "w") as f:
        f.write("\n".join(train_filenames) + "\n")

    print(f"\nVal images saved to: {OUTPUT_DIR / 'val_images.txt'}")
    print(f"Train images saved to: {OUTPUT_DIR / 'train_images.txt'}")

    # Build YOLO dataset
    COCO_IMAGES = DATA_DIR / "coco_dataset" / "train" / "images"
    for split in ["train", "val"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Val: only original images
    def coco_to_yolo(bbox, img_w, img_h):
        x, y, w, h = bbox
        return (
            max(0, min(1, (x + w/2) / img_w)),
            max(0, min(1, (y + h/2) / img_h)),
            max(0, min(1, w / img_w)),
            max(0, min(1, h / img_h)),
        )

    for img_id in val_ids:
        info = img_map[img_id]
        src = COCO_IMAGES / info["file_name"]
        if not src.exists():
            continue
        dst = OUTPUT_DIR / "val" / "images" / info["file_name"]
        if not dst.exists():
            dst.symlink_to(src.resolve())
        # Write labels
        anns = img_anns.get(img_id, [])
        label_path = OUTPUT_DIR / "val" / "labels" / (Path(info["file_name"]).stem + ".txt")
        with open(label_path, "w") as f:
            for ann in anns:
                cx, cy, nw, nh = coco_to_yolo(ann["bbox"], info["width"], info["height"])
                f.write(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Train: original train images
    for img_id in train_ids:
        info = img_map[img_id]
        src = COCO_IMAGES / info["file_name"]
        if not src.exists():
            continue
        dst = OUTPUT_DIR / "train" / "images" / info["file_name"]
        if not dst.exists():
            dst.symlink_to(src.resolve())
        anns = img_anns.get(img_id, [])
        label_path = OUTPUT_DIR / "train" / "labels" / (Path(info["file_name"]).stem + ".txt")
        with open(label_path, "w") as f:
            for ann in anns:
                cx, cy, nw, nh = coco_to_yolo(ann["bbox"], info["width"], info["height"])
                f.write(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

    # Add ALL augmented data to train (exclude val original images)
    val_stems = {Path(fn).stem for fn in val_filenames}
    aug_sources = [
        DATA_DIR / "yolo_clean_with_val" / "train",
    ]
    aug_added = 0
    for src_dir in aug_sources:
        if not (src_dir / "images").exists():
            continue
        for img_path in (src_dir / "images").iterdir():
            stem = img_path.stem
            # Skip original images (already handled above) and val leaks
            if stem in val_stems:
                continue
            if img_path.name.startswith("img_"):
                continue  # Original images already added from COCO directly
            dst = OUTPUT_DIR / "train" / "images" / img_path.name
            if dst.exists():
                continue
            dst.symlink_to(img_path.resolve())
            lsrc = src_dir / "labels" / (stem + ".txt")
            ldst = OUTPUT_DIR / "train" / "labels" / (stem + ".txt")
            if lsrc.exists() and not ldst.exists():
                shutil.copy2(lsrc, ldst)
            aug_added += 1

    # Write dataset.yaml
    with open(COCO_ANN) as f:
        names = {c["id"]: c["name"] for c in json.load(f)["categories"]}
    yaml_data = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names,
    }
    with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
        f.write("# LARGE stratified val split for reliable model selection\n")
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    total_train = len(list((OUTPUT_DIR / "train/images").iterdir()))
    total_val = len(list((OUTPUT_DIR / "val/images").iterdir()))
    print(f"\nFinal dataset:")
    print(f"  Train: {total_train} ({len(train_ids)} original + {aug_added} augmented)")
    print(f"  Val: {total_val} (original only, stratified)")
    print(f"  dataset.yaml: {OUTPUT_DIR / 'dataset.yaml'}")

    # Annotation stats
    train_counts = Counter()
    for lf in (OUTPUT_DIR / "train/labels").glob("*.txt"):
        for line in open(lf):
            parts = line.strip().split()
            if parts:
                train_counts[int(parts[0])] += 1
    vals = sorted(train_counts.values())
    print(f"  Train annotations: {sum(vals)}, {len(train_counts)} cats, min={vals[0]}/cat")


if __name__ == "__main__":
    main()

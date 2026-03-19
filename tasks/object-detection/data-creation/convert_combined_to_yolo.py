"""
Convert SAM3 combined augmented COCO dataset to YOLO format.
- Original 50 val images stay as val (same split as yolo_dataset)
- All augmented images + remaining original train images go to train
"""
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
COMBINED_ANN = DATA_DIR / "augmented" / "combined_annotations.json"
AUG_IMAGES_DIR = DATA_DIR / "augmented" / "images"
ORIG_IMAGES_DIR = DATA_DIR / "coco_dataset" / "train" / "images"
YOLO_OUT = DATA_DIR / "yolo_augmented_v2"
# Use same val split as original yolo_dataset
ORIG_YOLO_VAL = DATA_DIR / "yolo_dataset" / "val" / "images"

SEED = 42


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """Convert COCO bbox [x, y, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    nw = max(0, min(1, nw))
    nh = max(0, min(1, nh))
    return cx, cy, nw, nh


def main():
    print("Loading combined annotations...")
    with open(COMBINED_ANN) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    img_map = {img["id"]: img for img in images}

    # Group annotations by image
    img_anns = defaultdict(list)
    for ann in annotations:
        img_anns[ann["image_id"]].append(ann)

    # Determine val image filenames from original YOLO val split
    # Match by stem to handle .jpg/.jpeg differences
    val_stems = set()
    if ORIG_YOLO_VAL.exists():
        val_stems = {f.stem for f in ORIG_YOLO_VAL.iterdir() if f.suffix in (".jpg", ".jpeg")}
    print(f"Val image stems from original split: {len(val_stems)}")

    # Create output dirs
    for split in ["train", "val"]:
        (YOLO_OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (YOLO_OUT / split / "labels").mkdir(parents=True, exist_ok=True)

    stats = {"train": 0, "val": 0, "train_anns": 0, "val_anns": 0, "skipped": 0}

    for img_id, img_info in img_map.items():
        filename = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # Determine split: original val images stay val, everything else is train
        if Path(filename).stem in val_stems:
            split = "val"
        else:
            split = "train"

        # Find source image
        if filename.startswith("aug_"):
            src_path = AUG_IMAGES_DIR / filename
        else:
            src_path = ORIG_IMAGES_DIR / filename

        if not src_path.exists():
            stats["skipped"] += 1
            continue

        # Copy/symlink image
        dst_img = YOLO_OUT / split / "images" / filename
        if not dst_img.exists():
            # Use symlink to save disk space
            dst_img.symlink_to(src_path.resolve())

        # Write YOLO labels
        anns = img_anns.get(img_id, [])
        label_file = YOLO_OUT / split / "labels" / (Path(filename).stem + ".txt")
        with open(label_file, "w") as f:
            for ann in anns:
                cat_id = ann["category_id"]
                cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                f.write(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        stats[split] += 1
        stats[f"{split}_anns"] += len(anns)

    # Write dataset.yaml
    yaml_content = f"""# NorgesGruppen Object Detection - Augmented v2 (SAM3 + Original)
path: {YOLO_OUT.resolve()}
train: train/images
val: val/images

nc: {len(categories)}
names:
"""
    for cat in categories:
        name = cat["name"].replace("'", "\\'").replace('"', '\\"')
        yaml_content += f"  {cat['id']}: '{name}'\n"

    with open(YOLO_OUT / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    print(f"\n=== YOLO AUGMENTED V2 DATASET CREATED ===")
    print(f"Output: {YOLO_OUT}")
    print(f"Train: {stats['train']} images, {stats['train_anns']} annotations")
    print(f"Val: {stats['val']} images, {stats['val_anns']} annotations")
    print(f"Skipped (missing source): {stats['skipped']}")
    print(f"dataset.yaml: {YOLO_OUT / 'dataset.yaml'}")
    print(f"Classes: {len(categories)}")
    print(f"\nNote: Images are symlinked to save disk space.")


if __name__ == "__main__":
    main()

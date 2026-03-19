"""
Convert COCO dataset to YOLO format with 80/20 train/val split.
Output: data/yolo_dataset/{train,val}/{images,labels}/
"""
import json
import random
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
COCO_DIR = DATA_DIR / "coco_dataset" / "train"
ANNOTATIONS = COCO_DIR / "annotations.json"
YOLO_DIR = DATA_DIR / "yolo_dataset"

TRAIN_RATIO = 0.8
SEED = 42


def coco_to_yolo_bbox(bbox, img_w, img_h):
    """Convert COCO bbox [x, y, w, h] to YOLO [cx, cy, w, h] normalized."""
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    nw = max(0, min(1, nw))
    nh = max(0, min(1, nh))
    return cx, cy, nw, nh


def main():
    with open(ANNOTATIONS) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    # Build image id -> info mapping
    img_map = {img["id"]: img for img in images}

    # Group annotations by image
    from collections import defaultdict
    img_anns = defaultdict(list)
    for ann in annotations:
        img_anns[ann["image_id"]].append(ann)

    # Shuffle and split
    random.seed(SEED)
    image_ids = list(img_map.keys())
    random.shuffle(image_ids)
    split_idx = int(len(image_ids) * TRAIN_RATIO)
    train_ids = set(image_ids[:split_idx])
    val_ids = set(image_ids[split_idx:])

    print(f"Total images: {len(image_ids)}")
    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Create directory structure
    for split in ["train", "val"]:
        (YOLO_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Convert
    stats = {"train": 0, "val": 0, "train_anns": 0, "val_anns": 0}
    for img_id, img_info in img_map.items():
        split = "train" if img_id in train_ids else "val"
        img_file = img_info["file_name"]
        img_w = img_info["width"]
        img_h = img_info["height"]

        # Copy image
        src = COCO_DIR / "images" / img_file
        dst = YOLO_DIR / split / "images" / img_file
        if not dst.exists():
            shutil.copy2(src, dst)

        # Write YOLO label file
        label_file = YOLO_DIR / split / "labels" / (Path(img_file).stem + ".txt")
        anns = img_anns.get(img_id, [])
        with open(label_file, "w") as f:
            for ann in anns:
                cat_id = ann["category_id"]
                cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
                f.write(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        stats[split] += 1
        stats[f"{split}_anns"] += len(anns)

    # Write dataset.yaml for ultralytics
    yaml_content = f"""# NorgesGruppen Object Detection Dataset
path: {YOLO_DIR.resolve()}
train: train/images
val: val/images

nc: {len(categories)}
names:
"""
    for cat in categories:
        name = cat["name"].replace("'", "\\'").replace('"', '\\"')
        yaml_content += f"  {cat['id']}: '{name}'\n"

    yaml_path = YOLO_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"\n=== YOLO DATASET CREATED ===")
    print(f"Output: {YOLO_DIR}")
    print(f"Train: {stats['train']} images, {stats['train_anns']} annotations")
    print(f"Val: {stats['val']} images, {stats['val_anns']} annotations")
    print(f"dataset.yaml: {yaml_path}")
    print(f"Classes: {len(categories)}")


if __name__ == "__main__":
    main()

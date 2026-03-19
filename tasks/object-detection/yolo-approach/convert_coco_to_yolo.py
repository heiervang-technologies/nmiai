"""Convert COCO annotations to YOLO format and create train/val split."""
import json
import random
import shutil
from pathlib import Path

# Paths
COCO_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data/coco_dataset/train")
ANNOTATIONS = COCO_DIR / "annotations.json"
IMAGES_DIR = COCO_DIR / "images"

OUTPUT_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/yolo-approach/dataset")
TRAIN_IMAGES = OUTPUT_DIR / "images" / "train"
VAL_IMAGES = OUTPUT_DIR / "images" / "val"
TRAIN_LABELS = OUTPUT_DIR / "labels" / "train"
VAL_LABELS = OUTPUT_DIR / "labels" / "val"

VAL_RATIO = 0.15  # 15% validation (~37 images)
SEED = 42

def main():
    # Load COCO annotations
    with open(ANNOTATIONS) as f:
        coco = json.load(f)

    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])}")

    # Build image info lookup
    img_info = {img['id']: img for img in coco['images']}

    # Group annotations by image
    img_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    # Train/val split
    random.seed(SEED)
    all_img_ids = list(img_info.keys())
    random.shuffle(all_img_ids)
    n_val = max(1, int(len(all_img_ids) * VAL_RATIO))
    val_ids = set(all_img_ids[:n_val])
    train_ids = set(all_img_ids[n_val:])

    print(f"Train: {len(train_ids)} images, Val: {len(val_ids)} images")

    # Create directories
    for d in [TRAIN_IMAGES, VAL_IMAGES, TRAIN_LABELS, VAL_LABELS]:
        d.mkdir(parents=True, exist_ok=True)

    # Convert and copy
    train_count = val_count = 0
    for img_id, info in img_info.items():
        w, h = info['width'], info['height']
        fname = info['file_name']
        stem = Path(fname).stem

        is_val = img_id in val_ids
        img_dst = VAL_IMAGES if is_val else TRAIN_IMAGES
        lbl_dst = VAL_LABELS if is_val else TRAIN_LABELS

        # Symlink image (save disk space)
        src = IMAGES_DIR / fname
        dst = img_dst / fname
        if not dst.exists():
            dst.symlink_to(src)

        # Convert annotations to YOLO format
        anns = img_anns.get(img_id, [])
        lines = []
        for ann in anns:
            cat_id = ann['category_id']  # Already 0-355
            bx, by, bw, bh = ann['bbox']  # COCO: x_min, y_min, width, height

            # Convert to YOLO: center_x, center_y, width, height (normalized)
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h

            # Clamp to [0, 1]
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))

            lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_file = lbl_dst / f"{stem}.txt"
        label_file.write_text("\n".join(lines) + "\n" if lines else "")

        if is_val:
            val_count += 1
        else:
            train_count += 1

    print(f"Written {train_count} train, {val_count} val label files")

    # Generate data.yaml
    categories = sorted(coco['categories'], key=lambda c: c['id'])
    names = {c['id']: c['name'] for c in categories}

    yaml_lines = [
        f"path: {OUTPUT_DIR}",
        f"train: images/train",
        f"val: images/val",
        f"",
        f"nc: {len(categories)}",
        f"names:",
    ]
    for cat_id in range(len(categories)):
        # Escape quotes in names
        name = names[cat_id].replace("'", "\\'")
        yaml_lines.append(f"  {cat_id}: '{name}'")

    data_yaml = OUTPUT_DIR / "data.yaml"
    data_yaml.write_text("\n".join(yaml_lines) + "\n")
    print(f"Written {data_yaml}")

    # Also save to yolo-approach root for convenience
    root_yaml = Path("/home/me/ht/nmiai/tasks/object-detection/yolo-approach/data.yaml")
    shutil.copy2(data_yaml, root_yaml)
    print(f"Copied to {root_yaml}")

    # Print class distribution stats
    class_counts = {}
    for ann in coco['annotations']:
        cid = ann['category_id']
        class_counts[cid] = class_counts.get(cid, 0) + 1

    counts = sorted(class_counts.values())
    print(f"\nClass distribution:")
    print(f"  Min annotations/class: {counts[0]}")
    print(f"  Max annotations/class: {counts[-1]}")
    print(f"  Median: {counts[len(counts)//2]}")
    print(f"  Classes with <10 annotations: {sum(1 for c in counts if c < 10)}")
    print(f"  Classes with <5 annotations: {sum(1 for c in counts if c < 5)}")


if __name__ == "__main__":
    main()

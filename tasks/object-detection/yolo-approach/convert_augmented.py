"""Convert augmented COCO dataset to YOLO format with train/val split."""
import json
import random
import yaml
from pathlib import Path

AUGMENTED_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data/augmented")
ORIGINAL_IMAGES = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data/coco_dataset/train/images")
AUGMENTED_IMAGES = AUGMENTED_DIR / "images"
ANNOTATIONS = AUGMENTED_DIR / "combined_annotations.json"

OUTPUT_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/yolo-approach/dataset_augmented")

VAL_RATIO = 0.10  # 10% val from original images only
SEED = 42


def main():
    with open(ANNOTATIONS) as f:
        coco = json.load(f)

    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")
    print(f"Categories: {len(coco['categories'])}")

    img_info = {img['id']: img for img in coco['images']}

    # Group annotations by image
    img_anns = {}
    for ann in coco['annotations']:
        img_id = ann['image_id']
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    # Split: only original images go to val, augmented always train
    original_ids = []
    augmented_ids = []
    for img in coco['images']:
        if img['file_name'].startswith('aug_'):
            augmented_ids.append(img['id'])
        else:
            original_ids.append(img['id'])

    print(f"Original images: {len(original_ids)}, Augmented: {len(augmented_ids)}")

    random.seed(SEED)
    random.shuffle(original_ids)
    n_val = max(1, int(len(original_ids) * VAL_RATIO))
    val_ids = set(original_ids[:n_val])
    train_ids = set(original_ids[n_val:]) | set(augmented_ids)

    print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")

    # Create directories
    for split in ['train', 'val']:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Process all images
    for img_id, info in img_info.items():
        w, h = info['width'], info['height']
        fname = info['file_name']
        stem = Path(fname).stem

        is_val = img_id in val_ids
        split = "val" if is_val else "train"

        # Find source image
        if fname.startswith('aug_'):
            src = AUGMENTED_IMAGES / fname
        else:
            src = ORIGINAL_IMAGES / fname

        # Symlink image
        dst = OUTPUT_DIR / "images" / split / fname
        if not dst.exists() and src.exists():
            dst.symlink_to(src)

        # Convert annotations
        anns = img_anns.get(img_id, [])
        lines = []
        for ann in anns:
            cat_id = ann['category_id']
            bx, by, bw, bh = ann['bbox']
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h
            cx = max(0.0, min(1.0, cx))
            cy = max(0.0, min(1.0, cy))
            nw = max(0.001, min(1.0, nw))
            nh = max(0.001, min(1.0, nh))
            lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        label_file = OUTPUT_DIR / "labels" / split / f"{stem}.txt"
        label_file.write_text("\n".join(lines) + "\n" if lines else "")

    # Generate data.yaml with proper escaping
    categories = sorted(coco['categories'], key=lambda c: c['id'])
    names = {c['id']: c['name'] for c in categories}

    data = {
        'path': str(OUTPUT_DIR),
        'train': 'images/train',
        'val': 'images/val',
        'nc': len(categories),
        'names': names,
    }

    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Verify
    with open(yaml_path) as f:
        loaded = yaml.safe_load(f)
    assert loaded['nc'] == 356
    assert len(loaded['names']) == 356

    # Also copy to yolo-approach root
    import shutil
    root_yaml = Path("/home/me/ht/nmiai/tasks/object-detection/yolo-approach/data_augmented.yaml")
    shutil.copy2(yaml_path, root_yaml)

    print(f"Written {yaml_path}")
    print(f"Copied to {root_yaml}")

    # Stats
    train_imgs = list((OUTPUT_DIR / "images" / "train").glob("*"))
    val_imgs = list((OUTPUT_DIR / "images" / "val").glob("*"))
    print(f"\nFinal: {len(train_imgs)} train images, {len(val_imgs)} val images")

    # Class distribution on augmented
    class_counts = {}
    for ann in coco['annotations']:
        cid = ann['category_id']
        class_counts[cid] = class_counts.get(cid, 0) + 1
    counts = sorted(class_counts.values())
    print(f"Class distribution (augmented):")
    print(f"  Min: {counts[0]}, Max: {counts[-1]}, Median: {counts[len(counts)//2]}")
    print(f"  <5 annotations: {sum(1 for c in counts if c < 5)}")
    print(f"  <10 annotations: {sum(1 for c in counts if c < 10)}")


if __name__ == "__main__":
    main()

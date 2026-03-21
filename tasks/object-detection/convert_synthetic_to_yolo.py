#!/usr/bin/env python3
"""
Convert synthetic COCO dataset to YOLO format and merge with existing training data.
"""

import json
import shutil
from pathlib import Path

BASE_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data")
SYNTHETIC_DIR = BASE_DIR / "synthetic_dataset"
SYNTHETIC_ANNOT = SYNTHETIC_DIR / "annotations.json"
SYNTHETIC_IMAGES = SYNTHETIC_DIR / "images"

# Output YOLO format
YOLO_OUTPUT = BASE_DIR / "synthetic_yolo"
YOLO_IMAGES = YOLO_OUTPUT / "images" / "train"
YOLO_LABELS = YOLO_OUTPUT / "labels" / "train"


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
    print("Converting synthetic COCO to YOLO format...")

    with open(SYNTHETIC_ANNOT) as f:
        data = json.load(f)

    # Build image lookup
    images = {img["id"]: img for img in data["images"]}

    # Group annotations by image
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    # Create output dirs
    YOLO_IMAGES.mkdir(parents=True, exist_ok=True)
    YOLO_LABELS.mkdir(parents=True, exist_ok=True)

    converted = 0
    skipped = 0

    for img_id, img_info in images.items():
        fname = img_info["file_name"]
        src = SYNTHETIC_IMAGES / fname

        # Skip background-only images
        if fname.startswith("bg_"):
            continue

        if not src.exists():
            skipped += 1
            continue

        # Copy image
        dst_img = YOLO_IMAGES / fname
        shutil.copy2(src, dst_img)

        # Write label file
        label_name = Path(fname).stem + ".txt"
        dst_label = YOLO_LABELS / label_name

        anns = anns_by_image.get(img_id, [])
        img_w = img_info["width"]
        img_h = img_info["height"]

        lines = []
        for ann in anns:
            cx, cy, nw, nh = coco_to_yolo_bbox(ann["bbox"], img_w, img_h)
            # Filter out degenerate boxes
            if nw < 0.005 or nh < 0.005:
                continue
            lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        with open(dst_label, "w") as f:
            f.write("\n".join(lines))

        converted += 1

    # Write YAML config
    yaml_path = YOLO_OUTPUT / "synthetic_data.yaml"
    with open(yaml_path, "w") as f:
        f.write(f"path: {YOLO_OUTPUT}\n")
        f.write("train: images/train\n")
        f.write("val: images/train\n")  # No separate val for synthetic
        f.write("\n")
        # Copy class names from main config
        main_yaml = BASE_DIR / "yolo_all_data.yaml"
        if main_yaml.exists():
            with open(main_yaml) as mf:
                f.write(mf.read())

    print(f"Converted {converted} images, skipped {skipped}")
    print(f"YOLO dataset at: {YOLO_OUTPUT}")
    print(f"YAML config: {yaml_path}")


if __name__ == "__main__":
    main()

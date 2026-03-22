#!/usr/bin/env python3
"""Convert Grocery Shelves dataset (Supervisely JSON) to COCO format.

Input: data-creation/data/external/grocery_shelves/Supermarket shelves/Supermarket shelves/
Output: data-creation/data/external/grocery_shelves/coco_annotations.json

Only keeps "Product" class (ignores "Price" tags).
"""
import json
from pathlib import Path

SRC = Path(__file__).parent / "data-creation/data/external/grocery_shelves/Supermarket shelves/Supermarket shelves"
IMG_DIR = SRC / "images"
ANN_DIR = SRC / "annotations"
OUT = Path(__file__).parent / "data-creation/data/external/grocery_shelves/coco_annotations.json"


def main():
    images = []
    annotations = []
    ann_id = 0

    for idx, img_path in enumerate(sorted(IMG_DIR.glob("*.jpg"))):
        ann_path = ANN_DIR / f"{img_path.name}.json"
        if not ann_path.exists():
            continue

        with open(ann_path) as f:
            data = json.load(f)

        w = data["size"]["width"]
        h = data["size"]["height"]

        images.append({
            "id": idx,
            "file_name": img_path.name,
            "width": w,
            "height": h,
        })

        for obj in data["objects"]:
            if obj["classTitle"] != "Product":
                continue
            pts = obj["points"]["exterior"]
            x1, y1 = pts[0]
            x2, y2 = pts[1]
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": idx,
                "category_id": 0,  # single class: product
                "bbox": [x1, y1, bw, bh],
                "area": bw * bh,
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "product", "supercategory": "food"}],
    }

    with open(OUT, "w") as f:
        json.dump(coco, f)

    print(f"Converted: {len(images)} images, {len(annotations)} product annotations")
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()

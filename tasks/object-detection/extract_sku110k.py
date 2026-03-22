#!/usr/bin/env python3
"""Extract SKU-110K from parquet to images + COCO JSON.

Usage: python extract_sku110k.py [--max-images 2000]
"""
import argparse
import json
from pathlib import Path

PARQUET_DIR = Path(__file__).parent / "data-creation/data/external/sku110k/data"
OUT_DIR = Path(__file__).parent / "data-creation/data/external/sku110k_extracted"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=None)
    args = parser.parse_args()

    import pyarrow.parquet as pq

    img_dir = OUT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(PARQUET_DIR.glob("*.parquet"))
    print(f"Found {len(parquet_files)} parquet shards")

    images = []
    annotations = []
    ann_id = 0
    img_count = 0

    for pf in parquet_files:
        table = pq.read_table(pf)

        for i in range(table.num_rows):
            if args.max_images and img_count >= args.max_images:
                break

            img_id_str = table.column("image_id")[i].as_py()
            w = table.column("width")[i].as_py()
            h = table.column("height")[i].as_py()
            img_struct = table.column("image")[i].as_py()
            img_bytes = img_struct["bytes"]

            fname = f"{img_id_str}.jpg"
            img_path = img_dir / fname

            if not img_path.exists():
                with open(img_path, "wb") as f:
                    f.write(img_bytes)

            images.append({
                "id": img_count,
                "file_name": fname,
                "width": w,
                "height": h,
            })

            objects = table.column("objects")[i].as_py()
            for obj in objects:
                bbox = obj["bbox"]
                x, y, bw, bh = bbox
                if max(x, y, bw, bh) <= 1.0:
                    x *= w
                    y *= h
                    bw *= w
                    bh *= h

                annotations.append({
                    "id": ann_id,
                    "image_id": img_count,
                    "category_id": 0,
                    "bbox": [float(x), float(y), float(bw), float(bh)],
                    "area": float(bw * bh),
                    "iscrowd": 0,
                })
                ann_id += 1

            img_count += 1
            if img_count % 500 == 0:
                print(f"  Extracted {img_count} images, {ann_id} annotations...")

        if args.max_images and img_count >= args.max_images:
            break

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 0, "name": "product", "supercategory": "food"}],
    }

    ann_path = OUT_DIR / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(coco, f)

    print(f"\nExtracted: {img_count} images, {ann_id} annotations")
    print(f"Images: {img_dir}")
    print(f"Annotations: {ann_path}")


if __name__ == "__main__":
    main()

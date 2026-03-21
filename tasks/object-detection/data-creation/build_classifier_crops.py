"""
Build a classifier training dataset from all available sources.
Extracts and organizes crops by category for DINOv2/classifier training.

Output structure:
  classifier_crops/
    {category_id}/
      crop_0001.jpg
      crop_0002.jpg
      ...
"""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
COCO_IMGS = DATA_DIR / "coco_dataset" / "train" / "images"
PRODUCT_CUTOUTS = DATA_DIR / "product_cutouts"
SCRAPED_DIR = DATA_DIR / "scraped_products"
PRODUCT_IMAGES = DATA_DIR / "product_images"
EXTRACTED_CROPS = DATA_DIR / "extracted_crops"
OUTPUT_DIR = DATA_DIR / "classifier_crops"

MIN_CROPS_PER_CAT = 20
MAX_CROPS_PER_CAT = 100


def main():
    print("Building classifier crop dataset...")

    # Load COCO annotations
    with open(COCO_ANN) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    # Load barcode mapping
    mapping_path = Path(__file__).parent / "outputs" / "barcode_category_mapping.json"
    barcode_map = {}
    if mapping_path.exists():
        with open(mapping_path) as f:
            barcode_map = json.load(f)

    cat_crops = Counter()

    # Step 1: Extract crops from COCO training annotations
    print("1. Extracting COCO training crops...")
    for ann in coco["annotations"]:
        cat_id = ann["category_id"]
        if cat_crops[cat_id] >= MAX_CROPS_PER_CAT:
            continue

        img_info = img_map[ann["image_id"]]
        x, y, w, h = ann["bbox"]
        if w < 15 or h < 15:
            continue

        img_path = COCO_IMGS / img_info["file_name"]
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            pad = 2
            x1 = max(0, int(x) - pad)
            y1 = max(0, int(y) - pad)
            x2 = min(img.width, int(x + w) + pad)
            y2 = min(img.height, int(y + h) + pad)
            crop = img.crop((x1, y1, x2, y2))

            cat_dir = OUTPUT_DIR / str(cat_id)
            cat_dir.mkdir(parents=True, exist_ok=True)
            crop.save(cat_dir / f"coco_{cat_crops[cat_id]:04d}.jpg", quality=92)
            cat_crops[cat_id] += 1
        except Exception:
            pass

    print(f"   Extracted {sum(cat_crops.values())} COCO crops")

    # Step 2: Add product cutouts
    print("2. Adding product cutouts...")
    added = 0
    for f in PRODUCT_CUTOUTS.glob("*.png"):
        try:
            cat_id = int(f.stem.split("_")[0].replace("cat", ""))
            if cat_crops[cat_id] >= MAX_CROPS_PER_CAT:
                continue
            img = Image.open(f).convert("RGB")
            cat_dir = OUTPUT_DIR / str(cat_id)
            cat_dir.mkdir(parents=True, exist_ok=True)
            img.save(cat_dir / f"cutout_{cat_crops[cat_id]:04d}.jpg", quality=92)
            cat_crops[cat_id] += 1
            added += 1
        except Exception:
            pass
    print(f"   Added {added} cutouts")

    # Step 3: Add scraped product images
    print("3. Adding scraped products...")
    added = 0
    for cat_dir_src in SCRAPED_DIR.iterdir():
        if not cat_dir_src.is_dir():
            continue
        try:
            cat_id = int(cat_dir_src.name)
        except ValueError:
            continue
        if cat_crops[cat_id] >= MAX_CROPS_PER_CAT:
            continue

        for img_path in list(cat_dir_src.glob("*.jpg")) + list(cat_dir_src.glob("*.png")):
            if cat_crops[cat_id] >= MAX_CROPS_PER_CAT:
                break
            try:
                img = Image.open(img_path).convert("RGB")
                cat_dir = OUTPUT_DIR / str(cat_id)
                cat_dir.mkdir(parents=True, exist_ok=True)
                img.save(cat_dir / f"scraped_{cat_crops[cat_id]:04d}.jpg", quality=92)
                cat_crops[cat_id] += 1
                added += 1
            except Exception:
                pass
    print(f"   Added {added} scraped images")

    # Step 4: Add reference product images
    print("4. Adding reference images...")
    added = 0
    for barcode, info in barcode_map.items():
        cat_id = info["category_id"]
        if cat_crops[cat_id] >= MAX_CROPS_PER_CAT:
            continue
        ref_dir = PRODUCT_IMAGES / barcode
        if not ref_dir.exists():
            continue
        for img_path in ref_dir.glob("*.jpg"):
            if cat_crops[cat_id] >= MAX_CROPS_PER_CAT:
                break
            try:
                img = Image.open(img_path).convert("RGB")
                cat_dir = OUTPUT_DIR / str(cat_id)
                cat_dir.mkdir(parents=True, exist_ok=True)
                img.save(cat_dir / f"ref_{cat_crops[cat_id]:04d}.jpg", quality=92)
                cat_crops[cat_id] += 1
                added += 1
            except Exception:
                pass
    print(f"   Added {added} reference images")

    # Step 5: Augment categories that are still under MIN_CROPS
    print(f"5. Augmenting categories under {MIN_CROPS_PER_CAT} crops...")
    augmented = 0
    for cat_id in range(356):
        current = cat_crops[cat_id]
        if current == 0 or current >= MIN_CROPS_PER_CAT:
            continue

        cat_dir = OUTPUT_DIR / str(cat_id)
        existing = list(cat_dir.glob("*.jpg"))
        needed = MIN_CROPS_PER_CAT - current

        for i in range(needed):
            src = random.choice(existing)
            try:
                img = Image.open(src).convert("RGB")
                # Apply augmentation
                for Enh in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
                    img = Enh(img).enhance(random.uniform(0.7, 1.3))
                # Random flip
                if random.random() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                img.save(cat_dir / f"aug_{cat_crops[cat_id]:04d}.jpg", quality=90)
                cat_crops[cat_id] += 1
                augmented += 1
            except Exception:
                pass
    print(f"   Augmented {augmented} crops")

    # Summary
    print(f"\n{'=' * 50}")
    print(f"CLASSIFIER CROP DATASET COMPLETE")
    print(f"{'=' * 50}")
    total = sum(cat_crops.values())
    print(f"Total crops: {total}")
    print(f"Categories: {len([c for c in cat_crops if cat_crops[c] > 0])}")
    print(f"Min per category: {min(cat_crops.values()) if cat_crops else 0}")
    print(f"Max per category: {max(cat_crops.values()) if cat_crops else 0}")
    print(f"Median per category: {sorted(cat_crops.values())[len(cat_crops)//2]}")

    under_20 = [c for c in range(356) if cat_crops[c] < 20]
    if under_20:
        print(f"\nCategories still under 20 crops:")
        for c in under_20:
            print(f"  cat {c} ({cat_names.get(c, '?')}): {cat_crops[c]}")

    print(f"\nOutput: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

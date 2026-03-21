"""
Targeted data boosting for the 8 weakest categories.

Strategy:
1. Heavy copy-paste augmentation using reference product images
2. For categories with reference images: crop, augment, paste onto shelf backgrounds
3. Generate 50+ synthetic annotations per weak category
"""
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
PRODUCT_DIR = DATA_DIR / "product_images"
MAPPING_FILE = Path(__file__).parent / "outputs" / "barcode_category_mapping.json"
SHELF_IMAGES = DATA_DIR / "coco_dataset" / "train" / "images"
OUTPUT_DIR = DATA_DIR / "boosted_weak"

# Weak categories from gap analysis
WEAK_CATEGORIES = {
    153: "FRIELE FROKOST KOFFEINFRI FILTERMALT 250G",
    242: "BOG 390G GILDE",
    76: "BRUSCHETTA LIGURISK 130G OLIVINO",
    167: "TROPISK AROMA FILTERMALT 200G JACOBS",
    254: "SMØREMYK MELKEFRI 400G BERIT",
    285: "Leka Egg 10stk",
    335: "STORFE SHORT RIBS GREATER OMAHA LV",
    149: "KNEKKEBRØD SESAM&HAVSALT GL.FRI 240G",
}

TARGET_PER_CAT = 50
random.seed(42)


def segment_product_simple(img: Image.Image) -> Image.Image:
    """Simple background removal for studio shots."""
    arr = np.array(img.convert("RGB"))
    # White background detection
    bg = np.all(arr > 230, axis=2)
    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha).filter(ImageFilter.MedianFilter(5))
    rgba = img.convert("RGB").copy()
    rgba.putalpha(alpha_img)
    return rgba


def augment_product(img: Image.Image) -> Image.Image:
    """Apply random augmentation to a product image."""
    rgb = img.convert("RGB") if img.mode != "RGB" else img
    alpha = img.split()[3] if img.mode == "RGBA" else None

    # Random brightness/contrast/color
    for Enhancer in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
        factor = random.uniform(0.8, 1.2)
        rgb = Enhancer(rgb).enhance(factor)

    # Random slight blur
    if random.random() > 0.5:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    if alpha:
        result = rgb.copy()
        result.putalpha(alpha)
        return result
    return rgb


def paste_on_shelf(shelf_img: Image.Image, product: Image.Image, x: int, y: int, w: int, h: int) -> Image.Image:
    """Paste product onto shelf at given location."""
    resized = product.resize((w, h), Image.LANCZOS)
    result = shelf_img.copy()
    if resized.mode == "RGBA":
        result.paste(resized, (x, y), resized.split()[3])
    else:
        result.paste(resized, (x, y))
    return result


def main():
    # Load barcode mapping
    with open(MAPPING_FILE) as f:
        barcode_map = json.load(f)

    # Reverse mapping: category_id -> list of barcode folders
    cat_to_barcodes = {}
    for barcode, info in barcode_map.items():
        cat_id = info["category_id"]
        if cat_id not in cat_to_barcodes:
            cat_to_barcodes[cat_id] = []
        cat_to_barcodes[cat_id].append(barcode)

    # Load original annotations for shelf geometry
    with open(COCO_ANN) as f:
        coco = json.load(f)

    img_anns = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_anns:
            img_anns[img_id] = []
        img_anns[img_id].append(ann)

    img_map = {img["id"]: img for img in coco["images"]}

    # Collect shelf backgrounds
    bg_images = sorted(SHELF_IMAGES.glob("*.jpg")) + sorted(SHELF_IMAGES.glob("*.jpeg"))

    # Setup output
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    new_images = 0
    new_annotations = 0

    for cat_id, cat_name in WEAK_CATEGORIES.items():
        print(f"\n[{cat_id}] {cat_name}")

        # Find reference product images
        barcodes = cat_to_barcodes.get(cat_id, [])
        ref_images = []
        for bc in barcodes:
            ref_dir = PRODUCT_DIR / bc
            if ref_dir.exists():
                for view in ["front.jpg", "main.jpg"]:
                    p = ref_dir / view
                    if p.exists():
                        ref_images.append(p)
                        break
                else:
                    imgs = list(ref_dir.glob("*.jpg"))
                    if imgs:
                        ref_images.append(imgs[0])

        if not ref_images:
            # Try scraped product images
            scraped_dir = DATA_DIR / "scraped_products" / str(cat_id)
            if scraped_dir.exists():
                ref_images = list(scraped_dir.glob("*.jpg"))

        if not ref_images:
            print(f"  No reference images found, skipping")
            continue

        print(f"  Found {len(ref_images)} reference images")

        # Generate augmented images
        for aug_i in range(TARGET_PER_CAT):
            # Pick random reference and background
            ref_path = random.choice(ref_images)
            bg_path = random.choice(bg_images)

            try:
                ref_img = Image.open(ref_path).convert("RGB")
                product_rgba = segment_product_simple(ref_img)
                product_rgba = augment_product(product_rgba)
                bg_img = Image.open(bg_path).convert("RGB")
            except Exception:
                continue

            # Random placement
            # Use typical shelf object sizes from training data
            target_w = random.randint(80, 200)
            target_h = random.randint(100, 250)
            max_x = max(1, bg_img.width - target_w)
            max_y = max(1, bg_img.height - target_h)
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            aug_img = paste_on_shelf(bg_img, product_rgba, x, y, target_w, target_h)

            # Save
            fname = f"boost_{cat_id:03d}_{aug_i:04d}"
            aug_img.save(OUTPUT_DIR / "images" / f"{fname}.jpg", quality=95)

            # YOLO label
            cx = (x + target_w / 2) / bg_img.width
            cy = (y + target_h / 2) / bg_img.height
            nw = target_w / bg_img.width
            nh = target_h / bg_img.height
            with open(OUTPUT_DIR / "labels" / f"{fname}.txt", "w") as f:
                f.write(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            new_images += 1
            new_annotations += 1

        print(f"  Generated {TARGET_PER_CAT} augmented images")

    print(f"\n=== BOOST COMPLETE ===")
    print(f"New images: {new_images}")
    print(f"New annotations: {new_annotations}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

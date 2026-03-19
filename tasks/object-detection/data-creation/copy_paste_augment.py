"""
Copy-paste augmentation pipeline for shelf detection.

Strategy:
1. Segment reference product images (simple background removal since they're studio shots)
2. Paste segmented products onto shelf images at plausible locations
3. Focus on underrepresented categories (< 20 annotations)
4. Generate COCO-format annotations for augmented images
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

DATA_DIR = Path(__file__).parent / "data"
COCO_DIR = DATA_DIR / "coco_dataset" / "train"
PRODUCT_DIR = DATA_DIR / "product_images"
ANNOTATIONS = COCO_DIR / "annotations.json"
OUTPUT_DIR = Path(__file__).parent / "outputs"
AUG_DIR = Path(__file__).parent / "data" / "augmented"


def segment_product(img: Image.Image, threshold: int = 240) -> tuple[Image.Image, Image.Image]:
    """
    Simple background removal for studio product shots.
    Returns (rgba_image, mask).
    """
    img = img.convert("RGB")
    arr = np.array(img)

    # Studio shots typically have white/light backgrounds
    # Detect background as pixels where all channels are > threshold
    bg_mask = np.all(arr > threshold, axis=2)

    # Also check for very uniform regions (studio background)
    gray = np.mean(arr, axis=2)
    std_local = np.zeros_like(gray)
    # Simple edge detection approach
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        std_local += np.abs(gray - shifted)
    uniform_mask = std_local < 20

    # Combine: background is white AND uniform
    combined_bg = bg_mask & uniform_mask

    # Create alpha mask (255 = foreground, 0 = background)
    alpha = np.where(combined_bg, 0, 255).astype(np.uint8)

    # Light morphological cleanup
    from PIL import ImageFilter
    alpha_img = Image.fromarray(alpha).filter(ImageFilter.MedianFilter(5))

    # Create RGBA
    rgba = img.copy()
    rgba.putalpha(alpha_img)

    return rgba, alpha_img


def get_shelf_regions(coco_data: dict) -> dict:
    """
    Analyze training images to find shelf row regions.
    Returns dict of image_id -> list of shelf rows (y_ranges).
    """
    # Group annotations by image
    img_anns = defaultdict(list)
    for ann in coco_data["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    shelf_regions = {}
    for img_id, anns in img_anns.items():
        # Extract y-centers of all boxes
        y_centers = [(ann["bbox"][1] + ann["bbox"][3] / 2) for ann in anns]
        # Typical box heights
        heights = [ann["bbox"][3] for ann in anns]
        avg_h = np.median(heights) if heights else 100

        shelf_regions[img_id] = {
            "y_centers": sorted(y_centers),
            "avg_box_height": avg_h,
            "avg_box_width": np.median([ann["bbox"][2] for ann in anns]) if anns else 80,
        }
    return shelf_regions


def paste_product_on_shelf(
    shelf_img: Image.Image,
    product_rgba: Image.Image,
    target_w: int,
    target_h: int,
    x: int,
    y: int,
) -> Image.Image:
    """Paste a segmented product onto a shelf image."""
    # Resize product to target size
    product_resized = product_rgba.resize((target_w, target_h), Image.LANCZOS)

    # Random augmentations on the product
    # Slight color jitter
    if random.random() > 0.3:
        enhancer = ImageEnhance.Brightness(product_resized.split()[0] if product_resized.mode == "RGBA" else product_resized)
        # Work with RGB channels only
        rgb = product_resized.convert("RGB")
        alpha = product_resized.split()[3] if product_resized.mode == "RGBA" else None

        for EnhancerClass in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
            factor = random.uniform(0.85, 1.15)
            rgb = EnhancerClass(rgb).enhance(factor)

        if alpha:
            product_resized = rgb.copy()
            product_resized.putalpha(alpha)
        else:
            product_resized = rgb

    # Paste
    result = shelf_img.copy()
    if product_resized.mode == "RGBA":
        result.paste(product_resized, (x, y), product_resized.split()[3])
    else:
        result.paste(product_resized, (x, y))

    return result


def generate_augmented_dataset(
    num_augmented_per_rare_class: int = 50,
    rare_threshold: int = 20,
):
    """Generate augmented images focusing on rare classes."""
    print("Loading annotations...")
    with open(ANNOTATIONS) as f:
        coco = json.load(f)

    categories = coco["categories"]
    cat_names = {c["id"]: c["name"] for c in categories}

    # Count annotations per category
    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    # Identify rare categories
    rare_cats = {cid for cid, count in cat_counts.items() if count < rare_threshold}
    print(f"Rare categories (< {rare_threshold} annotations): {len(rare_cats)}")

    # Load barcode mapping if available
    mapping_path = OUTPUT_DIR / "barcode_category_mapping.json"
    barcode_mapping = {}
    if mapping_path.exists():
        with open(mapping_path) as f:
            barcode_mapping = json.load(f)
        print(f"Loaded barcode mapping for {len(barcode_mapping)} products")

    # Build category -> product reference images mapping
    cat_to_ref_images = defaultdict(list)
    for barcode, info in barcode_mapping.items():
        cat_id = info["category_id"]
        confidence = info["confidence"]
        if confidence > 0.3:  # Only use reasonably confident matches
            ref_dir = PRODUCT_DIR / barcode
            if ref_dir.exists():
                # Prefer front/main views
                for view in ["front.jpg", "main.jpg"]:
                    p = ref_dir / view
                    if p.exists():
                        cat_to_ref_images[cat_id].append(p)
                        break
                else:
                    # Use any available image
                    imgs = list(ref_dir.glob("*.jpg"))
                    if imgs:
                        cat_to_ref_images[cat_id].append(imgs[0])

    print(f"Categories with reference images: {len(cat_to_ref_images)}")

    # Get shelf layout info
    shelf_regions = get_shelf_regions(coco)

    # Collect training images as backgrounds
    img_dir = COCO_DIR / "images"
    bg_images = sorted(img_dir.glob("*.jpg"))
    print(f"Background images: {len(bg_images)}")

    # Create output directories
    aug_img_dir = AUG_DIR / "images"
    aug_img_dir.mkdir(parents=True, exist_ok=True)

    # Generate augmented annotations
    new_images = []
    new_annotations = []
    ann_id_counter = max(ann["id"] for ann in coco["annotations"]) + 1
    img_id_counter = max(img["id"] for img in coco["images"]) + 1

    # For each rare category that has reference images
    augmented_count = 0
    for cat_id in sorted(rare_cats):
        ref_images = cat_to_ref_images.get(cat_id, [])
        if not ref_images:
            continue

        current_count = cat_counts.get(cat_id, 0)
        needed = num_augmented_per_rare_class - current_count
        if needed <= 0:
            continue

        cat_name = cat_names.get(cat_id, f"unknown_{cat_id}")
        print(f"  Augmenting cat {cat_id} ({cat_name}): {current_count} -> +{needed}")

        for aug_i in range(needed):
            # Pick random background
            bg_path = random.choice(bg_images)
            bg_img = Image.open(bg_path).convert("RGB")

            # Pick random reference image
            ref_path = random.choice(ref_images)
            try:
                ref_img = Image.open(ref_path).convert("RGB")
                product_rgba, mask = segment_product(ref_img)
            except Exception as e:
                print(f"    Warning: failed to process {ref_path}: {e}")
                continue

            # Get shelf region info for this background
            bg_img_id = None
            for img in coco["images"]:
                if img["file_name"] == bg_path.name:
                    bg_img_id = img["id"]
                    break

            if bg_img_id and bg_img_id in shelf_regions:
                region = shelf_regions[bg_img_id]
                avg_w = int(region["avg_box_width"])
                avg_h = int(region["avg_box_height"])
            else:
                avg_w, avg_h = 120, 150

            # Random size variation
            target_w = int(avg_w * random.uniform(0.7, 1.3))
            target_h = int(avg_h * random.uniform(0.7, 1.3))

            # Random position (within image bounds)
            max_x = max(1, bg_img.width - target_w)
            max_y = max(1, bg_img.height - target_h)
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            # Paste product
            aug_img = paste_product_on_shelf(bg_img, product_rgba, target_w, target_h, x, y)

            # Save augmented image
            aug_filename = f"aug_{cat_id:03d}_{aug_i:04d}.jpg"
            aug_img.save(aug_img_dir / aug_filename, quality=95)

            # Create annotation
            new_images.append({
                "id": img_id_counter,
                "file_name": aug_filename,
                "width": aug_img.width,
                "height": aug_img.height,
            })
            new_annotations.append({
                "id": ann_id_counter,
                "image_id": img_id_counter,
                "category_id": cat_id,
                "bbox": [x, y, target_w, target_h],
                "area": target_w * target_h,
                "iscrowd": 0,
            })

            ann_id_counter += 1
            img_id_counter += 1
            augmented_count += 1

    # Save augmented annotations as separate COCO file
    aug_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": categories,
    }
    aug_ann_path = AUG_DIR / "annotations.json"
    with open(aug_ann_path, "w") as f:
        json.dump(aug_coco, f, indent=2)

    # Also save combined dataset (original + augmented)
    combined_coco = {
        "images": coco["images"] + new_images,
        "annotations": coco["annotations"] + new_annotations,
        "categories": categories,
    }
    combined_path = OUTPUT_DIR / "combined_annotations.json"
    with open(combined_path, "w") as f:
        json.dump(combined_coco, f, indent=2)

    print(f"\n=== AUGMENTATION COMPLETE ===")
    print(f"Generated {augmented_count} augmented images")
    print(f"Augmented annotations: {aug_ann_path}")
    print(f"Combined dataset: {combined_path}")

    # Print updated class distribution stats
    combined_counts = Counter(ann["category_id"] for ann in combined_coco["annotations"])
    counts = sorted(combined_counts.values())
    print(f"\nCombined dataset stats:")
    print(f"  Total images: {len(combined_coco['images'])}")
    print(f"  Total annotations: {len(combined_coco['annotations'])}")
    print(f"  Min per class: {counts[0]}")
    print(f"  Median per class: {np.median(counts):.0f}")


if __name__ == "__main__":
    generate_augmented_dataset(
        num_augmented_per_rare_class=50,
        rare_threshold=20,
    )

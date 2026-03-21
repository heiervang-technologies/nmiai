#!/usr/bin/env python3
"""
Synthetic data generation pipeline for NM i AI object detection.

Strategy:
1. Generate empty shelf backgrounds using FLUX.1-schnell via HuggingFace
2. Composite product cutouts onto shelves at known positions
3. Output COCO-format annotations with bounding boxes

This gives us unlimited annotated training data for free.
"""

import json
import os
import random
import time
import io
import sys
from pathlib import Path
from collections import Counter

import requests
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

# Config
HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_API_URL = "https://router.huggingface.co/hf-inference/models/black-forest-labs/FLUX.1-schnell"
BASE_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data")
CUTOUTS_DIR = BASE_DIR / "product_cutouts"
CROPS_DIR = BASE_DIR / "extracted_crops"
COCO_ANNOT = BASE_DIR / "coco_dataset/train/annotations.json"
OUTPUT_DIR = BASE_DIR / "synthetic_dataset"
OUTPUT_IMAGES = OUTPUT_DIR / "images"
OUTPUT_ANNOT = OUTPUT_DIR / "annotations.json"

# Shelf background prompts
SHELF_PROMPTS = [
    "Empty metal grocery store shelves in a Norwegian supermarket, 4 shelves, well lit, no products, clean white price tags on shelf edges, realistic photo",
    "Empty wooden grocery store shelving unit, 5 shelves, bright fluorescent lighting, Norwegian supermarket interior, no products on shelves, realistic photograph",
    "Clean empty supermarket shelf unit with metal brackets, 4 rows, overhead lighting, white price labels, no products, photorealistic",
    "Empty grocery store aisle shelving, Norwegian store, 3-4 metal shelves, bright lighting from above, realistic photo, no items on shelves",
    "Blank store shelf display unit in a well-lit Scandinavian grocery store, 4 tiers, metal shelving with white labels, no products, photo",
    "Empty refrigerated shelf unit in a Norwegian grocery store, 3 shelves, cold section, glass doors open, no products, realistic",
    "Empty convenience store shelving, small format, 3-4 wooden shelves, warm lighting, Nordic style, no products, photo",
    "Empty supermarket endcap display, metal shelving, 4 levels, bright store lighting, Norwegian style, realistic photograph",
]

# Additional prompts for variety - shelf sections of different stores
SHELF_PROMPTS_SPECIFIC = [
    "Empty cereal and breakfast aisle shelves in Norwegian supermarket, 4 metal shelves, bright lighting, no products, realistic photo",
    "Empty coffee and tea section shelves in Scandinavian grocery store, 4 shelf tiers, price tags visible, no products, photorealistic",
    "Empty egg and dairy refrigerated display in Norwegian store, 3 shelves, cool lighting, no products, realistic",
    "Empty snack and chips aisle in a Norwegian grocery store, 4 shelves, bright overhead lights, no products, photo",
    "Empty baking supplies shelf section in supermarket, 4 metal shelves, white price labels, clean, no products, realistic photo",
]


def generate_shelf_background(prompt: str, retries: int = 3) -> Image.Image | None:
    """Generate an empty shelf background using FLUX.1-schnell."""
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    for attempt in range(retries):
        try:
            response = requests.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": prompt},
                timeout=120,
            )
            if response.status_code == 200 and "image" in response.headers.get("content-type", ""):
                img = Image.open(io.BytesIO(response.content)).convert("RGB")
                # Upscale to a more useful resolution
                img = img.resize((1280, 1024), Image.LANCZOS)
                return img
            elif response.status_code == 503:
                print(f"  Model loading, waiting 10s... (attempt {attempt+1})")
                time.sleep(10)
            else:
                print(f"  API error {response.status_code}: {response.text[:200]}")
                time.sleep(2)
        except Exception as e:
            print(f"  Request failed: {e}")
            time.sleep(2)
    return None


def make_blurred_background(real_img_path: Path) -> Image.Image:
    """Take a real shelf image and selectively blur to preserve shelf structure.

    Uses edge-preserving blur: heavy blur on product areas but keeps
    shelf lines and structural edges visible.
    """
    img = Image.open(real_img_path).convert("RGB")
    # Resize to standard size
    img = img.resize((1280, 1024), Image.LANCZOS)

    # Two-pass approach: detect edges, blend blurred with original at edges
    import numpy as np

    arr = np.array(img).astype(float)

    # Heavy blur for product removal
    heavily_blurred = img.filter(ImageFilter.GaussianBlur(radius=20))

    # Light blur to soften but keep structure
    lightly_blurred = img.filter(ImageFilter.GaussianBlur(radius=5))

    # Edge detection to find shelf lines (horizontal edges especially)
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    # Boost horizontal edges (shelf lines)
    edges_arr = np.array(edges).astype(float)
    # Dilate edges
    from PIL import ImageFilter as IF
    edge_mask = edges.filter(ImageFilter.GaussianBlur(radius=3))
    edge_arr = np.array(edge_mask).astype(float) / 255.0
    # Threshold to binary-ish mask
    edge_arr = np.clip(edge_arr * 3, 0, 1)

    # Blend: edges get lightly blurred (preserve structure), rest gets heavily blurred
    heavy_arr = np.array(heavily_blurred).astype(float)
    light_arr = np.array(lightly_blurred).astype(float)
    edge_3ch = edge_arr[:, :, np.newaxis]
    blended = heavy_arr * (1 - edge_3ch) + light_arr * edge_3ch
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    result = Image.fromarray(blended, "RGB")

    # Slight color shift for variety
    enhancer = ImageEnhance.Color(result)
    result = enhancer.enhance(random.uniform(0.85, 1.15))
    enhancer = ImageEnhance.Brightness(result)
    result = enhancer.enhance(random.uniform(0.9, 1.1))
    return result


def color_match_product(product_img: Image.Image, background: Image.Image, paste_x: int, paste_y: int) -> Image.Image:
    """Adjust product color temperature to match the local background region."""
    # Sample background region where product will be placed
    pw, ph = product_img.size
    # Get the background region (clamped to bounds)
    bg_w, bg_h = background.size
    x1 = max(0, paste_x)
    y1 = max(0, paste_y)
    x2 = min(bg_w, paste_x + pw)
    y2 = min(bg_h, paste_y + ph)

    if x2 <= x1 or y2 <= y1:
        return product_img

    bg_region = background.crop((x1, y1, x2, y2))
    bg_arr = np.array(bg_region).astype(float)
    prod_arr = np.array(product_img.convert("RGB")).astype(float)

    # Get mean color of background region and product
    bg_mean = bg_arr.mean(axis=(0, 1))
    prod_mean = prod_arr.mean(axis=(0, 1))

    # Gentle color shift (30% toward background color temperature)
    shift = (bg_mean - prod_mean) * 0.25
    prod_arr = np.clip(prod_arr + shift, 0, 255).astype(np.uint8)

    result = Image.fromarray(prod_arr, "RGB")
    # Re-apply original alpha channel
    result.putalpha(product_img.split()[3])
    return result


def load_cutouts_by_category() -> dict[int, list[Path]]:
    """Load product cutout paths organized by category ID."""
    cutouts = {}
    for f in CUTOUTS_DIR.glob("cat*_*.png"):
        try:
            cat_id = int(f.name.split("_")[0].replace("cat", ""))
            cutouts.setdefault(cat_id, []).append(f)
        except ValueError:
            continue
    return cutouts


def load_crops_by_category() -> dict[int, list[Path]]:
    """Load extracted crop paths organized by category ID."""
    crops = {}
    for f in CROPS_DIR.glob("cat*_crop*.png"):
        try:
            cat_id = int(f.name.split("_")[0].replace("cat", ""))
            crops.setdefault(cat_id, []).append(f)
        except ValueError:
            continue
    return crops


def get_underrepresented_categories(min_count: int = 30) -> list[tuple[int, str, int]]:
    """Get categories with fewer than min_count annotations."""
    with open(COCO_ANNOT) as f:
        data = json.load(f)

    cat_counts = Counter(ann["category_id"] for ann in data["annotations"])
    cat_names = {c["id"]: c["name"] for c in data["categories"]}

    under = []
    for cat_id, name in cat_names.items():
        count = cat_counts.get(cat_id, 0)
        if count < min_count:
            under.append((cat_id, name, count))

    under.sort(key=lambda x: x[2])
    return under


def augment_cutout(img: Image.Image) -> Image.Image:
    """Apply random augmentations to a product cutout."""
    # Random brightness
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.7, 1.3))

    # Random contrast
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Random slight rotation
    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=(0, 0, 0, 0))

    # Random slight blur
    if random.random() < 0.2:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    return img


def add_shadow(product_img: Image.Image, offset: int = 3, blur: float = 4.0) -> Image.Image:
    """Add a subtle drop shadow beneath the product for realism."""
    # Create shadow from alpha channel
    shadow = Image.new("RGBA", product_img.size, (0, 0, 0, 0))
    alpha = product_img.split()[3]
    shadow_layer = Image.new("RGBA", product_img.size, (0, 0, 0, 60))
    shadow_layer.putalpha(alpha)
    shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=blur))

    # Composite: shadow shifted down, then product on top
    result_size = (product_img.width, product_img.height + offset + int(blur))
    result = Image.new("RGBA", result_size, (0, 0, 0, 0))
    result.paste(shadow_layer, (0, offset), shadow_layer)
    result.paste(product_img, (0, 0), product_img)
    return result


def define_shelf_rows(img_w: int, img_h: int) -> list[dict]:
    """Define shelf rows as horizontal bands where product bottoms should sit.

    Returns rows sorted top-to-bottom, each with a y_bottom (where products sit)
    and a max product height.
    """
    # Model shelves as 3-5 horizontal rows occupying the middle 80% of image
    n_shelves = random.randint(3, 5)
    top_margin = img_h * 0.08
    bottom_margin = img_h * 0.05
    usable_h = img_h - top_margin - bottom_margin
    row_height = usable_h / n_shelves

    rows = []
    for i in range(n_shelves):
        # y_bottom is where product bottoms rest (shelf surface)
        y_bottom = int(top_margin + row_height * (i + 1) - row_height * 0.05)
        # max product height is most of the row height (products shouldn't overlap row above)
        max_h = int(row_height * 0.85)
        rows.append({
            "y_bottom": y_bottom,
            "max_h": max_h,
            "row_idx": i,
        })
    return rows


def composite_products_on_shelf(
    background: Image.Image,
    cutouts_by_cat: dict[int, list[Path]],
    crops_by_cat: dict[int, list[Path]],
    target_categories: list[int],
    products_per_image: int = 40,
) -> tuple[Image.Image, list[dict]]:
    """Place product cutouts densely on shelf rows with realistic positioning."""
    img = background.copy()
    img_w, img_h = img.size
    annotations = []

    rows = define_shelf_rows(img_w, img_h)

    for row in rows:
        # Fill this shelf row left-to-right with products, densely packed
        x_cursor = random.randint(5, 30)  # small left margin
        right_margin = img_w - random.randint(5, 30)

        while x_cursor < right_margin:
            # Pick a category - weighted toward underrepresented
            if target_categories and random.random() < 0.7:
                cat_id = random.choice(target_categories)
            else:
                cat_id = random.choice(list(cutouts_by_cat.keys()))

            # Get a cutout or crop for this category
            sources = []
            if cat_id in cutouts_by_cat:
                sources.extend(cutouts_by_cat[cat_id])
            if cat_id in crops_by_cat:
                sources.extend(crops_by_cat[cat_id])

            if not sources:
                x_cursor += 20
                continue

            src_path = random.choice(sources)
            try:
                product_img = Image.open(src_path).convert("RGBA")
            except Exception:
                x_cursor += 20
                continue

            # Target height: 60-95% of max row height for variety
            target_h = int(row["max_h"] * random.uniform(0.6, 0.95))
            if target_h < 30:
                x_cursor += 20
                continue

            # Scale maintaining aspect ratio
            aspect = product_img.width / max(product_img.height, 1)
            new_h = target_h
            new_w = max(int(target_h * aspect), 20)

            # Cap width so single product doesn't span entire shelf
            max_w = int(img_w * 0.15)
            if new_w > max_w:
                new_w = max_w
                new_h = max(int(new_w / aspect), 20)

            product_img = product_img.resize((new_w, new_h), Image.LANCZOS)
            product_img = augment_cutout(product_img)

            # Color-match product to local background
            x_tentative = x_cursor + random.randint(-2, 5)
            y_tentative = row["y_bottom"] - product_img.height
            product_img = color_match_product(product_img, img, x_tentative, y_tentative)

            # Add subtle shadow
            if random.random() < 0.6:
                product_img = add_shadow(product_img, offset=2, blur=3.0)

            # Position: bottom-aligned to shelf surface, tight horizontal packing
            x = x_tentative
            y = y_tentative

            # Clamp to image bounds
            if x + product_img.width > img_w or y < 0:
                x_cursor += new_w + random.randint(2, 8)
                continue

            # Paste with alpha
            img.paste(product_img, (x, y), product_img)

            # Record annotation - use the product size before shadow was added
            # (shadow adds a few pixels at bottom we don't want in bbox)
            bbox_h = new_h  # original product height
            bbox_y = row["y_bottom"] - bbox_h
            bbox = [x, bbox_y, new_w, bbox_h]
            annotations.append({
                "category_id": cat_id,
                "bbox": bbox,
                "area": new_w * bbox_h,
                "iscrowd": 0,
            })

            # Advance cursor - very tight packing, products nearly touching
            x_cursor += new_w + random.randint(-3, 4)

            if len(annotations) >= products_per_image:
                break

        if len(annotations) >= products_per_image:
            break

    return img, annotations


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate synthetic shelf images")
    parser.add_argument("--num-backgrounds", type=int, default=30, help="Number of shelf backgrounds to generate")
    parser.add_argument("--composites-per-bg", type=int, default=3, help="Composite variations per background")
    parser.add_argument("--products-per-image", type=int, default=45, help="Products per composite image")
    parser.add_argument("--min-count", type=int, default=30, help="Categories below this count are prioritized")
    args = parser.parse_args()

    # Setup output
    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)

    # Load resources
    print("Loading cutouts and crops...")
    cutouts_by_cat = load_cutouts_by_category()
    crops_by_cat = load_crops_by_category()
    print(f"  Cutouts: {len(cutouts_by_cat)} categories, {sum(len(v) for v in cutouts_by_cat.values())} files")
    print(f"  Crops: {len(crops_by_cat)} categories, {sum(len(v) for v in crops_by_cat.values())} files")

    # Get underrepresented categories
    under_cats = get_underrepresented_categories(args.min_count)
    target_cat_ids = [c[0] for c in under_cats if c[0] in cutouts_by_cat or c[0] in crops_by_cat]
    print(f"  Underrepresented categories (< {args.min_count} annotations): {len(under_cats)}")
    print(f"  With available cutouts/crops: {len(target_cat_ids)}")

    # Load COCO categories for output
    with open(COCO_ANNOT) as f:
        coco_data = json.load(f)
    categories = coco_data["categories"]

    # Generate
    all_prompts = SHELF_PROMPTS + SHELF_PROMPTS_SPECIFIC
    all_images = []
    all_annotations = []
    ann_id = 1
    img_id = 10000  # Start high to avoid collision with real data

    # Collect real images for blurred backgrounds
    real_images_dir = BASE_DIR / "coco_dataset/train/images"
    real_image_paths = sorted(real_images_dir.glob("*.jp*"))
    print(f"  Real images for blurred backgrounds: {len(real_image_paths)}")

    total_images = args.num_backgrounds * args.composites_per_bg
    print(f"\nGenerating {args.num_backgrounds} backgrounds x {args.composites_per_bg} composites = {total_images} synthetic images")
    print(f"Products per image: {args.products_per_image}")
    print(f"Strategy: 70% blurred real backgrounds, 30% FLUX-generated")
    print()

    for bg_idx in range(args.num_backgrounds):
        # 70% blurred real, 30% FLUX
        use_real = random.random() < 0.7 and real_image_paths

        if use_real:
            real_path = random.choice(real_image_paths)
            print(f"[{bg_idx+1}/{args.num_backgrounds}] Blurring real image: {real_path.name}")
            bg_img = make_blurred_background(real_path)
        else:
            prompt = random.choice(all_prompts)
            print(f"[{bg_idx+1}/{args.num_backgrounds}] Generating FLUX shelf background...")
            bg_img = generate_shelf_background(prompt)

        if bg_img is None:
            print(f"  FAILED to generate background, skipping")
            continue

        # Save raw background too (useful for debugging)
        bg_path = OUTPUT_IMAGES / f"bg_{bg_idx:04d}.jpg"
        bg_img.save(bg_path, quality=92)

        # Create composite variations
        for comp_idx in range(args.composites_per_bg):
            comp_img, annotations = composite_products_on_shelf(
                bg_img,
                cutouts_by_cat,
                crops_by_cat,
                target_cat_ids,
                args.products_per_image,
            )

            # Save composite
            img_filename = f"synth_{bg_idx:04d}_{comp_idx:02d}.jpg"
            img_path = OUTPUT_IMAGES / img_filename
            comp_img.save(img_path, quality=92)

            # Record in COCO format
            all_images.append({
                "id": img_id,
                "file_name": img_filename,
                "width": comp_img.width,
                "height": comp_img.height,
            })

            for ann in annotations:
                ann["id"] = ann_id
                ann["image_id"] = img_id
                all_annotations.append(ann)
                ann_id += 1

            img_id += 1
            print(f"  Composite {comp_idx+1}: {len(annotations)} products placed -> {img_filename}")

    # Save COCO annotations
    coco_output = {
        "images": all_images,
        "annotations": all_annotations,
        "categories": categories,
    }

    with open(OUTPUT_ANNOT, "w") as f:
        json.dump(coco_output, f)

    # Stats
    cat_counts = Counter(a["category_id"] for a in all_annotations)
    print(f"\n=== DONE ===")
    print(f"Generated {len(all_images)} synthetic images")
    print(f"Total annotations: {len(all_annotations)}")
    print(f"Categories covered: {len(cat_counts)}")
    print(f"Saved to: {OUTPUT_DIR}")

    # Show category boost
    target_boost = {cid: cat_counts.get(cid, 0) for cid in target_cat_ids}
    boosted = [(cid, cnt) for cid, cnt in target_boost.items() if cnt > 0]
    print(f"Underrepresented categories boosted: {len(boosted)}/{len(target_cat_ids)}")


if __name__ == "__main__":
    main()

"""
Silver Copy-Paste Augmentation Pipeline v2.

Improvements over existing copy_paste_augment.py:
1. Uses ALL visual sources: product_cutouts, scraped_products, extracted_crops
2. Multi-product paste per image (3-8 products for realistic density)
3. Shelf-row aligned placement (not random)
4. Targets ALL categories under 150 annotations, proportional boosting
5. Outputs YOLO format directly
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

random.seed(2026)
np.random.seed(2026)

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
COCO_IMGS = DATA_DIR / "coco_dataset" / "train" / "images"
PRODUCT_CUTOUTS = DATA_DIR / "product_cutouts"
SCRAPED_DIR = DATA_DIR / "scraped_products"
EXTRACTED_CROPS = DATA_DIR / "extracted_crops"
PRODUCT_IMAGES = DATA_DIR / "product_images"
STORE_PHOTOS = DATA_DIR / "store_photos"

OUTPUT_DIR = DATA_DIR / "silver_copypaste"
OUT_IMAGES = OUTPUT_DIR / "images"
OUT_LABELS = OUTPUT_DIR / "labels"

TARGET_MIN_ANNOTATIONS = 200  # boost weak cats toward this
MAX_IMAGES_TO_GENERATE = 2000


def load_category_sources():
    """Build mapping: category_id -> list of image paths (crops/cutouts/scraped)."""
    sources = defaultdict(list)

    # 1. Product cutouts (best quality - PNG with transparency)
    for f in PRODUCT_CUTOUTS.glob("*.png"):
        try:
            cat_id = int(f.stem.split("_")[0].replace("cat", ""))
            sources[cat_id].append(("cutout", f))
        except ValueError:
            pass

    # 2. Scraped product images
    for cat_dir in SCRAPED_DIR.iterdir():
        if cat_dir.is_dir():
            try:
                cat_id = int(cat_dir.name)
                for img in cat_dir.glob("*.jpg"):
                    sources[cat_id].append(("scraped", img))
                for img in cat_dir.glob("*.png"):
                    sources[cat_id].append(("scraped", img))
            except ValueError:
                pass

    # 3. Extracted training crops
    for f in EXTRACTED_CROPS.glob("*.png"):
        try:
            cat_id = int(f.stem.split("_")[0].replace("cat", ""))
            sources[cat_id].append(("crop", f))
        except ValueError:
            pass

    # 4. Product reference images (multi-angle studio shots)
    barcode_map_path = Path(__file__).parent / "outputs" / "barcode_category_mapping.json"
    if barcode_map_path.exists():
        with open(barcode_map_path) as f:
            barcode_map = json.load(f)
        for barcode, info in barcode_map.items():
            cat_id = info["category_id"]
            if info.get("confidence", 0) > 0.3:
                ref_dir = PRODUCT_IMAGES / barcode
                if ref_dir.exists():
                    for view in ["front.jpg", "main.jpg"]:
                        p = ref_dir / view
                        if p.exists():
                            sources[cat_id].append(("reference", p))
                            break
                    else:
                        imgs = list(ref_dir.glob("*.jpg"))
                        if imgs:
                            sources[cat_id].append(("reference", imgs[0]))

    return sources


def load_shelf_geometry():
    """Analyze training annotations for realistic shelf placement."""
    with open(COCO_ANN) as f:
        coco = json.load(f)

    img_map = {img["id"]: img for img in coco["images"]}

    # Group annotations by image, extract shelf rows
    shelf_rows = {}
    for img_info in coco["images"]:
        img_id = img_info["id"]
        anns = [a for a in coco["annotations"] if a["image_id"] == img_id]
        if not anns:
            continue

        # Cluster y-centers into shelf rows
        y_centers = sorted([(a["bbox"][1] + a["bbox"][3] / 2) for a in anns])
        heights = [a["bbox"][3] for a in anns]
        widths = [a["bbox"][2] for a in anns]

        # Simple row clustering: merge y-centers within median_height distance
        median_h = np.median(heights)
        rows = []
        current_row = [y_centers[0]]
        for yc in y_centers[1:]:
            if yc - current_row[-1] < median_h * 0.6:
                current_row.append(yc)
            else:
                rows.append(np.mean(current_row))
                current_row = [yc]
        rows.append(np.mean(current_row))

        shelf_rows[img_info["file_name"]] = {
            "rows": rows,
            "median_h": float(median_h),
            "median_w": float(np.median(widths)),
            "img_w": img_info["width"],
            "img_h": img_info["height"],
        }

    return shelf_rows


def load_product_image(source_type, path):
    """Load and prepare a product image for pasting."""
    img = Image.open(path)

    if source_type == "cutout":
        # Already has alpha channel
        return img.convert("RGBA")

    # For scraped/reference/crop: simple background removal
    img = img.convert("RGB")
    arr = np.array(img)

    # Detect white/light background
    bg_mask = np.all(arr > 230, axis=2)

    # Also detect very uniform regions
    gray = np.mean(arr, axis=2)
    local_var = np.zeros_like(gray)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
        local_var += np.abs(gray - shifted)
    uniform = local_var < 15

    bg = bg_mask & uniform

    # If background is less than 10% of image, it's probably a shelf crop - use as-is
    bg_ratio = bg.sum() / bg.size
    if bg_ratio < 0.10:
        # No significant background to remove, just convert
        alpha = np.full(arr.shape[:2], 255, dtype=np.uint8)
    else:
        alpha = np.where(bg, 0, 255).astype(np.uint8)

    alpha_img = Image.fromarray(alpha).filter(ImageFilter.MedianFilter(3))
    rgba = img.copy()
    rgba.putalpha(alpha_img)
    return rgba


def augment_product(product_rgba):
    """Apply random augmentations to product image."""
    rgb = product_rgba.convert("RGB")
    alpha = product_rgba.split()[3] if product_rgba.mode == "RGBA" else None

    # Color jitter
    for Enhancer in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
        factor = random.uniform(0.80, 1.20)
        rgb = Enhancer(rgb).enhance(factor)

    # Random slight blur (simulates distance/focus)
    if random.random() > 0.5:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    # Random slight rotation (-5 to +5 degrees)
    if random.random() > 0.6:
        angle = random.uniform(-5, 5)
        rgb = rgb.rotate(angle, expand=True, fillcolor=(255, 255, 255))
        if alpha:
            alpha = alpha.rotate(angle, expand=True, fillcolor=0)

    if alpha:
        # Resize alpha to match rgb if rotation changed size
        if rgb.size != alpha.size:
            alpha = alpha.resize(rgb.size, Image.LANCZOS)
        result = rgb.copy()
        result.putalpha(alpha)
        return result
    return rgb


def generate_silver_dataset():
    """Main generation pipeline."""
    print("Loading category sources...")
    sources = load_category_sources()
    print(f"  Categories with visual sources: {len(sources)}")
    total_sources = sum(len(v) for v in sources.values())
    print(f"  Total source images: {total_sources}")

    print("Loading shelf geometry...")
    shelf_geo = load_shelf_geometry()
    print(f"  Shelf layouts from {len(shelf_geo)} images")

    # Load current annotation counts from large_clean_split
    label_dir = DATA_DIR / "large_clean_split" / "train" / "labels"
    cat_counts = Counter()
    for f in label_dir.glob("*.txt"):
        for line in f.read_text().strip().split("\n"):
            if line.strip():
                cat_counts[int(line.split()[0])] += 1

    # Calculate how many annotations each category needs
    needs = {}
    for cat_id in range(356):
        current = cat_counts.get(cat_id, 0)
        deficit = TARGET_MIN_ANNOTATIONS - current
        if deficit > 0 and cat_id in sources:
            needs[cat_id] = deficit

    print(f"\nCategories needing boost: {len(needs)}")
    total_needed = sum(needs.values())
    print(f"Total annotations needed: {total_needed}")

    # Collect background images (training images + store photos)
    bg_images = sorted(COCO_IMGS.glob("*.jpg")) + sorted(COCO_IMGS.glob("*.jpeg"))
    store_bgs = sorted(STORE_PHOTOS.glob("*.jpg"))
    all_bgs = bg_images + store_bgs
    print(f"Background images: {len(all_bgs)} ({len(bg_images)} training + {len(store_bgs)} store)")

    # Setup output
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    # Build generation queue: distribute needed annotations into images
    # Each generated image will contain 3-8 products from various categories
    gen_queue = []
    for cat_id, deficit in sorted(needs.items(), key=lambda x: -x[1]):
        for _ in range(deficit):
            gen_queue.append(cat_id)

    random.shuffle(gen_queue)

    # Group into images of 3-8 products each
    products_per_image = 5  # average
    image_batches = []
    i = 0
    while i < len(gen_queue):
        batch_size = random.randint(3, 8)
        batch = gen_queue[i:i + batch_size]
        if len(batch) >= 2:  # at least 2 products per image
            image_batches.append(batch)
        i += batch_size

    # Cap at max images
    if len(image_batches) > MAX_IMAGES_TO_GENERATE:
        image_batches = image_batches[:MAX_IMAGES_TO_GENERATE]

    print(f"\nGenerating {len(image_batches)} augmented images...")

    generated_count = 0
    annotation_count = 0
    failed = 0

    for img_idx, batch_cats in enumerate(image_batches):
        if img_idx % 100 == 0:
            print(f"  Progress: {img_idx}/{len(image_batches)} images...")

        # Pick random background
        bg_path = random.choice(all_bgs)
        try:
            bg_img = Image.open(bg_path).convert("RGB")
        except Exception:
            failed += 1
            continue

        bg_w, bg_h = bg_img.size

        # Get shelf geometry for placement
        geo = shelf_geo.get(bg_path.name)
        if geo:
            shelf_rows_y = geo["rows"]
            median_w = geo["median_w"]
            median_h = geo["median_h"]
        else:
            # Estimate: divide image into 4-6 horizontal rows
            num_rows = random.randint(3, 5)
            row_height = bg_h / num_rows
            shelf_rows_y = [row_height * (i + 0.5) for i in range(num_rows)]
            median_w = bg_w * 0.05
            median_h = bg_h / num_rows * 0.7

        labels = []
        used_positions = []  # track placed boxes to avoid heavy overlap

        for cat_id in batch_cats:
            cat_sources = sources.get(cat_id, [])
            if not cat_sources:
                continue

            # Pick random source
            src_type, src_path = random.choice(cat_sources)

            try:
                product = load_product_image(src_type, src_path)
                product = augment_product(product)
            except Exception:
                continue

            # Size: based on shelf geometry with variation
            target_w = int(median_w * random.uniform(0.6, 1.4))
            target_h = int(median_h * random.uniform(0.6, 1.4))
            target_w = max(20, min(target_w, bg_w // 3))
            target_h = max(20, min(target_h, bg_h // 3))

            # Placement: pick a shelf row, random x position
            row_y = random.choice(shelf_rows_y)
            y = int(row_y - target_h / 2)
            y = max(0, min(y, bg_h - target_h))
            x = random.randint(0, max(0, bg_w - target_w))

            # Check overlap with existing placements
            overlap = False
            for px, py, pw, ph in used_positions:
                iou_x = max(0, min(x + target_w, px + pw) - max(x, px))
                iou_y = max(0, min(y + target_h, py + ph) - max(y, py))
                intersection = iou_x * iou_y
                area1 = target_w * target_h
                area2 = pw * ph
                iou = intersection / (area1 + area2 - intersection + 1e-6)
                if iou > 0.3:
                    overlap = True
                    break

            if overlap:
                # Try another position
                y_offset = random.choice([-1, 1]) * int(median_h * 0.8)
                y = max(0, min(y + y_offset, bg_h - target_h))
                x = random.randint(0, max(0, bg_w - target_w))

            # Resize and paste
            product_resized = product.resize((target_w, target_h), Image.LANCZOS)
            if product_resized.mode == "RGBA":
                bg_img.paste(product_resized, (x, y), product_resized.split()[3])
            else:
                bg_img.paste(product_resized, (x, y))

            used_positions.append((x, y, target_w, target_h))

            # YOLO label (normalized)
            cx = (x + target_w / 2) / bg_w
            cy = (y + target_h / 2) / bg_h
            nw = target_w / bg_w
            nh = target_h / bg_h
            labels.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            annotation_count += 1

        if labels:
            fname = f"silver_cp_{img_idx:05d}"
            bg_img.save(OUT_IMAGES / f"{fname}.jpg", quality=92)
            with open(OUT_LABELS / f"{fname}.txt", "w") as f:
                f.write("\n".join(labels) + "\n")
            generated_count += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"SILVER COPY-PASTE GENERATION COMPLETE")
    print(f"{'='*50}")
    print(f"Images generated: {generated_count}")
    print(f"Annotations generated: {annotation_count}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_DIR}")

    # Verify per-category distribution
    final_counts = Counter()
    for f in OUT_LABELS.glob("*.txt"):
        for line in f.read_text().strip().split("\n"):
            if line.strip():
                final_counts[int(line.split()[0])] += 1

    print(f"\nPer-category stats (silver data only):")
    for cat_id in sorted(final_counts.keys()):
        orig = cat_counts.get(cat_id, 0)
        silver = final_counts[cat_id]
        print(f"  cat {cat_id}: {orig} orig + {silver} silver = {orig + silver} total")


if __name__ == "__main__":
    generate_silver_dataset()

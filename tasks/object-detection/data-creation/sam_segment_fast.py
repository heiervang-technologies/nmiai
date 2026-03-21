"""
Fast SAM segmentation - only process unique reference product images.
Uses center-point prompt only (no automatic mask gen), skips fallbacks.
Targets: ~345 product reference images (1 per barcode, front view only).
Estimated time: ~1 hour on CPU.
"""
import json
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter
from collections import Counter, defaultdict

from segment_anything import sam_model_registry, SamPredictor

DATA_DIR = Path(__file__).parent / "data"
PRODUCT_IMAGES = DATA_DIR / "product_images"
SCRAPED_DIR = DATA_DIR / "scraped_products"
OUTPUT_DIR = DATA_DIR / "product_cutouts_sam"

SAM_CHECKPOINT = Path.home() / ".cache" / "sam" / "sam_vit_b_01ec64.pth"


def main():
    print("=" * 50)
    print("FAST SAM SEGMENTATION (reference images only)")
    print("=" * 50)

    # Load SAM
    print("Loading SAM ViT-B...")
    sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
    sam.to("cpu")
    predictor = SamPredictor(sam)
    print("SAM loaded")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load barcode mapping
    mapping_path = Path(__file__).parent / "outputs" / "barcode_category_mapping.json"
    barcode_map = {}
    if mapping_path.exists():
        with open(mapping_path) as f:
            barcode_map = json.load(f)

    # Collect ONE image per category (front/main view preferred)
    cat_to_image = {}

    # From product reference images
    for barcode, info in barcode_map.items():
        cat_id = info["category_id"]
        if cat_id in cat_to_image:
            continue
        ref_dir = PRODUCT_IMAGES / barcode
        if not ref_dir.exists():
            continue
        for view in ["front.jpg", "main.jpg"]:
            p = ref_dir / view
            if p.exists():
                cat_to_image[cat_id] = p
                break
        else:
            imgs = list(ref_dir.glob("*.jpg"))
            if imgs:
                cat_to_image[cat_id] = imgs[0]

    # From scraped for categories without reference
    for cat_dir in SCRAPED_DIR.iterdir():
        if not cat_dir.is_dir():
            continue
        try:
            cat_id = int(cat_dir.name)
        except ValueError:
            continue
        if cat_id in cat_to_image:
            continue
        imgs = list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png"))
        if imgs:
            cat_to_image[cat_id] = imgs[0]

    print(f"Categories to segment: {len(cat_to_image)}")

    processed = 0
    failed = 0

    for cat_id in sorted(cat_to_image.keys()):
        src_path = cat_to_image[cat_id]

        try:
            img = Image.open(src_path).convert("RGB")
            img_np = np.array(img)
            h, w = img_np.shape[:2]

            # Set image in SAM
            predictor.set_image(img_np)

            # Center point prompt
            center = np.array([[w // 2, h // 2]])
            label = np.array([1])

            masks, scores, _ = predictor.predict(
                point_coords=center,
                point_labels=label,
                multimask_output=True,
            )

            # Pick best mask
            best_idx = scores.argmax()
            mask = masks[best_idx]

            # Check quality
            coverage = mask.sum() / mask.size
            if coverage < 0.03 or coverage > 0.97:
                # Try with additional corner background points
                bg_points = np.array([[5, 5], [w-5, 5], [5, h-5], [w-5, h-5]])
                bg_labels = np.array([0, 0, 0, 0])
                all_points = np.vstack([center, bg_points])
                all_labels = np.concatenate([label, bg_labels])

                masks, scores, _ = predictor.predict(
                    point_coords=all_points,
                    point_labels=all_labels,
                    multimask_output=True,
                )
                best_idx = scores.argmax()
                mask = masks[best_idx]

            # Create RGBA
            alpha = (mask * 255).astype(np.uint8)
            alpha_pil = Image.fromarray(alpha).filter(ImageFilter.MedianFilter(3))

            rgba = img.copy()
            rgba.putalpha(alpha_pil)

            out_name = f"cat{cat_id:03d}_sam.png"
            rgba.save(OUTPUT_DIR / out_name)
            processed += 1

            if processed % 25 == 0:
                print(f"  Processed {processed}/{len(cat_to_image)}...")

        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  Failed cat {cat_id}: {e}")

    print(f"\n{'=' * 50}")
    print(f"SAM SEGMENTATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

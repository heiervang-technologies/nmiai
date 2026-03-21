"""
SAM-based product segmentation for high-quality copy-paste augmentation.

Uses SAM ViT-B to generate precise masks for:
1. Product reference images (studio shots with backgrounds)
2. Scraped product images
3. Training crop re-segmentation

Output: product_cutouts_sam/ with RGBA PNGs (transparent background)
"""
import numpy as np
from pathlib import Path
from PIL import Image
from collections import Counter

# SAM imports
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

DATA_DIR = Path(__file__).parent / "data"
PRODUCT_IMAGES = DATA_DIR / "product_images"
SCRAPED_DIR = DATA_DIR / "scraped_products"
EXTRACTED_CROPS = DATA_DIR / "extracted_crops"
EXISTING_CUTOUTS = DATA_DIR / "product_cutouts"
OUTPUT_DIR = DATA_DIR / "product_cutouts_sam"

SAM_CHECKPOINT = Path.home() / ".cache" / "sam" / "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"


def get_largest_foreground_mask(masks):
    """From SAM masks, pick the largest non-background mask."""
    if not masks:
        return None

    # Sort by area (largest first)
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)

    # The largest mask is often the background - skip it if it covers >70% of image
    img_area = masks[0]["segmentation"].size
    for mask in masks:
        coverage = mask["area"] / img_area
        if coverage < 0.70:
            return mask["segmentation"]

    # If all masks are large, use the second largest (first is background)
    if len(masks) > 1:
        return masks[1]["segmentation"]

    return masks[0]["segmentation"]


def segment_with_center_point(predictor, image_np):
    """Use SAM with a center point prompt to segment the main product."""
    h, w = image_np.shape[:2]
    center_point = np.array([[w // 2, h // 2]])
    center_label = np.array([1])  # foreground

    predictor.set_image(image_np)
    masks, scores, _ = predictor.predict(
        point_coords=center_point,
        point_labels=center_label,
        multimask_output=True,
    )

    # Pick highest scoring mask
    best_idx = scores.argmax()
    return masks[best_idx]


def apply_mask_to_image(image_pil, mask):
    """Apply binary mask to create RGBA image with transparent background."""
    img_array = np.array(image_pil.convert("RGB"))
    alpha = (mask * 255).astype(np.uint8)

    # Light morphological cleanup
    from PIL import ImageFilter
    alpha_pil = Image.fromarray(alpha).filter(ImageFilter.MedianFilter(3))
    alpha = np.array(alpha_pil)

    rgba = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = img_array
    rgba[:, :, 3] = alpha

    return Image.fromarray(rgba)


def main():
    import torch
    print("=" * 50)
    print("SAM PRODUCT SEGMENTATION")
    print("=" * 50)

    device = "cpu"  # GPU occupied by training
    print(f"Device: {device}")

    # Load SAM
    print(f"Loading SAM ViT-B from {SAM_CHECKPOINT}...")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device)
    predictor = SamPredictor(sam)
    print("SAM loaded")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load barcode mapping
    import json
    mapping_path = Path(__file__).parent / "outputs" / "barcode_category_mapping.json"
    barcode_map = {}
    if mapping_path.exists():
        with open(mapping_path) as f:
            barcode_map = json.load(f)

    # Build category -> source images
    from collections import defaultdict
    cat_sources = defaultdict(list)

    # 1. Product reference images (studio shots - best for SAM)
    for barcode, info in barcode_map.items():
        cat_id = info["category_id"]
        ref_dir = PRODUCT_IMAGES / barcode
        if ref_dir.exists():
            for img in ref_dir.glob("*.jpg"):
                cat_sources[cat_id].append(("reference", img))

    # 2. Scraped product images
    for cat_dir in SCRAPED_DIR.iterdir():
        if cat_dir.is_dir():
            try:
                cat_id = int(cat_dir.name)
                for img in list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png")):
                    cat_sources[cat_id].append(("scraped", img))
            except ValueError:
                pass

    total_sources = sum(len(v) for v in cat_sources.values())
    print(f"Source images: {total_sources} across {len(cat_sources)} categories")

    # Process each source image
    processed = 0
    failed = 0
    cat_counts = Counter()

    for cat_id in sorted(cat_sources.keys()):
        sources = cat_sources[cat_id]

        for src_type, src_path in sources:
            try:
                img = Image.open(src_path).convert("RGB")
                img_np = np.array(img)

                # Use center-point prompt (product is usually centered)
                mask = segment_with_center_point(predictor, img_np)

                # Check mask quality - should cover 10-90% of image
                coverage = mask.sum() / mask.size
                if coverage < 0.05 or coverage > 0.95:
                    # Fallback: use automatic mask generator
                    auto_gen = SamAutomaticMaskGenerator(
                        sam,
                        min_mask_region_area=100,
                        pred_iou_thresh=0.8,
                    )
                    auto_masks = auto_gen.generate(img_np)
                    mask = get_largest_foreground_mask(auto_masks)
                    if mask is None:
                        failed += 1
                        continue

                # Apply mask
                rgba = apply_mask_to_image(img, mask)

                # Save
                out_name = f"cat{cat_id:03d}_{src_type}_{cat_counts[cat_id]:02d}.png"
                rgba.save(OUTPUT_DIR / out_name)
                cat_counts[cat_id] += 1
                processed += 1

                if processed % 50 == 0:
                    print(f"  Processed {processed}/{total_sources}...")

            except Exception as e:
                failed += 1
                if failed <= 5:
                    print(f"  Failed {src_path.name}: {e}")

    print(f"\n{'=' * 50}")
    print(f"SAM SEGMENTATION COMPLETE")
    print(f"{'=' * 50}")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Categories covered: {len(cat_counts)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

"""
Extract product cutouts from reference images using SAM3.

For each product reference image (front/main view), runs SAM3 to segment
the product and saves an RGBA PNG with transparent background.

Usage:
    uv run python extract_cutouts.py
"""

import json
import pathlib
import torch
import numpy as np
from PIL import Image

PRODUCT_IMAGES_DIR = pathlib.Path(
    "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/product_images"
)
COCO_ANNOTATIONS = pathlib.Path(
    "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/coco_dataset/train/annotations.json"
)
OUTPUT_DIR = pathlib.Path(
    "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/product_cutouts_sam3"
)
METADATA_PATH = PRODUCT_IMAGES_DIR / "metadata.json"


def load_mappings():
    """Build barcode -> category_id mapping via product name matching."""
    with open(COCO_ANNOTATIONS) as f:
        coco = json.load(f)
    with open(METADATA_PATH) as f:
        meta = json.load(f)

    cat_by_name = {c["name"]: c["id"] for c in coco["categories"]}

    barcode_to_cat = {}
    for p in meta["products"]:
        name = p["product_name"]
        code = p["product_code"]
        if name in cat_by_name:
            barcode_to_cat[code] = {
                "category_id": cat_by_name[name],
                "name": name,
                "has_images": p["has_images"],
            }
    return barcode_to_cat


def load_sam3_model(device):
    """Load SAM3 model and processor."""
    from transformers import Sam3Model, Sam3Processor

    print("Loading SAM3 model...")
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    model.eval()
    processor = Sam3Processor.from_pretrained("facebook/sam3")
    print("SAM3 loaded.")
    return model, processor


def segment_product(image, model, processor, device):
    """Segment the main product from an image using SAM3 text prompt."""
    inputs = processor(
        images=image, text="product", return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=0.3,
        mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    if len(results["masks"]) == 0:
        return None

    # Take the largest mask (likely the main product)
    masks = results["masks"]
    scores = results["scores"]

    # Sort by mask area descending, pick largest
    areas = [m.sum().item() for m in masks]
    best_idx = max(range(len(areas)), key=lambda i: areas[i])

    return masks[best_idx].cpu().numpy().astype(np.uint8) * 255


def create_rgba_cutout(image, mask):
    """Create RGBA image with transparent background from mask."""
    img_array = np.array(image)
    # Create RGBA
    rgba = np.zeros((*img_array.shape[:2], 4), dtype=np.uint8)
    rgba[:, :, :3] = img_array
    rgba[:, :, 3] = mask

    # Crop to bounding box of mask
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    cropped = rgba[rmin : rmax + 1, cmin : cmax + 1]
    return Image.fromarray(cropped, "RGBA")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    barcode_to_cat = load_mappings()
    print(f"Found {len(barcode_to_cat)} barcode->category mappings")

    model, processor = load_sam3_model(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track results
    cutout_manifest = {}
    success = 0
    failed = 0

    product_dirs = sorted(
        [d for d in PRODUCT_IMAGES_DIR.iterdir() if d.is_dir()]
    )
    print(f"Processing {len(product_dirs)} product directories...")

    for i, product_dir in enumerate(product_dirs):
        barcode = product_dir.name
        if barcode not in barcode_to_cat:
            print(f"  [{i+1}/{len(product_dirs)}] SKIP {barcode} (no category mapping)")
            continue

        cat_info = barcode_to_cat[barcode]
        cat_id = cat_info["category_id"]

        # Prefer front.jpg, then main.jpg
        img_path = None
        for name in ["front.jpg", "main.jpg"]:
            candidate = product_dir / name
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            # Use any available jpg
            jpgs = list(product_dir.glob("*.jpg"))
            if jpgs:
                img_path = jpgs[0]

        if img_path is None:
            print(f"  [{i+1}/{len(product_dirs)}] SKIP {barcode} (no images)")
            failed += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            mask = segment_product(image, model, processor, device)

            if mask is None:
                print(f"  [{i+1}/{len(product_dirs)}] FAIL {barcode} (no mask)")
                failed += 1
                continue

            cutout = create_rgba_cutout(image, mask)
            if cutout is None:
                print(f"  [{i+1}/{len(product_dirs)}] FAIL {barcode} (empty crop)")
                failed += 1
                continue

            # Save with category_id in filename for easy lookup
            out_name = f"cat{cat_id:03d}_{barcode}.png"
            out_path = OUTPUT_DIR / out_name
            cutout.save(out_path)

            cutout_manifest[barcode] = {
                "category_id": cat_id,
                "name": cat_info["name"],
                "file": out_name,
                "width": cutout.width,
                "height": cutout.height,
            }
            success += 1
            if (i + 1) % 20 == 0:
                print(
                    f"  [{i+1}/{len(product_dirs)}] OK cat={cat_id} {cat_info['name'][:40]}"
                )

        except Exception as e:
            print(f"  [{i+1}/{len(product_dirs)}] ERROR {barcode}: {e}")
            failed += 1
            continue

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(cutout_manifest, f, indent=2)

    print(f"\nDone! Success: {success}, Failed: {failed}")
    print(f"Cutouts saved to: {OUTPUT_DIR}")
    print(f"Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()

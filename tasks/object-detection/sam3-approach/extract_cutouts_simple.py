"""
Fast fallback: extract product cutouts using simple background removal.

Many reference images have light/white studio backgrounds. This uses
GrabCut (OpenCV) with automatic initialization for faster processing
when SAM3 is overkill or too slow.

Usage:
    uv run python extract_cutouts_simple.py
"""

import json
import pathlib
import numpy as np
from PIL import Image

PRODUCT_IMAGES_DIR = pathlib.Path(
    "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/product_images"
)
COCO_ANNOTATIONS = pathlib.Path(
    "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/coco_dataset/train/annotations.json"
)
OUTPUT_DIR = pathlib.Path(
    "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/product_cutouts"
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


def remove_background_threshold(image: Image.Image, margin_pct=0.05):
    """Remove near-white background using adaptive thresholding.

    1. Sample corner pixels to estimate background color
    2. Create mask of pixels far from background
    3. Morphological cleanup
    4. Crop to content bounding box
    """
    img = np.array(image)
    h, w = img.shape[:2]

    # Sample corners for background color estimation
    margin_h = max(int(h * margin_pct), 5)
    margin_w = max(int(w * margin_pct), 5)
    corners = np.concatenate(
        [
            img[:margin_h, :margin_w].reshape(-1, 3),
            img[:margin_h, -margin_w:].reshape(-1, 3),
            img[-margin_h:, :margin_w].reshape(-1, 3),
            img[-margin_h:, -margin_w:].reshape(-1, 3),
        ]
    )
    bg_color = np.median(corners, axis=0)

    # Distance from background
    diff = np.sqrt(np.sum((img.astype(float) - bg_color) ** 2, axis=2))

    # Adaptive threshold — higher for lighter backgrounds
    bg_brightness = np.mean(bg_color)
    if bg_brightness > 200:
        threshold = 30  # White background
    elif bg_brightness > 150:
        threshold = 40  # Light background
    else:
        threshold = 50  # Darker background

    mask = (diff > threshold).astype(np.uint8) * 255

    # Morphological cleanup: close small holes, remove noise
    from PIL import ImageFilter

    mask_img = Image.fromarray(mask)
    # Dilate then erode (close)
    mask_img = mask_img.filter(ImageFilter.MaxFilter(5))
    mask_img = mask_img.filter(ImageFilter.MinFilter(5))
    # Remove small noise
    mask_img = mask_img.filter(ImageFilter.MinFilter(3))
    mask_img = mask_img.filter(ImageFilter.MaxFilter(3))
    mask = np.array(mask_img)

    # Check if mask covers a reasonable portion (10-95% of image)
    coverage = mask.sum() / (255 * h * w)
    if coverage < 0.05 or coverage > 0.98:
        return None

    # Create RGBA
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, :3] = img
    rgba[:, :, 3] = mask

    # Crop to bounding box
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    if not rows.any() or not cols.any():
        return None
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Add small padding
    pad = 3
    rmin = max(0, rmin - pad)
    rmax = min(h - 1, rmax + pad)
    cmin = max(0, cmin - pad)
    cmax = min(w - 1, cmax + pad)

    cropped = rgba[rmin : rmax + 1, cmin : cmax + 1]
    return Image.fromarray(cropped, "RGBA")


def main():
    barcode_to_cat = load_mappings()
    print(f"Found {len(barcode_to_cat)} barcode->category mappings")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
            jpgs = list(product_dir.glob("*.jpg"))
            if jpgs:
                img_path = jpgs[0]
        if img_path is None:
            failed += 1
            continue

        try:
            image = Image.open(img_path).convert("RGB")
            cutout = remove_background_threshold(image)

            if cutout is None:
                # Try with main.jpg if front.jpg failed
                alt_path = product_dir / "main.jpg"
                if alt_path.exists() and alt_path != img_path:
                    image = Image.open(alt_path).convert("RGB")
                    cutout = remove_background_threshold(image)

            if cutout is None:
                failed += 1
                if (i + 1) % 50 == 0:
                    print(f"  [{i+1}/{len(product_dirs)}] FAIL {barcode}")
                continue

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

        except Exception as e:
            print(f"  [{i+1}/{len(product_dirs)}] ERROR {barcode}: {e}")
            failed += 1

    # Save manifest
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(cutout_manifest, f, indent=2)

    print(f"\nDone! Success: {success}, Failed: {failed}")
    print(f"Cutouts saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

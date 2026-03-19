"""
Copy-paste augmentation: compose product cutouts onto shelf training images.

Takes:
  - Product cutouts (RGBA PNGs from extract_cutouts.py)
  - Original training images + COCO annotations
Produces:
  - Augmented images with pasted products
  - Updated COCO annotations (original + pasted bboxes)

Usage:
    uv run python augment_copypaste.py [--num-images 1000] [--pastes-per-image 10]
"""

import argparse
import json
import pathlib
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from collections import Counter

TRAIN_IMAGES_DIR = pathlib.Path(
    "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/coco_dataset/train/images"
)
COCO_ANNOTATIONS = pathlib.Path(
    "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/coco_dataset/train/annotations.json"
)
CUTOUTS_DIR = pathlib.Path(
    "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/product_cutouts"
)
OUTPUT_DIR = pathlib.Path(
    "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/augmented"
)


def load_coco():
    with open(COCO_ANNOTATIONS) as f:
        return json.load(f)


def load_cutouts():
    """Load cutout manifest and images."""
    manifest_path = CUTOUTS_DIR / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No manifest at {manifest_path}. Run extract_cutouts.py first."
        )
    with open(manifest_path) as f:
        manifest = json.load(f)

    cutouts = {}
    for barcode, info in manifest.items():
        img_path = CUTOUTS_DIR / info["file"]
        if img_path.exists():
            cutouts[barcode] = {
                **info,
                "path": img_path,
            }
    return cutouts


def get_category_weights(coco_data):
    """Weight rare categories higher for balanced augmentation."""
    cat_counts = Counter(a["category_id"] for a in coco_data["annotations"])
    max_count = max(cat_counts.values())
    # Inverse frequency weighting
    weights = {}
    for cat_id, count in cat_counts.items():
        weights[cat_id] = max_count / count
    return weights


def random_augment_cutout(cutout_img: Image.Image, target_h: int):
    """Apply random augmentation to a cutout before pasting.

    Args:
        cutout_img: RGBA cutout image
        target_h: Approximate target height on shelf (pixels)
    """
    # Scale to target height with some randomness
    scale = target_h / cutout_img.height
    scale *= random.uniform(0.8, 1.2)
    new_w = max(10, int(cutout_img.width * scale))
    new_h = max(10, int(cutout_img.height * scale))
    img = cutout_img.resize((new_w, new_h), Image.LANCZOS)

    # Slight rotation (-5 to 5 degrees)
    if random.random() < 0.3:
        angle = random.uniform(-5, 5)
        img = img.rotate(angle, expand=True, resample=Image.BICUBIC)

    # Brightness/contrast jitter
    if random.random() < 0.5:
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))
    if random.random() < 0.5:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(random.uniform(0.8, 1.2))

    # Slight blur to blend with background
    if random.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))

    return img


def paste_cutout(bg_image, cutout_rgba, x, y):
    """Paste an RGBA cutout onto a background image at (x, y).

    Returns the pasted bbox in COCO format [x, y, w, h] or None if out of bounds.
    """
    bg_w, bg_h = bg_image.size
    cw, ch = cutout_rgba.size

    # Clip to image boundaries
    paste_x = max(0, x)
    paste_y = max(0, y)
    # Ensure cutout doesn't extend past image
    if paste_x + cw > bg_w:
        cw = bg_w - paste_x
    if paste_y + ch > bg_h:
        ch = bg_h - paste_y
    if cw < 10 or ch < 10:
        return None

    # Crop cutout if needed
    crop_x = paste_x - x
    crop_y = paste_y - y
    cutout_cropped = cutout_rgba.crop((crop_x, crop_y, crop_x + cw, crop_y + ch))

    # Paste using alpha channel as mask
    bg_image.paste(cutout_cropped, (paste_x, paste_y), cutout_cropped)

    return [paste_x, paste_y, cw, ch]


def check_overlap(new_bbox, existing_bboxes, max_iou=0.3):
    """Check if new bbox overlaps too much with existing ones."""
    nx, ny, nw, nh = new_bbox
    for ex, ey, ew, eh in existing_bboxes:
        # Compute IoU
        ix1 = max(nx, ex)
        iy1 = max(ny, ey)
        ix2 = min(nx + nw, ex + ew)
        iy2 = min(ny + nh, ey + eh)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area_new = nw * nh
        area_ex = ew * eh
        union = area_new + area_ex - intersection
        iou = intersection / union if union > 0 else 0
        if iou > max_iou:
            return True
    return False


def generate_augmented_image(
    bg_image_path,
    original_annotations,
    cutouts,
    category_weights,
    num_pastes=10,
):
    """Generate one augmented image by pasting cutouts onto a training image.

    Returns (augmented_image, new_annotations_list).
    """
    bg = Image.open(bg_image_path).convert("RGB")
    bg_w, bg_h = bg.size

    # Start with existing annotations' bboxes
    existing_bboxes = [a["bbox"] for a in original_annotations]
    new_annotations = []

    # Estimate typical product height from existing annotations
    if original_annotations:
        heights = [a["bbox"][3] for a in original_annotations]
        typical_h = int(np.median(heights))
    else:
        typical_h = int(bg_h * 0.15)

    # Select cutouts weighted by category rarity
    cutout_list = list(cutouts.values())
    cutout_weights = []
    for c in cutout_list:
        cat_id = c["category_id"]
        cutout_weights.append(category_weights.get(cat_id, 1.0))

    total_weight = sum(cutout_weights)
    cutout_probs = [w / total_weight for w in cutout_weights]

    paste_count = 0
    max_attempts = num_pastes * 3

    for _ in range(max_attempts):
        if paste_count >= num_pastes:
            break

        # Pick a cutout (weighted toward rare categories)
        idx = np.random.choice(len(cutout_list), p=cutout_probs)
        cutout_info = cutout_list[idx]

        cutout_img = Image.open(cutout_info["path"])
        augmented = random_augment_cutout(cutout_img, typical_h)

        # Random position
        x = random.randint(0, max(1, bg_w - augmented.width))
        y = random.randint(0, max(1, bg_h - augmented.height))

        proposed_bbox = [x, y, augmented.width, augmented.height]

        # Skip if too much overlap
        if check_overlap(proposed_bbox, existing_bboxes, max_iou=0.3):
            continue

        actual_bbox = paste_cutout(bg, augmented, x, y)
        if actual_bbox is None:
            continue

        existing_bboxes.append(actual_bbox)
        new_annotations.append(
            {
                "bbox": actual_bbox,
                "category_id": cutout_info["category_id"],
                "area": actual_bbox[2] * actual_bbox[3],
                "iscrowd": 0,
            }
        )
        paste_count += 1

    return bg, new_annotations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-images",
        type=int,
        default=1000,
        help="Number of augmented images to generate",
    )
    parser.add_argument(
        "--pastes-per-image",
        type=int,
        default=10,
        help="Number of products to paste per image",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed"
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading data...")
    coco = load_coco()
    cutouts = load_cutouts()
    print(f"  {len(coco['images'])} training images")
    print(f"  {len(cutouts)} product cutouts available")

    if len(cutouts) == 0:
        print("ERROR: No cutouts found. Run extract_cutouts.py first.")
        return

    category_weights = get_category_weights(coco)

    # Build image_id -> annotations lookup
    img_annotations = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)

    # Build image_id -> file path lookup
    img_paths = {}
    for img_info in coco["images"]:
        img_paths[img_info["id"]] = TRAIN_IMAGES_DIR / img_info["file_name"]

    # Create output dirs
    out_images_dir = OUTPUT_DIR / "images"
    out_images_dir.mkdir(parents=True, exist_ok=True)

    # Generate augmented dataset
    aug_images = []
    aug_annotations = []
    next_ann_id = max(a["id"] for a in coco["annotations"]) + 1
    next_img_id = max(img["id"] for img in coco["images"]) + 1

    image_ids = list(img_paths.keys())

    print(f"Generating {args.num_images} augmented images...")
    for i in range(args.num_images):
        # Pick a random training image as background
        img_id = random.choice(image_ids)
        img_path = img_paths[img_id]
        orig_anns = img_annotations.get(img_id, [])

        # Find original image dimensions
        orig_info = next(
            im for im in coco["images"] if im["id"] == img_id
        )

        aug_img, new_anns = generate_augmented_image(
            img_path,
            orig_anns,
            cutouts,
            category_weights,
            num_pastes=args.pastes_per_image,
        )

        # Save augmented image
        aug_filename = f"aug_{i:05d}.jpg"
        aug_img.save(out_images_dir / aug_filename, quality=95)

        aug_img_id = next_img_id + i
        aug_images.append(
            {
                "id": aug_img_id,
                "file_name": aug_filename,
                "width": aug_img.width,
                "height": aug_img.height,
            }
        )

        # Add original annotations (re-IDed)
        for ann in orig_anns:
            aug_annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": aug_img_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": ann.get("iscrowd", 0),
                }
            )
            next_ann_id += 1

        # Add pasted annotations
        for ann in new_anns:
            aug_annotations.append(
                {
                    "id": next_ann_id,
                    "image_id": aug_img_id,
                    "category_id": ann["category_id"],
                    "bbox": ann["bbox"],
                    "area": ann["area"],
                    "iscrowd": 0,
                }
            )
            next_ann_id += 1

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{args.num_images} images")

    # Save augmented COCO annotations
    aug_coco = {
        "images": aug_images,
        "annotations": aug_annotations,
        "categories": coco["categories"],
    }
    ann_path = OUTPUT_DIR / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(aug_coco, f)

    # Also save a combined dataset (original + augmented)
    combined_coco = {
        "images": coco["images"] + aug_images,
        "annotations": coco["annotations"] + aug_annotations,
        "categories": coco["categories"],
    }
    combined_path = OUTPUT_DIR / "combined_annotations.json"
    with open(combined_path, "w") as f:
        json.dump(combined_coco, f)

    # Stats
    orig_cat_counts = Counter(a["category_id"] for a in coco["annotations"])
    aug_cat_counts = Counter(a["category_id"] for a in aug_annotations)
    combined_counts = Counter(
        a["category_id"] for a in coco["annotations"] + aug_annotations
    )
    cats_under_5 = sum(1 for c in combined_counts.values() if c < 5)

    print(f"\nDone!")
    print(f"  Original: {len(coco['images'])} images, {len(coco['annotations'])} annotations")
    print(f"  Augmented: {len(aug_images)} images, {len(aug_annotations)} annotations")
    print(f"  Combined: {len(combined_coco['images'])} images, {len(combined_coco['annotations'])} annotations")
    print(f"  Categories with <5 annotations (combined): {cats_under_5}")
    print(f"\nOutput:")
    print(f"  Images: {out_images_dir}")
    print(f"  Augmented annotations: {ann_path}")
    print(f"  Combined annotations: {combined_path}")


if __name__ == "__main__":
    main()

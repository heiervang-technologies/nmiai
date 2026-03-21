"""
Silver Copy-Paste v3 - uses SAM-segmented cutouts for highest quality augmentation.

Run AFTER sam_segment_fast.py completes.
Uses product_cutouts_sam/ (precise masks) instead of basic white-bg removal.
Targets: categories still under 500 annotations in mega_dataset.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

random.seed(3333)

DATA_DIR = Path(__file__).parent / "data"
SAM_CUTOUTS = DATA_DIR / "product_cutouts_sam"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
COCO_IMGS = DATA_DIR / "coco_dataset" / "train" / "images"
STORE_PHOTOS = DATA_DIR / "store_photos"
MEGA_LBLS = DATA_DIR / "mega_dataset" / "train" / "labels"

OUTPUT_DIR = DATA_DIR / "silver_copypaste_v3"
OUT_IMAGES = OUTPUT_DIR / "images"
OUT_LABELS = OUTPUT_DIR / "labels"

TARGET_MIN = 500


def load_shelf_geometry():
    with open(COCO_ANN) as f:
        coco = json.load(f)
    heights, widths = [], []
    for ann in coco["annotations"]:
        widths.append(ann["bbox"][2])
        heights.append(ann["bbox"][3])
    return float(np.median(widths)), float(np.median(heights))


def augment_product(rgba):
    rgb = rgba.convert("RGB")
    alpha = rgba.split()[3] if rgba.mode == "RGBA" else None

    for Enh in [ImageEnhance.Brightness, ImageEnhance.Contrast, ImageEnhance.Color]:
        rgb = Enh(rgb).enhance(random.uniform(0.8, 1.2))

    if random.random() > 0.5:
        rgb = rgb.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.2, 0.8)))

    if random.random() > 0.6:
        angle = random.uniform(-5, 5)
        rgb = rgb.rotate(angle, expand=True, fillcolor=(114, 114, 114))
        if alpha:
            alpha = alpha.rotate(angle, expand=True, fillcolor=0)

    if alpha:
        if rgb.size != alpha.size:
            alpha = alpha.resize(rgb.size, Image.LANCZOS)
        result = rgb.copy()
        result.putalpha(alpha)
        return result
    return rgb


def main():
    print("=" * 50)
    print("SILVER COPY-PASTE V3 (SAM cutouts)")
    print("=" * 50)

    # Check SAM cutouts exist
    sam_files = sorted(SAM_CUTOUTS.glob("*.png"))
    if not sam_files:
        print(f"ERROR: No SAM cutouts found in {SAM_CUTOUTS}")
        print("Run sam_segment_fast.py first!")
        return

    # Build category -> SAM cutout mapping
    cat_cutouts = defaultdict(list)
    for f in sam_files:
        try:
            cat_id = int(f.stem.split("_")[0].replace("cat", ""))
            cat_cutouts[cat_id].append(f)
        except ValueError:
            pass

    print(f"SAM cutouts: {len(sam_files)} across {len(cat_cutouts)} categories")

    # Current mega distribution
    cat_counts = Counter()
    for f in MEGA_LBLS.glob("*.txt"):
        for line in f.read_text().strip().split("\n"):
            if line.strip():
                cat_counts[int(line.split()[0])] += 1

    # Find categories needing boost that have SAM cutouts
    needs = {}
    for cat_id, cutouts in cat_cutouts.items():
        current = cat_counts.get(cat_id, 0)
        deficit = TARGET_MIN - current
        if deficit > 0:
            needs[cat_id] = deficit

    if not needs:
        print("All categories with SAM cutouts already above target!")
        # Still generate some for quality improvement
        needs = {cid: 50 for cid in cat_cutouts if cat_counts.get(cid, 0) < 1000}

    print(f"Categories to boost: {len(needs)}")

    median_w, median_h = load_shelf_geometry()
    bg_images = sorted(COCO_IMGS.glob("*.jpg")) + sorted(STORE_PHOTOS.glob("*.jpg"))

    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    # Build queue
    gen_queue = []
    for cat_id, deficit in sorted(needs.items(), key=lambda x: -x[1]):
        for _ in range(min(deficit, 200)):  # cap per category
            gen_queue.append(cat_id)
    random.shuffle(gen_queue)

    # Group into multi-product images
    generated = 0
    gen_anns = 0
    i = 0
    img_idx = 0

    while i < len(gen_queue):
        batch_size = random.randint(3, 8)
        batch = gen_queue[i:i + batch_size]
        if len(batch) < 2:
            i += batch_size
            continue

        bg_path = random.choice(bg_images)
        try:
            bg_img = Image.open(bg_path).convert("RGB")
        except Exception:
            i += batch_size
            continue

        bg_w, bg_h = bg_img.size
        labels = []

        for cat_id in batch:
            cutouts = cat_cutouts.get(cat_id, [])
            if not cutouts:
                continue

            cutout_path = random.choice(cutouts)
            try:
                product = Image.open(cutout_path).convert("RGBA")
                product = augment_product(product)
            except Exception:
                continue

            tw = int(median_w * random.uniform(0.6, 1.4))
            th = int(median_h * random.uniform(0.6, 1.4))
            tw = max(20, min(tw, bg_w // 3))
            th = max(20, min(th, bg_h // 3))

            x = random.randint(0, max(0, bg_w - tw))
            y = random.randint(0, max(0, bg_h - th))

            resized = product.resize((tw, th), Image.LANCZOS)
            bg_img.paste(resized, (x, y), resized.split()[3])

            cx = (x + tw / 2) / bg_w
            cy = (y + th / 2) / bg_h
            nw = tw / bg_w
            nh = th / bg_h
            labels.append(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            gen_anns += 1

        if labels:
            fname = f"silver_v3_{img_idx:05d}"
            bg_img.save(OUT_IMAGES / f"{fname}.jpg", quality=92)
            with open(OUT_LABELS / f"{fname}.txt", "w") as f:
                f.write("\n".join(labels) + "\n")
            generated += 1

        img_idx += 1
        i += batch_size

        if img_idx % 100 == 0:
            print(f"  {img_idx} images, {gen_anns} annotations...")

    print(f"\n{'=' * 50}")
    print(f"SILVER V3 (SAM) COMPLETE")
    print(f"{'=' * 50}")
    print(f"Images: {generated}")
    print(f"Annotations: {gen_anns}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

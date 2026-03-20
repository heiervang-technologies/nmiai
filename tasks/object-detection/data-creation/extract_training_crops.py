"""
Extract training crops for MarkusNet from multiple data sources.

Sources:
1. V3 augmented YOLO dataset (2565 images with labels)
2. SKU-PL mapped images (images with labels mapped to our 356 categories)

Output format matches existing cached_dataset:
  crops/{id}.jpg - cropped product image
  samples.json - [{crop_path, category_name, category_id}, ...]
"""
import json
from collections import Counter
from pathlib import Path

from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
OUTPUT_DIR = DATA_DIR / "extra_crops"
EXISTING_CROPS = Path("/home/me/ht/nmiai/tasks/object-detection/vlm-approach/cached_dataset")


def load_category_names():
    with open(COCO_ANN) as f:
        coco = json.load(f)
    return {c["id"]: c["name"] for c in coco["categories"]}


def extract_from_yolo_dataset(dataset_dir: Path, cat_names: dict, crop_id_start: int) -> tuple[list, int]:
    """Extract crops from a YOLO-format dataset."""
    samples = []
    crop_id = crop_id_start
    images_dir = dataset_dir / "train" / "images"
    labels_dir = dataset_dir / "train" / "labels"

    if not images_dir.exists():
        print(f"  No images at {images_dir}")
        return samples, crop_id

    label_files = sorted(labels_dir.glob("*.txt"))
    print(f"  Processing {len(label_files)} label files...")

    for label_file in label_files:
        # Find corresponding image (try .jpg and .jpeg)
        stem = label_file.stem
        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            candidate = images_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = candidate
                break
            # Handle symlinks
            if candidate.is_symlink():
                target = candidate.resolve()
                if target.exists():
                    img_path = target
                    break

        if img_path is None:
            continue

        # Read labels
        lines = label_file.read_text().strip().split("\n")
        if not lines or lines[0] == "":
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        img_w, img_h = img.size

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue

            cat_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            # Convert YOLO normalized to pixel coords
            x1 = int((cx - w / 2) * img_w)
            y1 = int((cy - h / 2) * img_h)
            x2 = int((cx + w / 2) * img_w)
            y2 = int((cy + h / 2) * img_h)

            # Clamp
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(img_w, x2)
            y2 = min(img_h, y2)

            # Skip tiny crops
            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            crop = img.crop((x1, y1, x2, y2))
            crop_path = OUTPUT_DIR / "crops" / f"{crop_id}.jpg"
            crop.save(crop_path, quality=90)

            cat_name = cat_names.get(cat_id, f"unknown_{cat_id}")
            samples.append({
                "crop_path": str(crop_path),
                "category_name": cat_name,
                "category_id": cat_id,
            })
            crop_id += 1

    return samples, crop_id


def extract_from_pl_mapped(cat_names: dict, crop_id_start: int) -> tuple[list, int]:
    """Extract crops from PL dataset using our category mapping."""
    PL_DIR = DATA_DIR / "external" / "skus_on_shelves_pl" / "extracted"
    MAPPING = Path(__file__).parent / "outputs" / "pl_to_ng_simple_mapping.json"

    if not MAPPING.exists() or not PL_DIR.exists():
        print("  PL mapping or data not found")
        return [], crop_id_start

    with open(MAPPING) as f:
        pl_to_ng = json.load(f)

    pl_ann_path = PL_DIR / "annotations.json"
    if not pl_ann_path.exists():
        print("  PL annotations not found")
        return [], crop_id_start

    print("  Loading PL annotations...")
    with open(pl_ann_path) as f:
        pl_coco = json.load(f)

    pl_img_map = {img["id"]: img for img in pl_coco["images"]}

    # Group mapped annotations by image, cap per category
    from collections import defaultdict
    MAX_PER_CAT = 100
    cat_counts = Counter()
    img_anns = defaultdict(list)

    for ann in pl_coco["annotations"]:
        pl_cat_str = str(ann["category_id"])
        if pl_cat_str not in pl_to_ng:
            continue
        ng_cat_id = pl_to_ng[pl_cat_str]
        if ng_cat_id == 355:  # skip unknown_product
            continue
        if cat_counts[ng_cat_id] >= MAX_PER_CAT:
            continue
        cat_counts[ng_cat_id] += 1
        img_anns[ann["image_id"]].append({
            "category_id": ng_cat_id,
            "bbox": ann["bbox"],  # COCO format: [x, y, w, h]
        })

    print(f"  {len(img_anns)} PL images with mapped annotations")
    print(f"  {sum(cat_counts.values())} total mapped crops to extract")

    samples = []
    crop_id = crop_id_start
    processed = 0

    for img_id, anns in img_anns.items():
        img_info = pl_img_map.get(img_id)
        if not img_info:
            continue

        img_path = PL_DIR / img_info["file_name"]
        if not img_path.exists():
            continue

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            continue

        img_w, img_h = img.size

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1 = max(0, int(x))
            y1 = max(0, int(y))
            x2 = min(img_w, int(x + w))
            y2 = min(img_h, int(y + h))

            if (x2 - x1) < 10 or (y2 - y1) < 10:
                continue

            crop = img.crop((x1, y1, x2, y2))
            crop_path = OUTPUT_DIR / "crops" / f"{crop_id}.jpg"
            crop.save(crop_path, quality=90)

            cat_name = cat_names.get(ann["category_id"], f"unknown_{ann['category_id']}")
            samples.append({
                "crop_path": str(crop_path),
                "category_name": cat_name,
                "category_id": ann["category_id"],
            })
            crop_id += 1

        processed += 1
        if processed % 500 == 0:
            print(f"    Processed {processed} images, {crop_id - crop_id_start} crops")

    return samples, crop_id


def main():
    cat_names = load_category_names()
    (OUTPUT_DIR / "crops").mkdir(parents=True, exist_ok=True)

    all_samples = []
    crop_id = 100000  # Start high to avoid conflicts with existing crop IDs

    # 1. V3 augmented dataset crops
    print("\n=== Source 1: YOLO V3 Augmented Dataset ===")
    v3_dir = DATA_DIR / "yolo_augmented_v3"
    if v3_dir.exists():
        samples, crop_id = extract_from_yolo_dataset(v3_dir, cat_names, crop_id)
        print(f"  Extracted: {len(samples)} crops")
        all_samples.extend(samples)

    # 2. SKU-PL mapped crops
    print("\n=== Source 2: SKU-PL Mapped Crops ===")
    samples, crop_id = extract_from_pl_mapped(cat_names, crop_id)
    print(f"  Extracted: {len(samples)} crops")
    all_samples.extend(samples)

    # Save samples.json
    samples_path = OUTPUT_DIR / "samples.json"
    with open(samples_path, "w") as f:
        json.dump(all_samples, f, indent=2, ensure_ascii=False)

    # Stats
    cat_counts = Counter(s["category_id"] for s in all_samples)
    print(f"\n=== SUMMARY ===")
    print(f"Total new crops: {len(all_samples)}")
    print(f"Categories covered: {len(cat_counts)}/356")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Samples JSON: {samples_path}")

    counts = sorted(cat_counts.values())
    if counts:
        print(f"Crops per category: min={counts[0]}, median={counts[len(counts)//2]}, max={counts[-1]}")

    # Also create a combined samples.json merging with existing
    if EXISTING_CROPS.exists():
        existing = json.loads((EXISTING_CROPS / "samples.json").read_text())
        combined = existing + all_samples
        combined_path = OUTPUT_DIR / "combined_samples.json"
        with open(combined_path, "w") as f:
            json.dump(combined, f, indent=2, ensure_ascii=False)
        print(f"\nCombined (existing + new): {len(combined)} total crops")
        print(f"Combined JSON: {combined_path}")


if __name__ == "__main__":
    main()

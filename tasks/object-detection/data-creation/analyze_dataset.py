"""Analyze the COCO dataset: class distribution, image stats, annotation stats."""
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
COCO_DIR = DATA_DIR / "coco_dataset" / "train"
PRODUCT_DIR = DATA_DIR / "product_images"
ANNOTATIONS = COCO_DIR / "annotations.json"

def main():
    with open(ANNOTATIONS) as f:
        coco = json.load(f)

    images = coco["images"]
    annotations = coco["annotations"]
    categories = coco["categories"]

    print(f"=== DATASET OVERVIEW ===")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")

    # Category mapping
    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    cat_id_to_supercategory = {c["id"]: c.get("supercategory", "unknown") for c in categories}

    # Class distribution
    cat_counts = Counter(ann["category_id"] for ann in annotations)
    print(f"\n=== CLASS DISTRIBUTION ===")
    counts = sorted(cat_counts.values())
    print(f"Min annotations per class: {counts[0] if counts else 0}")
    print(f"Max annotations per class: {counts[-1] if counts else 0}")
    print(f"Median: {np.median(counts):.0f}")
    print(f"Mean: {np.mean(counts):.1f}")
    print(f"Std: {np.std(counts):.1f}")

    # Categories with 0 annotations
    all_cat_ids = {c["id"] for c in categories}
    annotated_cats = set(cat_counts.keys())
    missing_cats = all_cat_ids - annotated_cats
    print(f"\nCategories with 0 annotations: {len(missing_cats)}")
    if missing_cats:
        print("Missing categories:")
        for cid in sorted(missing_cats):
            print(f"  - {cid}: {cat_id_to_name.get(cid, 'unknown')}")

    # Distribution buckets
    print(f"\n=== ANNOTATION COUNT BUCKETS ===")
    buckets = [(0, 0), (1, 5), (6, 10), (11, 20), (21, 50), (51, 100), (101, 200), (201, 500), (501, 10000)]
    for lo, hi in buckets:
        if lo == 0:
            count = len(missing_cats)
            print(f"  0 annotations: {count} categories")
        else:
            count = sum(1 for c in cat_counts.values() if lo <= c <= hi)
            print(f"  {lo}-{hi} annotations: {count} categories")

    # Bottom 30 classes
    print(f"\n=== BOTTOM 30 CLASSES (fewest annotations) ===")
    sorted_cats = sorted(cat_counts.items(), key=lambda x: x[1])
    for cat_id, count in sorted_cats[:30]:
        print(f"  {cat_id}: {cat_id_to_name.get(cat_id, 'unknown')} -> {count} annotations")

    # Top 20 classes
    print(f"\n=== TOP 20 CLASSES (most annotations) ===")
    for cat_id, count in sorted_cats[-20:]:
        print(f"  {cat_id}: {cat_id_to_name.get(cat_id, 'unknown')} -> {count} annotations")

    # Image stats
    print(f"\n=== IMAGE STATISTICS ===")
    widths = [img["width"] for img in images]
    heights = [img["height"] for img in images]
    print(f"Width range: {min(widths)}-{max(widths)}")
    print(f"Height range: {min(heights)}-{max(heights)}")
    print(f"Unique resolutions: {len(set(zip(widths, heights)))}")

    # Annotations per image
    anns_per_img = Counter(ann["image_id"] for ann in annotations)
    apic = sorted(anns_per_img.values())
    print(f"\n=== ANNOTATIONS PER IMAGE ===")
    print(f"Min: {apic[0]}")
    print(f"Max: {apic[-1]}")
    print(f"Mean: {np.mean(apic):.1f}")
    print(f"Median: {np.median(apic):.0f}")

    # Bounding box size stats
    print(f"\n=== BOUNDING BOX STATISTICS ===")
    bw = [ann["bbox"][2] for ann in annotations]
    bh = [ann["bbox"][3] for ann in annotations]
    areas = [w * h for w, h in zip(bw, bh)]
    print(f"Width range: {min(bw):.0f}-{max(bw):.0f}")
    print(f"Height range: {min(bh):.0f}-{max(bh):.0f}")
    print(f"Area range: {min(areas):.0f}-{max(areas):.0f}")
    print(f"Median area: {np.median(areas):.0f}")

    # Supercategory distribution
    print(f"\n=== SUPERCATEGORY DISTRIBUTION ===")
    supercat_counts = Counter()
    for cat_id, count in cat_counts.items():
        sc = cat_id_to_supercategory.get(cat_id, "unknown")
        supercat_counts[sc] += count
    for sc, count in supercat_counts.most_common():
        print(f"  {sc}: {count}")

    # Product images analysis
    print(f"\n=== PRODUCT REFERENCE IMAGES ===")
    product_dirs = list(PRODUCT_DIR.iterdir()) if PRODUCT_DIR.exists() else []
    print(f"Product folders: {len(product_dirs)}")
    if product_dirs:
        views_per_product = [len(list(d.iterdir())) for d in product_dirs if d.is_dir()]
        print(f"Views per product: min={min(views_per_product)}, max={max(views_per_product)}, mean={np.mean(views_per_product):.1f}")
        total_ref_images = sum(views_per_product)
        print(f"Total reference images: {total_ref_images}")

    # Check overlap: product folder names (barcodes?) vs category names
    product_folder_names = {d.name for d in product_dirs if d.is_dir()}
    print(f"\n=== CATEGORY-PRODUCT MAPPING ===")
    # Check if category names contain barcodes or if there's a mapping
    print("Sample category names:")
    for c in categories[:10]:
        print(f"  id={c['id']}, name='{c['name']}', supercategory='{c.get('supercategory', 'N/A')}'")
    print("Sample product folder names:")
    for name in sorted(product_folder_names)[:10]:
        print(f"  {name}")

    # Save full analysis to JSON
    analysis = {
        "overview": {
            "num_images": len(images),
            "num_annotations": len(annotations),
            "num_categories": len(categories),
        },
        "class_distribution": {
            cat_id_to_name.get(cid, str(cid)): count
            for cid, count in sorted(cat_counts.items(), key=lambda x: x[1])
        },
        "missing_categories": [
            {"id": cid, "name": cat_id_to_name.get(cid, "unknown")}
            for cid in sorted(missing_cats)
        ],
        "categories": [
            {"id": c["id"], "name": c["name"], "supercategory": c.get("supercategory", "unknown"), "count": cat_counts.get(c["id"], 0)}
            for c in categories
        ],
    }
    out_path = Path(__file__).parent / "dataset_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\nFull analysis saved to {out_path}")

if __name__ == "__main__":
    main()

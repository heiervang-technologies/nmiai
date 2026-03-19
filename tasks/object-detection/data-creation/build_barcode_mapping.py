"""
Build mapping between product reference image barcodes and COCO category IDs.

Strategy:
1. Try exact string matching between barcode folder names and category names
2. Use barcode lookup APIs to get product names, then fuzzy match
3. Use visual similarity (DINOv2) between reference images and annotated crops as fallback
"""
import json
import os
import re
from pathlib import Path
from difflib import SequenceMatcher

DATA_DIR = Path(__file__).parent / "data"
COCO_DIR = DATA_DIR / "coco_dataset" / "train"
PRODUCT_DIR = DATA_DIR / "product_images"
ANNOTATIONS = COCO_DIR / "annotations.json"

def normalize(name: str) -> str:
    """Normalize product name for comparison."""
    name = name.lower().strip()
    # Remove common weight/size patterns
    name = re.sub(r'\d+\s*(g|kg|ml|l|cl|stk|pos|kapsler)\b', '', name)
    # Remove extra spaces
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def main():
    with open(ANNOTATIONS) as f:
        coco = json.load(f)

    categories = coco["categories"]
    cat_names = {c["id"]: c["name"] for c in categories}

    product_dirs = sorted([d.name for d in PRODUCT_DIR.iterdir() if d.is_dir()])
    print(f"Product reference folders: {len(product_dirs)}")
    print(f"COCO categories: {len(categories)}")

    # Strategy: We'll use visual matching later. For now, save the data structures needed.
    # The key insight: we don't necessarily need name mapping.
    # For DINOv2 embeddings, we can embed ALL reference images and ALL training crops,
    # then use nearest-neighbor to establish the mapping.

    # For now, let's create a simple structure
    mapping = {
        "barcode_to_category": {},  # Will be filled by visual matching
        "product_barcodes": product_dirs,
        "category_names": cat_names,
        "unmapped_barcodes": product_dirs,  # All unmapped initially
    }

    # Save intermediate result
    out_path = Path(__file__).parent / "barcode_mapping.json"
    with open(out_path, "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")
    print(f"\nNote: Visual matching via DINOv2 will be used to complete this mapping.")
    print(f"This is more reliable than barcode API lookups for Norwegian products.")

if __name__ == "__main__":
    main()

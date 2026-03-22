#!/usr/bin/env python3
"""Recipe-based dataset builder for data mixture experiments.

Builds YOLO-format training datasets from configurable source mixtures.
Validation set is ALWAYS the full 248 competition images (converted to YOLO format).

Usage:
  python build_data_mixture.py recipe.json
  python build_data_mixture.py recipe.json --dry-run
  python build_data_mixture.py --list-sources

Recipe format (JSON):
{
  "name": "exp001_polish_det_only",
  "sources": [
    {
      "name": "polish_shelves",
      "type": "coco",
      "images": "data-creation/data/external/skus_on_shelves_pl/extracted/images",
      "annotations": "data-creation/data/external/skus_on_shelves_pl/extracted/annotations.json",
      "category_mapping": "detection_only",
      "max_images": null
    }
  ]
}

Category mapping options:
  - "detection_only": All categories → class 0 (single "product" class for detection transfer)
  - "identity": Keep original category IDs (only for competition data)
  - path to JSON: Custom mapping file {source_cat_id: target_cat_id}
"""
import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "data-creation" / "data"
COCO_DIR = BASE / "coco_dataset" / "train"
OUTPUT_BASE = BASE / "mixture_datasets"

# Category names from competition
CATEGORY_NAMES = None  # Loaded lazily


def load_competition_categories():
    """Load the 356 category names from competition annotations."""
    global CATEGORY_NAMES
    if CATEGORY_NAMES is not None:
        return CATEGORY_NAMES
    with open(COCO_DIR / "annotations.json") as f:
        coco = json.load(f)
    CATEGORY_NAMES = {c["id"]: c["name"] for c in coco["categories"]}
    return CATEGORY_NAMES


def coco_to_yolo_labels(annotations_path, images_dir, output_images, output_labels,
                        category_mapping, max_images=None):
    """Convert COCO annotations to YOLO format with category mapping.

    Returns: (num_images, num_annotations, category_counts)
    """
    with open(annotations_path) as f:
        coco = json.load(f)

    img_lookup = {img["id"]: img for img in coco["images"]}

    # Group annotations by image
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    # Load category mapping
    if category_mapping == "detection_only":
        cat_map = None  # All → 0
    elif category_mapping == "identity":
        cat_map = "identity"
    elif isinstance(category_mapping, (str, Path)) and Path(category_mapping).exists():
        with open(category_mapping) as f:
            raw = json.load(f)
        cat_map = {int(k): int(v) for k, v in raw.items()}
    else:
        cat_map = category_mapping  # Already a dict

    images_dir = Path(images_dir)
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    num_images = 0
    num_anns = 0
    cat_counts = Counter()
    skipped_missing = 0

    image_ids = sorted(img_lookup.keys())
    if max_images:
        image_ids = image_ids[:max_images]

    for img_id in image_ids:
        img_info = img_lookup[img_id]
        fname = img_info["file_name"]
        src_path = images_dir / fname

        if not src_path.exists():
            skipped_missing += 1
            continue

        w, h = img_info["width"], img_info["height"]
        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue

        # Convert annotations to YOLO format
        yolo_lines = []
        for ann in anns:
            src_cat = ann["category_id"]

            if cat_map is None:  # detection_only
                target_cat = 0
            elif cat_map == "identity":
                target_cat = src_cat
            elif isinstance(cat_map, dict):
                target_cat = cat_map.get(src_cat)
                if target_cat is None:
                    continue  # Skip unmapped categories
            else:
                target_cat = 0

            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / w
            cy = (by + bh / 2) / h
            nw = bw / w
            nh = bh / h

            # Clip to [0, 1]
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            nw = max(0, min(1, nw))
            nh = max(0, min(1, nh))

            if nw < 0.001 or nh < 0.001:
                continue

            yolo_lines.append(f"{target_cat} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            cat_counts[target_cat] += 1

        if not yolo_lines:
            continue

        # Copy image
        dst_img = output_images / fname
        if not dst_img.exists():
            shutil.copy2(src_path, dst_img)

        # Write label
        label_name = Path(fname).stem + ".txt"
        with open(output_labels / label_name, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")

        num_images += 1
        num_anns += len(yolo_lines)

    if skipped_missing:
        print(f"  Skipped {skipped_missing} missing images")

    return num_images, num_anns, cat_counts


def copy_yolo_source(source_images, source_labels, output_images, output_labels,
                     prefix="", max_images=None):
    """Copy existing YOLO-format data with optional prefix to avoid name collisions."""
    source_images = Path(source_images)
    source_labels = Path(source_labels)
    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    images = sorted(source_images.glob("*.jpg")) + sorted(source_images.glob("*.jpeg")) + sorted(source_images.glob("*.png"))
    if max_images:
        images = images[:max_images]

    num_images = 0
    num_anns = 0
    cat_counts = Counter()

    for img_path in images:
        label_path = source_labels / (img_path.stem + ".txt")
        if not label_path.exists():
            continue

        dst_name = f"{prefix}{img_path.name}" if prefix else img_path.name
        dst_label = f"{prefix}{img_path.stem}.txt" if prefix else f"{img_path.stem}.txt"

        shutil.copy2(img_path, output_images / dst_name)
        shutil.copy2(label_path, output_labels / dst_label)

        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cat_counts[int(parts[0])] += 1
                    num_anns += 1
        num_images += 1

    return num_images, num_anns, cat_counts


def build_val_set(output_dir):
    """Convert ALL 248 competition images to YOLO format for validation."""
    val_images = output_dir / "val" / "images"
    val_labels = output_dir / "val" / "labels"

    print("Building validation set (all 248 competition images)...")
    n_img, n_ann, _ = coco_to_yolo_labels(
        COCO_DIR / "annotations.json",
        COCO_DIR / "images",
        val_images, val_labels,
        category_mapping="identity",
    )
    print(f"  Val: {n_img} images, {n_ann} annotations")
    return {p.name for p in val_images.glob("*")}


def build_dataset(recipe_path, dry_run=False):
    """Build a YOLO dataset from a recipe."""
    with open(recipe_path) as f:
        recipe = json.load(f)

    name = recipe["name"]
    output_dir = OUTPUT_BASE / name

    print(f"\n{'=' * 60}")
    print(f"Building dataset: {name}")
    print(f"Output: {output_dir}")
    print(f"{'=' * 60}")

    if dry_run:
        print("[DRY RUN - no files will be created]")

    if not dry_run:
        if output_dir.exists():
            print(f"Cleaning existing output: {output_dir}")
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)

    # Build val set first
    if not dry_run:
        val_filenames = build_val_set(output_dir)
    else:
        val_filenames = set()

    train_images = output_dir / "train" / "images"
    train_labels = output_dir / "train" / "labels"

    total_images = 0
    total_anns = 0
    total_cats = Counter()
    source_summary = []

    for source in recipe["sources"]:
        src_name = source["name"]
        src_type = source.get("type", "coco")
        max_images = source.get("max_images")
        mapping = source.get("category_mapping", "detection_only")

        print(f"\nProcessing source: {src_name} ({src_type})")

        if dry_run:
            print(f"  Would process with mapping={mapping}, max_images={max_images}")
            continue

        if src_type == "coco":
            images_dir = SCRIPT_DIR / source["images"]
            annotations = SCRIPT_DIR / source["annotations"]
            n_img, n_ann, cats = coco_to_yolo_labels(
                annotations, images_dir,
                train_images, train_labels,
                category_mapping=mapping,
                max_images=max_images,
            )
        elif src_type == "yolo":
            src_images = SCRIPT_DIR / source["images"]
            src_labels = SCRIPT_DIR / source["labels"]
            prefix = source.get("prefix", "")
            n_img, n_ann, cats = copy_yolo_source(
                src_images, src_labels,
                train_images, train_labels,
                prefix=prefix,
                max_images=max_images,
            )
        else:
            print(f"  Unknown source type: {src_type}")
            continue

        print(f"  Added: {n_img} images, {n_ann} annotations")
        total_images += n_img
        total_anns += n_ann
        total_cats += cats
        source_summary.append(f"{src_name}({n_img})")

    if dry_run:
        return

    # Leakage check
    train_files = {p.name for p in train_images.glob("*")}
    overlap = train_files & val_filenames
    if overlap:
        print(f"\n*** LEAKAGE DETECTED: {len(overlap)} images in both train and val ***")
        for f in sorted(overlap)[:10]:
            print(f"  - {f}")
        print("ABORTING")
        return

    # Determine num_classes
    if total_cats:
        nc = max(total_cats.keys()) + 1
    else:
        nc = 356

    # Generate dataset.yaml
    cat_names = load_competition_categories()
    names_list = [cat_names.get(i, f"class_{i}") for i in range(nc)]

    yaml_content = f"path: {output_dir}\ntrain: train/images\nval: val/images\n\nnc: {nc}\nnames:\n"
    for i, n in enumerate(names_list):
        yaml_content += f"  {i}: \"{n}\"\n"

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    # Write manifest
    manifest = {
        "name": name,
        "train_images": total_images,
        "train_annotations": total_anns,
        "val_images": len(val_filenames),
        "num_classes": nc,
        "categories_with_data": len(total_cats),
        "leakage": "ZERO",
        "sources": source_summary,
        "category_distribution": {str(k): v for k, v in sorted(total_cats.items())},
    }
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Dataset: {name}")
    print(f"  Train: {total_images} images, {total_anns} annotations")
    print(f"  Val: {len(val_filenames)} images (all 248 competition)")
    print(f"  Classes: {nc} ({len(total_cats)} with data)")
    print(f"  Leakage: ZERO (verified)")
    print(f"  Sources: {', '.join(source_summary)}")
    print(f"  YAML: {yaml_path}")
    print(f"{'=' * 60}")

    # Category coverage report
    if nc == 356:
        coverage = {
            "zero": sum(1 for i in range(356) if total_cats.get(i, 0) == 0),
            "1-5": sum(1 for i in range(356) if 1 <= total_cats.get(i, 0) <= 5),
            "6-50": sum(1 for i in range(356) if 6 <= total_cats.get(i, 0) <= 50),
            "50+": sum(1 for i in range(356) if total_cats.get(i, 0) > 50),
        }
        print(f"  Category coverage: {coverage}")

    return str(yaml_path)


def list_sources():
    """List available data sources."""
    print("Available data sources:")
    print()

    sources = [
        ("Competition (VAL ONLY)", COCO_DIR / "images", "248 images, COCO format"),
        ("External Polish", BASE / "external", "27K+ images, COCO format"),
        ("Store photos", BASE / "store_photos", "39 images, unlabeled"),
        ("Product references", BASE / "product_images", "327 products"),
        ("Silver augmented", BASE / "silver_augmented_dataset", "6.5K images, YOLO format"),
        ("Pseudo-labeled", BASE / "pseudo_labels", "2K images, YOLO format"),
    ]

    for name, path, desc in sources:
        exists = "OK" if path.exists() else "MISSING"
        count = ""
        if path.exists() and path.is_dir():
            imgs = list(path.glob("**/*.jpg")) + list(path.glob("**/*.png"))
            count = f" ({len(imgs)} images)"
        print(f"  [{exists}] {name}: {path}{count}")
        print(f"         {desc}")


def main():
    parser = argparse.ArgumentParser(description="Build YOLO dataset from recipe")
    parser.add_argument("recipe", nargs="?", help="Path to recipe JSON")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--list-sources", action="store_true", help="List available data sources")
    args = parser.parse_args()

    if args.list_sources:
        list_sources()
        return

    if not args.recipe:
        parser.print_help()
        return

    build_dataset(args.recipe, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

"""
Merge silver-labeled data into the main YOLO training dataset.

Takes data from:
1. silver_copypaste/ - copy-paste augmented images
2. pseudo_labeled_stores/ - pseudo-labeled store photos
3. boosted_weak/ - existing weak category boosts

And merges into a new dataset: silver_augmented_dataset/
"""
import shutil
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Source datasets
SOURCES = {
    "silver_copypaste": DATA_DIR / "silver_copypaste",
    "pseudo_labeled": DATA_DIR / "pseudo_labeled_stores",
    "boosted_weak": DATA_DIR / "boosted_weak",
}

# Base dataset to augment
BASE_DATASET = DATA_DIR / "large_clean_split"
OUTPUT_DIR = DATA_DIR / "silver_augmented_dataset"


def count_labels(label_dir):
    """Count annotations per category in a YOLO label directory."""
    counts = Counter()
    total_images = 0
    for f in label_dir.glob("*.txt"):
        total_images += 1
        for line in f.read_text().strip().split("\n"):
            if line.strip():
                cat_id = int(line.split()[0])
                counts[cat_id] += 1
    return counts, total_images


def main():
    print("=" * 60)
    print("MERGING SILVER DATA INTO TRAINING DATASET")
    print("=" * 60)

    # Setup output
    out_train_img = OUTPUT_DIR / "train" / "images"
    out_train_lbl = OUTPUT_DIR / "train" / "labels"
    out_val_img = OUTPUT_DIR / "val" / "images"
    out_val_lbl = OUTPUT_DIR / "val" / "labels"

    for d in [out_train_img, out_train_lbl, out_val_img, out_val_lbl]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Copy base dataset
    print(f"\n1. Copying base dataset from {BASE_DATASET.name}...")
    base_train_img = BASE_DATASET / "train" / "images"
    base_train_lbl = BASE_DATASET / "train" / "labels"
    base_val_img = BASE_DATASET / "val" / "images"
    base_val_lbl = BASE_DATASET / "val" / "labels"

    copied_images = 0
    for img in base_train_img.glob("*"):
        shutil.copy2(str(img), str(out_train_img / img.name))
        copied_images += 1
    for lbl in base_train_lbl.glob("*.txt"):
        shutil.copy2(str(lbl), str(out_train_lbl / lbl.name))

    # Copy validation set as-is
    for img in base_val_img.glob("*"):
        shutil.copy2(str(img), str(out_val_img / img.name))
    for lbl in base_val_lbl.glob("*.txt"):
        shutil.copy2(str(lbl), str(out_val_lbl / lbl.name))

    print(f"   Copied {copied_images} base training images")

    # Step 2: Add silver data sources
    silver_images = 0
    silver_annotations = 0

    for source_name, source_dir in SOURCES.items():
        img_dir = source_dir / "images"
        lbl_dir = source_dir / "labels"

        if not img_dir.exists() or not lbl_dir.exists():
            print(f"\n2. Skipping {source_name} (not found)")
            continue

        img_count = len(list(img_dir.glob("*.jpg"))) + len(list(img_dir.glob("*.png")))
        lbl_count = len(list(lbl_dir.glob("*.txt")))
        print(f"\n2. Adding {source_name}: {img_count} images, {lbl_count} labels")

        for img in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
            dest = out_train_img / img.name
            if dest.exists():
                # Rename to avoid collision
                dest = out_train_img / f"{source_name}_{img.name}"
            shutil.copy2(str(img), str(dest))
            silver_images += 1

        for lbl in lbl_dir.glob("*.txt"):
            dest = out_train_lbl / lbl.name
            if dest.exists():
                dest = out_train_lbl / f"{source_name}_{lbl.name}"
            shutil.copy2(str(lbl), str(dest))

            # Count annotations
            for line in lbl.read_text().strip().split("\n"):
                if line.strip():
                    silver_annotations += 1

    # Step 3: Create dataset.yaml
    yaml_content = f"""path: {OUTPUT_DIR}
train: train/images
val: val/images

nc: 356
names:
"""
    # Load category names from base dataset yaml
    base_yaml = BASE_DATASET / "dataset.yaml"
    if base_yaml.exists():
        import re
        yaml_text = base_yaml.read_text()
        # Extract names section
        names_match = re.search(r'names:\s*\n((?:\s+\d+:.*\n)*)', yaml_text)
        if names_match:
            yaml_content += names_match.group(1)
        else:
            # Simple numbered names
            for i in range(356):
                yaml_content += f"  {i}: class_{i}\n"
    else:
        for i in range(356):
            yaml_content += f"  {i}: class_{i}\n"

    (OUTPUT_DIR / "dataset.yaml").write_text(yaml_content)

    # Step 4: Summary statistics
    print(f"\n{'=' * 60}")
    print("MERGE COMPLETE")
    print(f"{'=' * 60}")
    print(f"Base images:   {copied_images}")
    print(f"Silver images: {silver_images}")
    print(f"Total train:   {copied_images + silver_images}")
    print(f"Silver annotations added: {silver_annotations}")

    # Per-category comparison
    base_counts, _ = count_labels(base_train_lbl)
    final_counts, _ = count_labels(out_train_lbl)

    print(f"\nCategory distribution improvement:")
    print(f"{'Cat':>5} {'Base':>8} {'Final':>8} {'Gain':>8}")
    print("-" * 35)

    improved = 0
    for cat_id in sorted(final_counts.keys()):
        base_c = base_counts.get(cat_id, 0)
        final_c = final_counts.get(cat_id, 0)
        if final_c > base_c:
            improved += 1
            print(f"{cat_id:>5} {base_c:>8} {final_c:>8} {'+' + str(final_c - base_c):>8}")

    print(f"\nCategories improved: {improved}")
    print(f"Min annotations: {min(final_counts.values())}")
    print(f"Median annotations: {sorted(final_counts.values())[len(final_counts)//2]}")
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"YAML:   {OUTPUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

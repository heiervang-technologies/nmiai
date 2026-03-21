"""
Build ImageFolder-style classifier dataset from crops.
Structure: data/classifier_dataset/{train,val}/category_id/image.jpg
Uses stratified split with same val images as detection datasets.
"""
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
import shutil

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "classifier_dataset"
SEED = 42
VAL_RATIO = 0.15  # 15% for classifier val


def main():
    random.seed(SEED)

    # Load crop metadata
    samples = json.load(open(DATA_DIR / "extra_crops" / "clean_combined_samples.json"))
    print(f"Total crop samples: {len(samples)}")

    # Group by category
    cat_samples = defaultdict(list)
    for s in samples:
        crop_path = Path(s["crop_path"])
        if crop_path.exists():
            cat_samples[s["category_id"]].append(s)

    print(f"Categories with crops: {len(cat_samples)}")

    # Also add scraped product reference images
    scraped_dir = DATA_DIR / "scraped_products"
    ann = json.load(open(DATA_DIR / "coco_dataset" / "train" / "annotations.json"))
    cat_names = {c["id"]: c["name"] for c in ann["categories"]}

    scraped_added = 0
    for cat_id, name in cat_names.items():
        # Find matching scraped products
        cat_scraped_dir = scraped_dir / str(cat_id)
        if cat_scraped_dir.exists():
            for img_path in cat_scraped_dir.glob("*.jpg"):
                cat_samples[cat_id].append({
                    "crop_path": str(img_path),
                    "category_id": cat_id,
                    "category_name": name,
                    "source": "scraped",
                })
                scraped_added += 1

    print(f"Added {scraped_added} scraped product images")

    # Setup output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (OUTPUT_DIR / split).mkdir(exist_ok=True)

    total_train = 0
    total_val = 0
    cat_stats = {}

    for cat_id in sorted(cat_samples.keys()):
        samples_list = cat_samples[cat_id]
        random.shuffle(samples_list)

        n_val = max(1, int(len(samples_list) * VAL_RATIO))
        val_samples = samples_list[:n_val]
        train_samples = samples_list[n_val:]

        for split, split_samples in [("train", train_samples), ("val", val_samples)]:
            split_dir = OUTPUT_DIR / split / str(cat_id)
            split_dir.mkdir(exist_ok=True)

            for i, s in enumerate(split_samples):
                src = Path(s["crop_path"])
                if not src.exists():
                    continue
                dst = split_dir / f"{cat_id}_{i:05d}.jpg"
                if not dst.exists():
                    try:
                        dst.symlink_to(src.resolve())
                    except (OSError, FileExistsError):
                        continue

        total_train += len(train_samples)
        total_val += len(val_samples)
        cat_stats[cat_id] = {"train": len(train_samples), "val": len(val_samples)}

    vals_train = sorted([v["train"] for v in cat_stats.values()])
    vals_val = sorted([v["val"] for v in cat_stats.values()])

    print(f"\nCLASSIFIER DATASET COMPLETE:")
    print(f"  Train: {total_train} crops across {len(cat_stats)} categories")
    print(f"  Val: {total_val} crops")
    print(f"  Train per cat: min={vals_train[0]}, median={vals_train[len(vals_train)//2]}, max={vals_train[-1]}")
    print(f"  Val per cat: min={vals_val[0]}, median={vals_val[len(vals_val)//2]}, max={vals_val[-1]}")
    print(f"  Path: {OUTPUT_DIR}")

    # Save class mapping
    mapping = {str(cat_id): cat_names.get(cat_id, f"unknown_{cat_id}") for cat_id in sorted(cat_stats.keys())}
    with open(OUTPUT_DIR / "class_mapping.json", "w") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
    print(f"  Class mapping: {OUTPUT_DIR / 'class_mapping.json'}")


if __name__ == "__main__":
    main()

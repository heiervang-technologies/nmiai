"""
Build hard-negative pair dataset for classifier training.

Near-duplicate categories (same product, different size/variant) are the main
source of classification errors. This script:
1. Identifies confusable category pairs
2. Extracts side-by-side crop pairs for contrastive learning
3. Outputs a JSON manifest for triplet/contrastive loss training
"""
import json
import difflib
import random
from collections import defaultdict
from pathlib import Path

from PIL import Image

random.seed(42)

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
COCO_IMGS = DATA_DIR / "coco_dataset" / "train" / "images"
CLASSIFIER_CROPS = DATA_DIR / "classifier_crops"
OUTPUT_DIR = DATA_DIR / "hard_negative_pairs"


def find_confusable_pairs(cat_names, threshold=0.7):
    """Find category pairs with similar names."""
    cats = list(cat_names.items())
    pairs = []
    for i in range(len(cats)):
        for j in range(i + 1, len(cats)):
            id_a, name_a = cats[i]
            id_b, name_b = cats[j]
            ratio = difflib.SequenceMatcher(None, name_a.lower(), name_b.lower()).ratio()
            if ratio > threshold:
                pairs.append({
                    "cat_a": id_a,
                    "cat_b": id_b,
                    "name_a": name_a,
                    "name_b": name_b,
                    "similarity": round(ratio, 3),
                })
    return sorted(pairs, key=lambda x: -x["similarity"])


def main():
    print("Building hard-negative pair dataset...")

    with open(COCO_ANN) as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    # Find confusable pairs
    pairs = find_confusable_pairs(cat_names, threshold=0.7)
    print(f"Confusable pairs (>0.7 similarity): {len(pairs)}")

    # Group by difficulty tier
    critical = [p for p in pairs if p["similarity"] > 0.9]
    hard = [p for p in pairs if 0.8 < p["similarity"] <= 0.9]
    medium = [p for p in pairs if 0.7 < p["similarity"] <= 0.8]

    print(f"  Critical (>0.9): {len(critical)} pairs")
    print(f"  Hard (0.8-0.9): {len(hard)} pairs")
    print(f"  Medium (0.7-0.8): {len(medium)} pairs")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save pair manifest
    manifest = {
        "pairs": pairs,
        "stats": {
            "total_pairs": len(pairs),
            "critical": len(critical),
            "hard": len(hard),
            "medium": len(medium),
        },
        "categories_involved": sorted(set(
            [p["cat_a"] for p in pairs] + [p["cat_b"] for p in pairs]
        )),
    }

    with open(OUTPUT_DIR / "pair_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Build triplet dataset: (anchor, positive, negative) for each confusable pair
    # anchor/positive = same category, negative = confusable category
    triplets = []

    for pair in pairs:
        cat_a = pair["cat_a"]
        cat_b = pair["cat_b"]

        # Get crops for both categories
        crops_a = sorted((CLASSIFIER_CROPS / str(cat_a)).glob("*.jpg")) if (CLASSIFIER_CROPS / str(cat_a)).exists() else []
        crops_b = sorted((CLASSIFIER_CROPS / str(cat_b)).glob("*.jpg")) if (CLASSIFIER_CROPS / str(cat_b)).exists() else []

        if len(crops_a) < 2 or len(crops_b) < 2:
            continue

        # Create triplets: (anchor from A, positive from A, negative from B)
        n_triplets = min(5, len(crops_a) - 1, len(crops_b))
        for _ in range(n_triplets):
            anchor, positive = random.sample(crops_a, 2)
            negative = random.choice(crops_b)
            triplets.append({
                "anchor": str(anchor),
                "positive": str(positive),
                "negative": str(negative),
                "anchor_cat": cat_a,
                "negative_cat": cat_b,
                "similarity": pair["similarity"],
            })

        # Also reverse: (anchor from B, positive from B, negative from A)
        n_triplets = min(5, len(crops_b) - 1, len(crops_a))
        for _ in range(n_triplets):
            anchor, positive = random.sample(crops_b, 2)
            negative = random.choice(crops_a)
            triplets.append({
                "anchor": str(anchor),
                "positive": str(positive),
                "negative": str(negative),
                "anchor_cat": cat_b,
                "negative_cat": cat_a,
                "similarity": pair["similarity"],
            })

    # Save triplet dataset
    with open(OUTPUT_DIR / "triplets.json", "w") as f:
        json.dump(triplets, f, indent=2)

    print(f"\nGenerated {len(triplets)} triplets for contrastive learning")
    print(f"Output: {OUTPUT_DIR}")
    print(f"  pair_manifest.json: {len(pairs)} confusable pairs")
    print(f"  triplets.json: {len(triplets)} (anchor, positive, negative) triplets")

    # Also create a simple CSV for quick reference
    with open(OUTPUT_DIR / "confusable_pairs.csv", "w") as f:
        f.write("cat_a,cat_b,name_a,name_b,similarity\n")
        for p in pairs:
            f.write(f'{p["cat_a"]},{p["cat_b"]},"{p["name_a"]}","{p["name_b"]}",{p["similarity"]}\n')

    # Summary of most affected categories
    cat_confusion_count = defaultdict(int)
    for p in pairs:
        cat_confusion_count[p["cat_a"]] += 1
        cat_confusion_count[p["cat_b"]] += 1

    print(f"\nMost confusable categories (involved in most pairs):")
    for cat_id, count in sorted(cat_confusion_count.items(), key=lambda x: -x[1])[:20]:
        print(f"  cat {cat_id} ({cat_names[cat_id]}): {count} confusable pairs")


if __name__ == "__main__":
    main()

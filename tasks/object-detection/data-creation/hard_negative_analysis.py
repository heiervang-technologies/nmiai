"""
Hard negative mining via data analysis (CPU only - no GPU inference needed).

Identifies likely failure categories by analyzing:
1. Categories with fewest training samples (model saw less)
2. Categories with smallest average bbox (hard to detect small products)
3. Categories with highest visual similarity to others (confusion risk)
4. Categories that appear in fewest unique images (less diverse contexts)
5. Categories with most overlap with other products (occlusion)
"""
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"
CLEAN_TRAIN = DATA_DIR / "yolo_clean_with_val" / "train" / "labels"


def analyze_original_coco():
    """Analyze the original COCO annotations for structural patterns."""
    with open(COCO_ANN) as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}

    # Per-category stats from original data
    cat_stats = defaultdict(lambda: {
        "count": 0,
        "unique_images": set(),
        "areas": [],
        "widths": [],
        "heights": [],
        "aspect_ratios": [],
        "overlaps_with": Counter(),  # other categories in same image
    })

    # Group annotations by image for overlap analysis
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    for ann in coco["annotations"]:
        cid = ann["category_id"]
        stats = cat_stats[cid]
        stats["count"] += 1
        stats["unique_images"].add(ann["image_id"])

        x, y, w, h = ann["bbox"]
        area = w * h
        stats["areas"].append(area)
        stats["widths"].append(w)
        stats["heights"].append(h)
        if h > 0:
            stats["aspect_ratios"].append(w / h)

    # Compute overlaps (categories that co-occur in same image)
    for img_id, anns in img_anns.items():
        cats_in_img = set(a["category_id"] for a in anns)
        for cid in cats_in_img:
            for other in cats_in_img:
                if other != cid:
                    cat_stats[cid]["overlaps_with"][other] += 1

    return cat_stats, cat_names, img_map


def analyze_train_distribution():
    """Analyze the clean training set distribution."""
    counts = Counter()
    unique_images_per_cat = defaultdict(set)

    for f in CLEAN_TRAIN.glob("*.txt"):
        for line in open(f):
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                counts[cid] += 1
                unique_images_per_cat[cid].add(f.stem)

    return counts, unique_images_per_cat


def main():
    print("=" * 80)
    print("HARD NEGATIVE MINING - FAILURE MODE ANALYSIS")
    print("=" * 80)

    cat_stats, cat_names, _ = analyze_original_coco()
    train_counts, train_images = analyze_train_distribution()

    # Risk score per category (higher = more likely to fail)
    risk_scores = {}

    for cid in range(356):
        name = cat_names.get(cid, f"unknown_{cid}")
        stats = cat_stats.get(cid, {"count": 0, "areas": [1], "unique_images": set()})

        orig_count = stats["count"]
        train_count = train_counts.get(cid, 0)
        n_images = len(train_images.get(cid, set()))
        avg_area = np.mean(stats["areas"]) if stats["areas"] else 0
        median_area = np.median(stats["areas"]) if stats["areas"] else 0

        # Risk factors
        risk = 0

        # Factor 1: Low training count
        if train_count < 30:
            risk += 3
        elif train_count < 50:
            risk += 2
        elif train_count < 100:
            risk += 1

        # Factor 2: Small objects (area < 5000 px²)
        if median_area < 3000:
            risk += 3
        elif median_area < 10000:
            risk += 2
        elif median_area < 20000:
            risk += 1

        # Factor 3: Low image diversity
        if n_images < 5:
            risk += 2
        elif n_images < 15:
            risk += 1

        # Factor 4: Only appears in original data (no augmentation benefit)
        if orig_count == train_count and orig_count < 20:
            risk += 2

        risk_scores[cid] = {
            "name": name,
            "risk": risk,
            "train_count": train_count,
            "orig_count": orig_count,
            "unique_images": n_images,
            "median_area": int(median_area),
            "avg_area": int(avg_area),
        }

    # Sort by risk (highest first)
    sorted_risks = sorted(risk_scores.items(), key=lambda x: (-x[1]["risk"], x[1]["train_count"]))

    print(f"\n{'='*80}")
    print("TOP 30 HIGHEST-RISK CATEGORIES (most likely to fail)")
    print(f"{'='*80}")
    print(f"{'Cat':>4} {'Risk':>5} {'Train':>6} {'Orig':>5} {'Imgs':>5} {'MedArea':>8} | Name")
    print("-" * 80)
    for cid, info in sorted_risks[:30]:
        print(f"{cid:4d} {info['risk']:5d} {info['train_count']:6d} {info['orig_count']:5d} {info['unique_images']:5d} {info['median_area']:8d} | {info['name'][:45]}")

    # Group by risk level
    high_risk = [cid for cid, info in sorted_risks if info["risk"] >= 5]
    medium_risk = [cid for cid, info in sorted_risks if 3 <= info["risk"] < 5]
    low_risk = [cid for cid, info in sorted_risks if info["risk"] < 3]

    print(f"\nRisk distribution:")
    print(f"  HIGH risk (>=5):   {len(high_risk)} categories")
    print(f"  MEDIUM risk (3-4): {len(medium_risk)} categories")
    print(f"  LOW risk (<3):     {len(low_risk)} categories")

    # Identify small object categories
    print(f"\n{'='*80}")
    print("SMALL OBJECT CATEGORIES (median area < 10000 px²)")
    print(f"{'='*80}")
    small_cats = [(cid, info) for cid, info in sorted_risks if info["median_area"] > 0 and info["median_area"] < 10000]
    small_cats.sort(key=lambda x: x[1]["median_area"])
    for cid, info in small_cats[:20]:
        print(f"  cat {cid:3d} area={info['median_area']:6d} train={info['train_count']:4d} | {info['name'][:50]}")

    # Categories with low image diversity
    print(f"\n{'='*80}")
    print("LOW DIVERSITY CATEGORIES (< 10 unique training images)")
    print(f"{'='*80}")
    low_div = [(cid, info) for cid, info in sorted_risks if info["unique_images"] < 10 and info["train_count"] > 0]
    low_div.sort(key=lambda x: x[1]["unique_images"])
    for cid, info in low_div[:20]:
        print(f"  cat {cid:3d} imgs={info['unique_images']:3d} train={info['train_count']:4d} | {info['name'][:50]}")

    # Save analysis
    out = {
        "high_risk_categories": [{"id": cid, **risk_scores[cid]} for cid in high_risk],
        "medium_risk_categories": [{"id": cid, **risk_scores[cid]} for cid in medium_risk],
        "small_object_categories": [cid for cid, _ in small_cats[:20]],
        "low_diversity_categories": [cid for cid, _ in low_div[:20]],
    }
    out_path = Path(__file__).parent / "hard_negative_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

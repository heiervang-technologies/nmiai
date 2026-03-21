"""
Deep gap analysis across all datasets:
1. Per-category annotation counts across all sources
2. Which categories are still underrepresented
3. Visual diversity analysis (how many unique images per category)
4. Cross-dataset coverage matrix
5. Recommendations for targeted data collection
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"


def load_yolo_labels(labels_dir: Path) -> Counter:
    """Count annotations per category from YOLO label files."""
    counts = Counter()
    for f in labels_dir.glob("*.txt"):
        for line in open(f):
            parts = line.strip().split()
            if len(parts) >= 1:
                counts[int(parts[0])] += 1
    return counts


def main():
    # Load category names
    with open(COCO_ANN) as f:
        coco = json.load(f)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    orig_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    # Count from each dataset
    datasets = {}

    # Original COCO
    datasets["original"] = orig_counts

    # V1 train
    v1_dir = DATA_DIR / "yolo_dataset" / "train" / "labels"
    if v1_dir.exists():
        datasets["v1_train"] = load_yolo_labels(v1_dir)

    # V3
    v3_dir = DATA_DIR / "yolo_augmented_v3" / "train" / "labels"
    if v3_dir.exists():
        datasets["v3_train"] = load_yolo_labels(v3_dir)

    # V4
    v4_dir = DATA_DIR / "yolo_augmented_v4" / "train" / "labels"
    if v4_dir.exists():
        datasets["v4_train"] = load_yolo_labels(v4_dir)

    # Pseudo-labeled store data only
    v4_store = Counter()
    v4_vframe = Counter()
    if v4_dir.exists():
        for f in v4_dir.glob("store_*.txt"):
            for line in open(f):
                parts = line.strip().split()
                if parts:
                    v4_store[int(parts[0])] += 1
        for f in v4_dir.glob("vframe_*.txt"):
            for line in open(f):
                parts = line.strip().split()
                if parts:
                    v4_vframe[int(parts[0])] += 1
    datasets["pseudo_store"] = v4_store
    datasets["pseudo_vframe"] = v4_vframe

    # Print comprehensive analysis
    print("=" * 100)
    print("COMPREHENSIVE DATA GAP ANALYSIS")
    print("=" * 100)

    print(f"\n{'Dataset':<20} {'Images':<10} {'Annotations':<15} {'Categories':<12} {'Med/Cat':<10}")
    print("-" * 70)
    for name, counts in datasets.items():
        total = sum(counts.values())
        n_cats = len(counts)
        vals = sorted(counts.values()) if counts else [0]
        median = vals[len(vals)//2] if vals else 0
        print(f"{name:<20} {'?':<10} {total:<15} {n_cats:<12} {median:<10}")

    # Identify weak categories
    v4_counts = datasets.get("v4_train", Counter())
    print(f"\n{'='*100}")
    print("CATEGORY WEAKNESS ANALYSIS (V4 dataset)")
    print(f"{'='*100}")

    # Group categories by annotation count
    buckets = [
        (0, 5, "CRITICAL"),
        (6, 15, "WEAK"),
        (16, 30, "BELOW_AVG"),
        (31, 100, "ADEQUATE"),
        (101, 300, "GOOD"),
        (301, 1000, "STRONG"),
        (1001, 99999, "ABUNDANT"),
    ]

    for lo, hi, label in buckets:
        cats_in_bucket = [(cid, v4_counts.get(cid, 0)) for cid in range(356) if lo <= v4_counts.get(cid, 0) <= hi]
        if cats_in_bucket:
            print(f"\n{label} ({lo}-{hi} annotations): {len(cats_in_bucket)} categories")
            if label in ("CRITICAL", "WEAK"):
                for cid, cnt in sorted(cats_in_bucket, key=lambda x: x[1]):
                    name = cat_names.get(cid, "?")
                    orig = orig_counts.get(cid, 0)
                    pseudo_s = v4_store.get(cid, 0)
                    pseudo_v = v4_vframe.get(cid, 0)
                    print(f"    cat {cid:3d} ({cnt:4d} v4, {orig:3d} orig, {pseudo_s:2d} store, {pseudo_v:2d} vframe): {name}")

    # Category supercategory analysis (by product type)
    print(f"\n{'='*100}")
    print("PRODUCT TYPE ANALYSIS")
    print(f"{'='*100}")

    # Infer product types from names
    type_keywords = {
        "kaffe/te": ["kaffe", "te ", "tea", "espresso", "latte", "cappuccino", "nescafe", "evergood", "friele", "twinings", "lipton"],
        "knekkebrød": ["knekkebrød", "knekke", "wasa", "leksands", "sigdal"],
        "egg": ["egg", "gårdsegg", "frokostegg"],
        "chips/snacks": ["chips", "potet", "snacks", "mais", "riskaker", "friggs", "torres"],
        "frokostblanding": ["müsli", "granola", "havregryn", "cornflakes", "frokostblanding", "cheerios", "smacks", "lion"],
        "brød/flatbrød": ["flatbrød", "brød", "grissini", "bruschetta"],
        "kjøtt": ["storfe", "entrecote", "short ribs", "kjøtt", "bog"],
        "meieri": ["smør", "ost", "jarlsberg", "smøreost", "melange"],
        "godteri/sjokolade": ["sjokolade", "dadler", "drops", "pastiller", "sweet"],
        "hygiene": ["ob ", "procomfort"],
        "krydder": ["kryddermiks", "hindu"],
        "pannekaker": ["pannekaker"],
        "juice/drikke": ["ice tea", "icetea", "pulver"],
    }

    type_stats = {}
    for type_name, keywords in type_keywords.items():
        matching_cats = []
        for cid, name in cat_names.items():
            name_lower = name.lower()
            if any(kw in name_lower for kw in keywords):
                matching_cats.append(cid)
        if matching_cats:
            v4_total = sum(v4_counts.get(cid, 0) for cid in matching_cats)
            orig_total = sum(orig_counts.get(cid, 0) for cid in matching_cats)
            v4_per_cat = v4_total / len(matching_cats) if matching_cats else 0
            type_stats[type_name] = {
                "cats": len(matching_cats),
                "v4_total": v4_total,
                "orig_total": orig_total,
                "v4_per_cat": v4_per_cat,
            }
            print(f"  {type_name:<20}: {len(matching_cats):3d} cats, {v4_per_cat:6.0f} ann/cat (v4), {orig_total:6d} orig")

    # Cross-source coverage
    print(f"\n{'='*100}")
    print("CROSS-SOURCE CONTRIBUTION")
    print(f"{'='*100}")

    # How many categories got new data from PL vs augmentation vs pseudo-labels
    pl_contribution = Counter()
    aug_contribution = Counter()
    if v4_dir.exists():
        for f in v4_dir.glob("pl_*.txt"):
            for line in open(f):
                parts = line.strip().split()
                if parts:
                    pl_contribution[int(parts[0])] += 1
        for f in v4_dir.glob("*_aug*.txt"):
            for line in open(f):
                parts = line.strip().split()
                if parts:
                    aug_contribution[int(parts[0])] += 1

    print(f"  PL data: {len(pl_contribution)} categories, {sum(pl_contribution.values())} annotations")
    print(f"  Augmented: {len(aug_contribution)} categories, {sum(aug_contribution.values())} annotations")
    print(f"  Pseudo store: {len(v4_store)} categories, {sum(v4_store.values())} annotations")
    print(f"  Pseudo vframe: {len(v4_vframe)} categories, {sum(v4_vframe.values())} annotations")

    # Recommendations
    print(f"\n{'='*100}")
    print("RECOMMENDATIONS")
    print(f"{'='*100}")

    critical = [cid for cid in range(356) if v4_counts.get(cid, 0) <= 5]
    weak = [cid for cid in range(356) if 6 <= v4_counts.get(cid, 0) <= 15]

    print(f"\n1. CRITICAL categories ({len(critical)} with <=5 annotations):")
    if critical:
        print(f"   These need targeted data collection or heavy augmentation")
        for cid in critical[:10]:
            print(f"     {cid}: {cat_names.get(cid, '?')} ({v4_counts.get(cid, 0)} ann)")

    print(f"\n2. WEAK categories ({len(weak)} with 6-15 annotations):")
    if weak:
        print(f"   These would benefit from more augmentation or external data")

    print(f"\n3. Pseudo-labeling coverage:")
    pseudo_total = len(v4_store) + len(v4_vframe)
    print(f"   Store photos covered {len(v4_store)} categories")
    print(f"   Video frames covered {len(v4_vframe)} categories")
    print(f"   Consider lowering conf threshold from 0.5 to 0.3 for more coverage")

    print(f"\n4. Product type gaps:")
    for type_name, stats in sorted(type_stats.items(), key=lambda x: x[1]["v4_per_cat"]):
        if stats["v4_per_cat"] < 50:
            print(f"   {type_name}: only {stats['v4_per_cat']:.0f} ann/cat - needs more data")

    # Save analysis
    analysis = {
        "v4_per_category": {str(cid): v4_counts.get(cid, 0) for cid in range(356)},
        "original_per_category": {str(cid): orig_counts.get(cid, 0) for cid in range(356)},
        "pseudo_store_per_category": {str(cid): v4_store.get(cid, 0) for cid in range(356) if v4_store.get(cid, 0) > 0},
        "pseudo_vframe_per_category": {str(cid): v4_vframe.get(cid, 0) for cid in range(356) if v4_vframe.get(cid, 0) > 0},
        "critical_categories": critical,
        "weak_categories": weak,
        "category_names": {str(k): v for k, v in cat_names.items()},
    }
    out_path = Path(__file__).parent / "gap_analysis.json"
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    print(f"\nFull analysis saved to {out_path}")


if __name__ == "__main__":
    main()

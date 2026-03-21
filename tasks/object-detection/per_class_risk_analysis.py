#!/usr/bin/env python3
"""
Per-Class Risk Analysis - No GPU needed.

Cross-references:
1. Clean split val ground truth (what the model is tested on)
2. Training set class distribution (what the model learned)
3. Difficulty scores from V2
4. Produces a final "risk score" per class = likelihood of mAP loss

Also computes:
- Train/val distribution mismatch (classes overfit or undertrained)
- Classes present in val but rare in train (guaranteed low AP)
- Confusion risk clusters (groups of similar products)
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).parent
COCO_PATH = BASE / "data-creation/data/coco_dataset/train/annotations.json"
CLEAN_VAL_LABELS = BASE / "data-creation/data/clean_split/val/labels"
CLEAN_TRAIN_LABELS = BASE / "data-creation/data/clean_split/train/labels"
SILVER_DATASET = BASE / "data-creation/data/silver_augmented_dataset"
DATA_YAML = BASE / "yolo-approach/dataset/data.yaml"
V2_REPORT = BASE / "deep_insights_v2_report.json"


def load_class_names():
    names = {}
    with open(DATA_YAML) as f:
        in_names = False
        for line in f:
            line = line.rstrip()
            if line.startswith("names:"):
                in_names = True
                continue
            if in_names and ":" in line:
                parts = line.strip().split(":", 1)
                try:
                    idx = int(parts[0].strip())
                    name = parts[1].strip().strip("'\"")
                    names[idx] = name
                except ValueError:
                    pass
    return names


def count_yolo_labels(label_dir):
    """Count annotations per class from YOLO format label files."""
    counts = Counter()
    n_files = 0
    if not label_dir.exists():
        return counts, 0

    for f in label_dir.iterdir():
        if f.suffix != '.txt':
            continue
        n_files += 1
        with open(f) as fh:
            for line in fh:
                parts = line.strip().split()
                if parts:
                    try:
                        cat_id = int(parts[0])
                        counts[cat_id] += 1
                    except ValueError:
                        pass
    return counts, n_files


def main():
    class_names = load_class_names()

    # Load COCO for original counts
    with open(COCO_PATH) as f:
        coco = json.load(f)
    original_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    # Count val labels
    val_counts, val_images = count_yolo_labels(CLEAN_VAL_LABELS)
    print(f"Clean val: {val_images} images, {sum(val_counts.values())} annotations")

    # Count train labels
    train_counts, train_images = count_yolo_labels(CLEAN_TRAIN_LABELS)
    print(f"Clean train: {train_images} images, {sum(train_counts.values())} annotations")

    # Check silver augmented dataset
    silver_train_labels = SILVER_DATASET / "train" / "labels"
    if silver_train_labels.exists():
        silver_counts, silver_images = count_yolo_labels(silver_train_labels)
        print(f"Silver augmented: {silver_images} images, {sum(silver_counts.values())} annotations")
    else:
        silver_counts = Counter()
        silver_images = 0
        # Try alternate path
        for alt in [SILVER_DATASET / "labels" / "train", SILVER_DATASET / "labels"]:
            if alt.exists():
                silver_counts, silver_images = count_yolo_labels(alt)
                print(f"Silver augmented ({alt}): {silver_images} images, {sum(silver_counts.values())} annotations")
                break
        if not silver_counts:
            print("Silver augmented dataset not found or empty")

    # Load V2 difficulty scores
    difficulty_scores = {}
    if V2_REPORT.exists():
        with open(V2_REPORT) as f:
            v2 = json.load(f)
        for item in v2.get("difficulty_ranking", []):
            difficulty_scores[item["cat_id"]] = item["scores"]["total"]

    # ============================================================
    # RISK ANALYSIS
    # ============================================================
    print(f"\n{'='*120}")
    print("PER-CLASS RISK ANALYSIS")
    print(f"{'='*120}")

    risks = []
    for cat_id in range(356):
        name = class_names.get(cat_id, "?")
        orig = original_counts.get(cat_id, 0)
        train = train_counts.get(cat_id, 0)
        val = val_counts.get(cat_id, 0)
        silver = silver_counts.get(cat_id, 0)
        diff = difficulty_scores.get(cat_id, 50)

        # Risk factors:
        # 1. Val presence with low train count (will definitely have low AP)
        val_risk = 0
        if val > 0:
            if train == 0:
                val_risk = 40  # In val but NOT in train = guaranteed 0 AP
            elif train < 5:
                val_risk = 30
            elif train < 15:
                val_risk = 20
            elif train < 30:
                val_risk = 10
            else:
                val_risk = max(0, 5 - train / 20)
        else:
            val_risk = 0  # Not in val = doesn't affect score directly

        # 2. Train/val ratio imbalance
        if val > 0 and train > 0:
            ratio = train / val
            if ratio < 1:
                imbalance = 15  # More val than train = bad
            elif ratio < 3:
                imbalance = 8
            elif ratio < 5:
                imbalance = 3
            else:
                imbalance = 0
        else:
            imbalance = 0

        # 3. Difficulty score contribution
        diff_risk = diff * 0.3  # Scale 0-30

        # 4. Silver data rescue factor (negative risk = silver helps)
        silver_rescue = 0
        if silver > 0 and train < 30:
            silver_rescue = -min(15, silver / 10)  # Silver data reduces risk

        total_risk = val_risk + imbalance + diff_risk + silver_rescue
        total_risk = max(0, min(100, total_risk))

        risks.append({
            "cat_id": cat_id,
            "name": name,
            "original": orig,
            "train": train,
            "val": val,
            "silver": silver,
            "difficulty": diff,
            "val_risk": round(val_risk, 1),
            "imbalance": round(imbalance, 1),
            "diff_risk": round(diff_risk, 1),
            "silver_rescue": round(silver_rescue, 1),
            "total_risk": round(total_risk, 1),
        })

    # Sort by risk descending
    risks.sort(key=lambda x: -x["total_risk"])

    # Classes IN val set (these are the ones that matter for scoring)
    val_classes = [r for r in risks if r["val"] > 0]
    non_val_classes = [r for r in risks if r["val"] == 0]

    print(f"\nClasses in val set: {len(val_classes)}")
    print(f"Classes NOT in val set: {len(non_val_classes)}")
    print(f"(Only val classes affect the clean-split evaluation score)")

    print(f"\n--- HIGHEST RISK CLASSES (in val set, sorted by risk) ---")
    print(f"{'Rank':<5} {'CatID':<6} {'Risk':<7} {'ValRisk':<8} {'Imbal':<7} {'Diff':<7} {'Silver':<8} {'Train':<7} {'Val':<5} {'Silv':<6} {'Name'}")
    print("-" * 130)
    rank = 0
    for r in risks:
        if r["val"] == 0:
            continue
        rank += 1
        if rank > 80:
            break
        print(f"{rank:<5} {r['cat_id']:<6} {r['total_risk']:<7.1f} {r['val_risk']:<8.1f} {r['imbalance']:<7.1f} {r['diff_risk']:<7.1f} {r['silver_rescue']:<8.1f} {r['train']:<7} {r['val']:<5} {r['silver']:<6} {r['name'][:42]}")

    # ============================================================
    # CRITICAL: CLASSES IN VAL WITH ZERO OR VERY FEW TRAIN SAMPLES
    # ============================================================
    print(f"\n{'='*120}")
    print("CRITICAL: CLASSES IN VALIDATION SET WITH INSUFFICIENT TRAINING DATA")
    print(f"{'='*120}")

    zero_train_in_val = [r for r in risks if r["val"] > 0 and r["train"] == 0]
    few_train_in_val = [r for r in risks if r["val"] > 0 and 0 < r["train"] <= 5]

    if zero_train_in_val:
        print(f"\n!!! {len(zero_train_in_val)} classes are IN VAL but have ZERO train samples !!!")
        print("These will GUARANTEED have 0 AP, dragging down overall mAP:")
        for r in zero_train_in_val:
            print(f"  [{r['cat_id']}] val={r['val']} train=0 silver={r['silver']} - {r['name']}")
    else:
        print("\nNo classes in val with zero train samples. Good.")

    if few_train_in_val:
        print(f"\n{len(few_train_in_val)} classes in val with <=5 train samples (very likely low AP):")
        for r in few_train_in_val:
            print(f"  [{r['cat_id']}] val={r['val']} train={r['train']} silver={r['silver']} - {r['name']}")

    # ============================================================
    # CONFUSION RISK CLUSTERS
    # ============================================================
    print(f"\n{'='*120}")
    print("CONFUSION RISK CLUSTERS")
    print(f"{'='*120}")

    # Group val classes by product type
    clusters = defaultdict(list)
    for r in val_classes:
        name = r["name"].upper()
        if any(k in name for k in ["KNEKKE", "WASA", "SIGDAL", "LEKSAND", "RYVITA"]):
            clusters["CRISPBREAD"].append(r)
        elif any(k in name for k in ["EGG", "GÅRDSEGG"]):
            clusters["EGGS"].append(r)
        elif any(k in name for k in ["KAPSEL", "KAPSLER", "DOLCE GUSTO", "TASSIMO"]):
            clusters["COFFEE_CAPSULES"].append(r)
        elif any(k in name for k in ["FILTERMALT", "KOKMALT", "PRESSMALT"]):
            clusters["GROUND_COFFEE"].append(r)
        elif any(k in name for k in ["HELE BØNNER"]):
            clusters["COFFEE_BEANS"].append(r)
        elif any(k in name for k in ["TE ", "TEA ", "PUKKA", "TWININGS", "LIPTON"]) and "KAFFE" not in name:
            clusters["TEA"].append(r)
        elif any(k in name for k in ["MÜSLI", "MUSLI", "GRANOLA"]):
            clusters["MUESLI_GRANOLA"].append(r)
        elif any(k in name for k in ["MARGARIN", "SMØR", "FLORA", "MELANGE", "BREMYKT", "BRELETT", "HJERTEGO", "OLIVERO"]):
            clusters["BUTTER_MARGARINE"].append(r)
        elif any(k in name for k in ["MELLOMBAR"]):
            clusters["SNACK_BARS"].append(r)
        elif any(k in name for k in ["NESCAFE", "NESCAFÉ"]):
            clusters["NESCAFE"].append(r)
        elif any(k in name for k in ["SJOKOLADEDRIKK", "O'BOY", "KAKAO", "NESQUIK"]):
            clusters["HOT_CHOCOLATE"].append(r)
        elif "SANDWICH" in name:
            clusters["WASA_SANDWICH"].append(r)

    for cluster_name, items in sorted(clusters.items(), key=lambda x: -len(x[1])):
        avg_risk = sum(r["total_risk"] for r in items) / len(items)
        avg_train = sum(r["train"] for r in items) / len(items)
        print(f"\n{cluster_name} ({len(items)} classes, avg_risk={avg_risk:.1f}, avg_train={avg_train:.0f}):")
        for r in sorted(items, key=lambda x: -x["total_risk"])[:8]:
            marker = "!!!" if r["total_risk"] > 50 else "  " if r["total_risk"] < 25 else " !"
            print(f"  {marker} [{r['cat_id']}] risk={r['total_risk']:.0f} train={r['train']} val={r['val']} - {r['name'][:45]}")
        if len(items) > 8:
            print(f"  ... and {len(items)-8} more")

    # ============================================================
    # SUMMARY STATS
    # ============================================================
    print(f"\n{'='*120}")
    print("SUMMARY")
    print(f"{'='*120}")

    if val_classes:
        avg_risk = sum(r["total_risk"] for r in val_classes) / len(val_classes)
        high_risk = sum(1 for r in val_classes if r["total_risk"] > 50)
        medium_risk = sum(1 for r in val_classes if 25 < r["total_risk"] <= 50)
        low_risk = sum(1 for r in val_classes if r["total_risk"] <= 25)

        print(f"\nVal set classes: {len(val_classes)}")
        print(f"Average risk score: {avg_risk:.1f}")
        print(f"High risk (>50): {high_risk} classes")
        print(f"Medium risk (25-50): {medium_risk} classes")
        print(f"Low risk (<25): {low_risk} classes")

        # Estimate mAP impact
        print(f"\nEstimated mAP breakdown:")
        print(f"  If we fix all high-risk classes: ~{high_risk * 0.3:.0f}% mAP improvement potential")
        print(f"  If we fix medium-risk classes: ~{medium_risk * 0.15:.0f}% additional")
        print(f"  Classes already well-trained: {low_risk}")

    # Save report
    report_path = BASE / "per_class_risk_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "summary": {
                "val_classes": len(val_classes),
                "high_risk": high_risk,
                "medium_risk": medium_risk,
                "low_risk": low_risk,
                "avg_risk": round(avg_risk, 1),
                "zero_train_in_val": len(zero_train_in_val),
            },
            "risks": risks,
            "clusters": {k: [{"cat_id": r["cat_id"], "name": r["name"], "risk": r["total_risk"], "train": r["train"], "val": r["val"]} for r in v] for k, v in clusters.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

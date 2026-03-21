#!/usr/bin/env python3
"""
Deep Insights Framework for NM i AI Object Detection Task.

Analyzes the full COCO dataset to produce actionable intelligence:
- Class distribution & weakness analysis
- Bounding box size/aspect ratio patterns per class
- Image-level difficulty metrics (density, resolution, class diversity)
- Co-occurrence analysis (which products appear together)
- Confusable class pairs (visually similar products)
- Shelf section analysis (where on shelves are products placed)
- Actionable photo-taking recommendations

Output: deep_insights_report.json + printed summary
"""

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
import math

# Paths
COCO_PATH = Path(__file__).parent / "data-creation/data/coco_dataset/train/annotations.json"
DATA_YAML = Path(__file__).parent / "yolo-approach/dataset/data.yaml"
REPORT_OUT = Path(__file__).parent / "deep_insights_report.json"

def load_coco():
    with open(COCO_PATH) as f:
        return json.load(f)

def load_class_names():
    """Parse class names from data.yaml"""
    names = {}
    with open(DATA_YAML) as f:
        in_names = False
        for line in f:
            line = line.rstrip()
            if line.startswith("names:"):
                in_names = True
                continue
            if in_names:
                if line.strip().startswith("path:") or line.strip().startswith("train:") or line.strip().startswith("val:") or line.strip().startswith("nc:"):
                    continue
                if ":" in line:
                    parts = line.strip().split(":", 1)
                    try:
                        idx = int(parts[0].strip())
                        name = parts[1].strip().strip("'\"")
                        names[idx] = name
                    except ValueError:
                        pass
    return names

def analyze():
    coco = load_coco()
    class_names = load_class_names()

    images = {img["id"]: img for img in coco["images"]}
    categories = {cat["id"]: cat for cat in coco["categories"]}
    annotations = coco["annotations"]

    print(f"=== DATASET OVERVIEW ===")
    print(f"Images: {len(images)}")
    print(f"Annotations: {len(annotations)}")
    print(f"Categories: {len(categories)}")
    print()

    # ============================================================
    # 1. CLASS DISTRIBUTION ANALYSIS
    # ============================================================
    class_counts = Counter()
    class_bboxes = defaultdict(list)  # cat_id -> list of (w, h, area, img_w, img_h)
    class_images = defaultdict(set)   # cat_id -> set of image_ids
    image_annotations = defaultdict(list)  # img_id -> list of annotations

    for ann in annotations:
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        bbox = ann["bbox"]  # [x, y, w, h]
        img = images[img_id]

        class_counts[cat_id] += 1
        class_bboxes[cat_id].append({
            "w": bbox[2], "h": bbox[3],
            "area": bbox[2] * bbox[3],
            "img_w": img["width"], "img_h": img["height"],
            "rel_w": bbox[2] / img["width"],
            "rel_h": bbox[3] / img["height"],
            "rel_area": (bbox[2] * bbox[3]) / (img["width"] * img["height"]),
            "cx": (bbox[0] + bbox[2]/2) / img["width"],  # center x normalized
            "cy": (bbox[1] + bbox[3]/2) / img["height"],  # center y normalized
        })
        class_images[cat_id].add(img_id)
        image_annotations[img_id].append(ann)

    # Sort by count ascending (weakest first)
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1])

    print("=== WEAKEST CLASSES (bottom 50) ===")
    print(f"{'Rank':<5} {'CatID':<6} {'Count':<6} {'Images':<7} {'Name'}")
    print("-" * 90)
    weak_classes = []
    for rank, (cat_id, count) in enumerate(sorted_classes[:50], 1):
        n_images = len(class_images[cat_id])
        name = class_names.get(cat_id, categories.get(cat_id, {}).get("name", "?"))
        print(f"{rank:<5} {cat_id:<6} {count:<6} {n_images:<7} {name}")
        weak_classes.append({
            "rank": rank, "cat_id": cat_id, "count": count,
            "n_images": n_images, "name": name
        })

    print()
    print("=== STRONGEST CLASSES (top 20) ===")
    print(f"{'Rank':<5} {'CatID':<6} {'Count':<6} {'Images':<7} {'Name'}")
    print("-" * 90)
    for rank, (cat_id, count) in enumerate(sorted(class_counts.items(), key=lambda x: -x[1])[:20], 1):
        n_images = len(class_images[cat_id])
        name = class_names.get(cat_id, categories.get(cat_id, {}).get("name", "?"))
        print(f"{rank:<5} {cat_id:<6} {count:<6} {n_images:<7} {name}")

    # ============================================================
    # 2. BBOX SIZE ANALYSIS PER CLASS
    # ============================================================
    print()
    print("=== BBOX SIZE ANALYSIS (relative to image) ===")
    print(f"{'CatID':<6} {'Count':<6} {'MedRelArea':<12} {'MedRelW':<10} {'MedRelH':<10} {'AspectRatio':<12} {'Name'}")
    print("-" * 100)

    class_size_stats = {}
    for cat_id in sorted(class_counts.keys()):
        bboxes = class_bboxes[cat_id]
        rel_areas = sorted([b["rel_area"] for b in bboxes])
        rel_ws = sorted([b["rel_w"] for b in bboxes])
        rel_hs = sorted([b["rel_h"] for b in bboxes])
        aspects = sorted([b["w"] / max(b["h"], 1) for b in bboxes])

        n = len(bboxes)
        med_area = rel_areas[n//2]
        med_w = rel_ws[n//2]
        med_h = rel_hs[n//2]
        med_aspect = aspects[n//2]

        class_size_stats[cat_id] = {
            "median_rel_area": round(med_area, 6),
            "median_rel_w": round(med_w, 4),
            "median_rel_h": round(med_h, 4),
            "median_aspect_ratio": round(med_aspect, 3),
            "min_rel_area": round(rel_areas[0], 6),
            "max_rel_area": round(rel_areas[-1], 6),
        }

    # Show classes with smallest bboxes (hardest to detect)
    smallest = sorted(class_size_stats.items(), key=lambda x: x[1]["median_rel_area"])
    print("\n--- Smallest products (hardest to detect) ---")
    for cat_id, stats in smallest[:20]:
        name = class_names.get(cat_id, "?")
        count = class_counts[cat_id]
        print(f"{cat_id:<6} {count:<6} {stats['median_rel_area']:<12.6f} {stats['median_rel_w']:<10.4f} {stats['median_rel_h']:<10.4f} {stats['median_aspect_ratio']:<12.3f} {name}")

    print("\n--- Largest products ---")
    for cat_id, stats in smallest[-10:]:
        name = class_names.get(cat_id, "?")
        count = class_counts[cat_id]
        print(f"{cat_id:<6} {count:<6} {stats['median_rel_area']:<12.6f} {stats['median_rel_w']:<10.4f} {stats['median_rel_h']:<10.4f} {stats['median_aspect_ratio']:<12.3f} {name}")

    # ============================================================
    # 3. IMAGE-LEVEL DIFFICULTY ANALYSIS
    # ============================================================
    print()
    print("=== IMAGE DIFFICULTY ANALYSIS ===")

    image_stats = []
    for img_id, img in images.items():
        anns = image_annotations.get(img_id, [])
        n_anns = len(anns)
        n_cats = len(set(a["category_id"] for a in anns))

        # Density: annotations per megapixel
        megapixels = (img["width"] * img["height"]) / 1e6
        density = n_anns / max(megapixels, 0.1)

        # Average bbox area
        if anns:
            avg_area = sum(a["bbox"][2] * a["bbox"][3] for a in anns) / len(anns)
            avg_rel_area = avg_area / (img["width"] * img["height"])
        else:
            avg_area = 0
            avg_rel_area = 0

        image_stats.append({
            "id": img_id,
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
            "megapixels": round(megapixels, 2),
            "n_annotations": n_anns,
            "n_categories": n_cats,
            "density": round(density, 1),
            "avg_bbox_area": round(avg_area),
            "avg_rel_bbox_area": round(avg_rel_area, 6),
        })

    image_stats.sort(key=lambda x: -x["n_annotations"])

    print(f"\n--- Densest images (most annotations) ---")
    print(f"{'File':<20} {'W×H':<12} {'Anns':<6} {'Cats':<6} {'Density':<8} {'AvgRelArea'}")
    for s in image_stats[:15]:
        print(f"{s['file_name']:<20} {s['width']}×{s['height']:<5} {s['n_annotations']:<6} {s['n_categories']:<6} {s['density']:<8} {s['avg_rel_bbox_area']:.6f}")

    print(f"\n--- Sparsest images (fewest annotations) ---")
    for s in image_stats[-10:]:
        print(f"{s['file_name']:<20} {s['width']}×{s['height']:<5} {s['n_annotations']:<6} {s['n_categories']:<6} {s['density']:<8} {s['avg_rel_bbox_area']:.6f}")

    # Resolution distribution
    resolutions = [(img["width"], img["height"]) for img in images.values()]
    widths = [r[0] for r in resolutions]
    heights = [r[1] for r in resolutions]
    print(f"\n--- Resolution distribution ---")
    print(f"Width:  min={min(widths)}, max={max(widths)}, median={sorted(widths)[len(widths)//2]}")
    print(f"Height: min={min(heights)}, max={max(heights)}, median={sorted(heights)[len(heights)//2]}")

    # ============================================================
    # 4. CO-OCCURRENCE ANALYSIS
    # ============================================================
    print()
    print("=== CO-OCCURRENCE ANALYSIS (which products appear together) ===")

    # Build co-occurrence matrix (sparse)
    cooccurrence = Counter()
    for img_id, anns in image_annotations.items():
        cats_in_image = set(a["category_id"] for a in anns)
        cats_list = sorted(cats_in_image)
        for i in range(len(cats_list)):
            for j in range(i+1, len(cats_list)):
                cooccurrence[(cats_list[i], cats_list[j])] += 1

    # Show top co-occurring pairs
    print(f"\n--- Most frequent product pairs ---")
    for (c1, c2), count in cooccurrence.most_common(20):
        n1 = class_names.get(c1, "?")[:30]
        n2 = class_names.get(c2, "?")[:30]
        print(f"  [{c1}] {n1:<32} + [{c2}] {n2:<32} = {count} images")

    # ============================================================
    # 5. PRODUCT CATEGORY GROUPING (by product type)
    # ============================================================
    print()
    print("=== PRODUCT CATEGORY GROUPS ===")

    # Group by keywords
    groups = {
        "EGGS": [], "COFFEE_CAPSULES": [], "COFFEE_GROUND": [], "COFFEE_BEANS": [],
        "COFFEE_INSTANT": [], "TEA": [], "KNEKKEBRØD": [], "FLATBRØD": [],
        "MÜSLI_GRANOLA": [], "FROKOSTBLANDING": [], "MARGARIN_BUTTER": [],
        "CHOCOLATE_DRINK": [], "SANDWICH_WASA": [], "GRISSINI": [],
        "MELLOMBAR": [], "MAISKAKER": [], "OTHER": []
    }

    for cat_id, name in class_names.items():
        name_upper = name.upper()
        if "EGG" in name_upper or "GÅRDSEGG" in name_upper:
            groups["EGGS"].append(cat_id)
        elif "KAPSEL" in name_upper or "KAPSLER" in name_upper:
            groups["COFFEE_CAPSULES"].append(cat_id)
        elif "FILTERMALT" in name_upper or "KOKMALT" in name_upper or "PRESSMALT" in name_upper:
            groups["COFFEE_GROUND"].append(cat_id)
        elif "HELE BØNNER" in name_upper:
            groups["COFFEE_BEANS"].append(cat_id)
        elif "NESCAFE" in name_upper or "INSTANT" in name_upper or "PULVERKAFFE" in name_upper:
            groups["COFFEE_INSTANT"].append(cat_id)
        elif "TE " in name_upper or "TEA " in name_upper or "TE " in name_upper or name_upper.endswith(" TE") or "URTETE" in name_upper or "CHAI" in name_upper or "ROOIBOS" in name_upper:
            groups["TEA"].append(cat_id)
        elif "KNEKKEBRØD" in name_upper or "KNEKKE" in name_upper or "KNEKKS" in name_upper:
            groups["KNEKKEBRØD"].append(cat_id)
        elif "FLATBRØD" in name_upper or "LEFSA" in name_upper:
            groups["FLATBRØD"].append(cat_id)
        elif "MÜSLI" in name_upper or "MUSLI" in name_upper or "GRANOLA" in name_upper or "HAVREGRØT" in name_upper or "HAVREGRYN" in name_upper:
            groups["MÜSLI_GRANOLA"].append(cat_id)
        elif "FROKOSTBLANDING" in name_upper or "CHEERIOS" in name_upper or "COCO POPS" in name_upper or "CORN FLAKES" in name_upper or "SMACKS" in name_upper or "FROKOST-TALL" in name_upper:
            groups["FROKOSTBLANDING"].append(cat_id)
        elif "MARGARIN" in name_upper or "SMØR" in name_upper or "BREMYKT" in name_upper or "FLORA" in name_upper or "HJERTEGO" in name_upper or "MELANGE" in name_upper or "BRELETT" in name_upper or "OLIVERO" in name_upper:
            groups["MARGARIN_BUTTER"].append(cat_id)
        elif "SJOKOLADEDRIKK" in name_upper or "O'BOY" in name_upper or "KAKAO" in name_upper or "NESQUIK" in name_upper or "SOLBÆRTODDY" in name_upper:
            groups["CHOCOLATE_DRINK"].append(cat_id)
        elif "SANDWICH" in name_upper:
            groups["SANDWICH_WASA"].append(cat_id)
        elif "GRISSINI" in name_upper:
            groups["GRISSINI"].append(cat_id)
        elif "MELLOMBAR" in name_upper:
            groups["MELLOMBAR"].append(cat_id)
        elif "MAISKAKER" in name_upper or "RISKAKER" in name_upper:
            groups["MAISKAKER"].append(cat_id)
        else:
            groups["OTHER"].append(cat_id)

    for group_name, cat_ids in sorted(groups.items()):
        if not cat_ids:
            continue
        total = sum(class_counts.get(c, 0) for c in cat_ids)
        avg = total / len(cat_ids) if cat_ids else 0
        weakest = min(cat_ids, key=lambda c: class_counts.get(c, 0))
        weakest_count = class_counts.get(weakest, 0)
        print(f"\n{group_name} ({len(cat_ids)} classes, {total} total annotations, avg {avg:.0f}/class)")
        print(f"  Weakest: [{weakest}] {class_names.get(weakest, '?')} ({weakest_count} annotations)")
        # List all if <10
        if len(cat_ids) <= 15:
            for c in sorted(cat_ids, key=lambda x: class_counts.get(x, 0)):
                print(f"    [{c}] {class_counts.get(c, 0):>4} annotations - {class_names.get(c, '?')}")

    # ============================================================
    # 6. SPATIAL ANALYSIS (where on shelf are products placed)
    # ============================================================
    print()
    print("=== SPATIAL ANALYSIS (product positions on shelf) ===")

    # Group by vertical position (top/middle/bottom shelf)
    shelf_positions = {"top_third": 0, "middle_third": 0, "bottom_third": 0}
    for ann in annotations:
        img = images[ann["image_id"]]
        cy = (ann["bbox"][1] + ann["bbox"][3]/2) / img["height"]
        if cy < 0.33:
            shelf_positions["top_third"] += 1
        elif cy < 0.66:
            shelf_positions["middle_third"] += 1
        else:
            shelf_positions["bottom_third"] += 1

    total_anns = len(annotations)
    for pos, count in shelf_positions.items():
        print(f"  {pos}: {count} annotations ({100*count/total_anns:.1f}%)")

    # Per-class spatial tendency
    print(f"\n--- Classes found mostly at top of shelf (cy < 0.3) ---")
    top_classes = []
    for cat_id in class_counts:
        bboxes = class_bboxes[cat_id]
        if len(bboxes) < 5:
            continue
        avg_cy = sum(b["cy"] for b in bboxes) / len(bboxes)
        if avg_cy < 0.3:
            top_classes.append((cat_id, avg_cy, len(bboxes)))
    for cat_id, avg_cy, n in sorted(top_classes, key=lambda x: x[1])[:10]:
        print(f"  [{cat_id}] cy={avg_cy:.2f} n={n} - {class_names.get(cat_id, '?')}")

    print(f"\n--- Classes found mostly at bottom of shelf (cy > 0.7) ---")
    bottom_classes = []
    for cat_id in class_counts:
        bboxes = class_bboxes[cat_id]
        if len(bboxes) < 5:
            continue
        avg_cy = sum(b["cy"] for b in bboxes) / len(bboxes)
        if avg_cy > 0.7:
            bottom_classes.append((cat_id, avg_cy, len(bboxes)))
    for cat_id, avg_cy, n in sorted(bottom_classes, key=lambda x: -x[1])[:10]:
        print(f"  [{cat_id}] cy={avg_cy:.2f} n={n} - {class_names.get(cat_id, '?')}")

    # ============================================================
    # 7. CONFUSABLE CLASSES (same group, similar sizes)
    # ============================================================
    print()
    print("=== POTENTIALLY CONFUSABLE CLASSES ===")
    print("(Same product group + similar bbox sizes = hard to distinguish)")

    for group_name, cat_ids in groups.items():
        if group_name == "OTHER" or len(cat_ids) < 3:
            continue
        # Find pairs with similar median area
        pairs = []
        for i in range(len(cat_ids)):
            for j in range(i+1, len(cat_ids)):
                c1, c2 = cat_ids[i], cat_ids[j]
                if c1 not in class_size_stats or c2 not in class_size_stats:
                    continue
                a1 = class_size_stats[c1]["median_rel_area"]
                a2 = class_size_stats[c2]["median_rel_area"]
                if a1 > 0 and a2 > 0:
                    ratio = max(a1, a2) / min(a1, a2)
                    if ratio < 1.5:  # Similar size
                        pairs.append((c1, c2, ratio, a1, a2))

        if pairs:
            print(f"\n{group_name} - {len(pairs)} confusable pairs:")
            for c1, c2, ratio, a1, a2 in sorted(pairs, key=lambda x: x[2])[:5]:
                n1 = class_names.get(c1, "?")[:35]
                n2 = class_names.get(c2, "?")[:35]
                cnt1 = class_counts.get(c1, 0)
                cnt2 = class_counts.get(c2, 0)
                print(f"  [{c1}] {n1:<37} ({cnt1:>3} ann) vs [{c2}] {n2:<37} ({cnt2:>3} ann) size_ratio={ratio:.2f}")

    # ============================================================
    # 8. ANNOTATION QUALITY SIGNALS
    # ============================================================
    print()
    print("=== ANNOTATION QUALITY SIGNALS ===")

    # Very small bboxes (potentially hard/noisy)
    tiny_bboxes = []
    for ann in annotations:
        img = images[ann["image_id"]]
        rel_area = (ann["bbox"][2] * ann["bbox"][3]) / (img["width"] * img["height"])
        if rel_area < 0.0005:  # less than 0.05% of image
            tiny_bboxes.append(ann)
    print(f"Very small bboxes (< 0.05% of image): {len(tiny_bboxes)} ({100*len(tiny_bboxes)/len(annotations):.1f}%)")

    # Very large bboxes
    large_bboxes = [a for a in annotations if (a["bbox"][2] * a["bbox"][3]) / (images[a["image_id"]]["width"] * images[a["image_id"]]["height"]) > 0.05]
    print(f"Very large bboxes (> 5% of image): {len(large_bboxes)} ({100*len(large_bboxes)/len(annotations):.1f}%)")

    # Extreme aspect ratios
    extreme_aspect = [a for a in annotations if a["bbox"][2] / max(a["bbox"][3], 1) > 5 or a["bbox"][3] / max(a["bbox"][2], 1) > 5]
    print(f"Extreme aspect ratio (>5:1): {len(extreme_aspect)}")

    # Category 300 (empty name) and 355 (unknown_product)
    print(f"\nCategory 300 (empty name): {class_counts.get(300, 0)} annotations")
    print(f"Category 355 (unknown_product): {class_counts.get(355, 0)} annotations")

    # ============================================================
    # 9. PHOTO-TAKING RECOMMENDATIONS
    # ============================================================
    print()
    print("=" * 80)
    print("=== ACTIONABLE PHOTO-TAKING RECOMMENDATIONS ===")
    print("=" * 80)

    # Classes with <=5 annotations - absolute priority
    critical_classes = [(c, cnt) for c, cnt in sorted_classes if cnt <= 5]
    important_classes = [(c, cnt) for c, cnt in sorted_classes if 5 < cnt <= 15]

    print(f"\n🔴 CRITICAL ({len(critical_classes)} classes with <=5 annotations):")
    print("These products MUST be photographed. Each photo is worth massive mAP gain.")
    for cat_id, count in critical_classes:
        name = class_names.get(cat_id, "?")
        print(f"  [{cat_id}] {count} ann - {name}")

    print(f"\n🟡 IMPORTANT ({len(important_classes)} classes with 6-15 annotations):")
    for cat_id, count in important_classes:
        name = class_names.get(cat_id, "?")
        print(f"  [{cat_id}] {count} ann - {name}")

    print(f"\n📸 PHOTOGRAPHY TIPS:")
    print("1. STORES: Visit SPAR, Meny, Kiwi - these are the chains in the dataset")
    print("2. AISLES TO FOCUS ON:")

    # Group critical classes by aisle
    critical_aisles = defaultdict(list)
    for cat_id, count in critical_classes + important_classes:
        name = class_names.get(cat_id, "").upper()
        if "EGG" in name or "GÅRDSEGG" in name:
            critical_aisles["EGGS/DAIRY (cooler section)"].append(class_names.get(cat_id, ""))
        elif any(k in name for k in ["KAFFE", "COFFEE", "KAPSEL", "FILTERMALT", "BØNNER", "NESCAFE"]):
            critical_aisles["COFFEE aisle"].append(class_names.get(cat_id, ""))
        elif any(k in name for k in ["TE ", "TEA", "CHAI", "ROOIBOS", "PUKKA", "LIPTON", "TWININGS"]):
            critical_aisles["TEA aisle"].append(class_names.get(cat_id, ""))
        elif any(k in name for k in ["KNEKKE", "FLATBRØD", "WASA", "SIGDAL", "LEKSAND"]):
            critical_aisles["BREAD/CRISPBREAD aisle"].append(class_names.get(cat_id, ""))
        elif any(k in name for k in ["MÜSLI", "MUSLI", "GRANOLA", "FROKOST", "HAVRE", "CORN FLAKES", "CHEERIO"]):
            critical_aisles["CEREAL/BREAKFAST aisle"].append(class_names.get(cat_id, ""))
        elif any(k in name for k in ["MARGARIN", "SMØR", "FLORA", "MELANGE", "BREMYKT", "BRELETT"]):
            critical_aisles["BUTTER/MARGARINE (cooler)"].append(class_names.get(cat_id, ""))
        elif any(k in name for k in ["KAKAO", "SJOKOLADEDRIKK", "O'BOY", "NESQUIK", "TODDY"]):
            critical_aisles["HOT DRINKS (cocoa section)"].append(class_names.get(cat_id, ""))
        else:
            critical_aisles["OTHER/SPECIALTY"].append(class_names.get(cat_id, ""))

    for aisle, products in sorted(critical_aisles.items(), key=lambda x: -len(x[1])):
        print(f"\n   {aisle} ({len(products)} products needed):")
        for p in products[:10]:
            print(f"     - {p}")
        if len(products) > 10:
            print(f"     ... and {len(products)-10} more")

    print(f"\n📐 PHOTO TECHNIQUE:")
    print(f"   - Take WIDE shots of full shelf sections (captures many products at once)")
    print(f"   - Resolution: highest your phone allows (dataset has up to 5712px wide)")
    print(f"   - Lighting: well-lit aisles, avoid flash reflections on packaging")
    print(f"   - Angle: straight-on (perpendicular to shelf), minimize perspective distortion")
    print(f"   - Distance: ~1-2m from shelf (matches training data product sizes)")
    print(f"   - Cover ALL shelf levels (top to bottom) - dataset is ~evenly distributed")
    print(f"   - Multiple angles per aisle: left side, center, right side")
    print(f"   - Include price tags visible (helps with annotation/identification)")

    median_anns = sorted([s["n_annotations"] for s in image_stats])[len(image_stats)//2]
    print(f"\n📊 TARGET DENSITY:")
    print(f"   - Median products per image in dataset: {median_anns}")
    print(f"   - Densest image has {image_stats[0]['n_annotations']} products")
    print(f"   - Aim for 30-100 visible products per photo")

    # ============================================================
    # 10. BUILD FULL REPORT
    # ============================================================
    report = {
        "overview": {
            "n_images": len(images),
            "n_annotations": len(annotations),
            "n_categories": len(categories),
            "median_annotations_per_image": median_anns,
            "resolution": {
                "min_width": min(widths), "max_width": max(widths),
                "min_height": min(heights), "max_height": max(heights),
            }
        },
        "weak_classes": weak_classes,
        "class_size_stats": {str(k): v for k, v in class_size_stats.items()},
        "image_stats": image_stats[:20],  # top 20 densest
        "product_groups": {k: [{"cat_id": c, "name": class_names.get(c, "?"), "count": class_counts.get(c, 0)} for c in v] for k, v in groups.items()},
        "photo_recommendations": {
            "critical_classes": [{"cat_id": c, "count": cnt, "name": class_names.get(c, "?")} for c, cnt in critical_classes],
            "important_classes": [{"cat_id": c, "count": cnt, "name": class_names.get(c, "?")} for c, cnt in important_classes],
            "critical_aisles": {k: v for k, v in critical_aisles.items()},
        },
        "spatial": shelf_positions,
        "quality_signals": {
            "tiny_bboxes": len(tiny_bboxes),
            "large_bboxes": len(large_bboxes),
            "extreme_aspect": len(extreme_aspect),
            "empty_name_cat300": class_counts.get(300, 0),
            "unknown_cat355": class_counts.get(355, 0),
        }
    }

    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n\nFull report saved to: {REPORT_OUT}")


if __name__ == "__main__":
    analyze()

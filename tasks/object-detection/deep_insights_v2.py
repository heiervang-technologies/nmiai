#!/usr/bin/env python3
"""
Deep Insights V2 - Cross-reference analysis:
1. Existing store photos quality audit (rotation, blur, coverage gaps)
2. Training data vs store photo gap analysis
3. Per-class difficulty score combining: annotation count, bbox size, confusability, image diversity
4. Prioritized shopping list for next store visit
5. Lighting/angle distribution in training data
"""

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image, ImageStat
import math

COCO_PATH = Path(__file__).parent / "data-creation/data/coco_dataset/train/annotations.json"
DATA_YAML = Path(__file__).parent / "yolo-approach/dataset/data.yaml"
STORE_PHOTOS = Path(__file__).parent / "data-creation/data/store_photos"
TRAIN_IMAGES = Path(__file__).parent / "yolo-approach/dataset/images/train"
REPORT_V1 = Path(__file__).parent / "deep_insights_report.json"
REPORT_OUT = Path(__file__).parent / "deep_insights_v2_report.json"


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


def analyze_image_quality(img_path):
    """Analyze a single image for quality metrics."""
    try:
        img = Image.open(img_path)
        w, h = img.size

        # Check orientation (rotated = width < height for landscape scenes)
        is_rotated = w < h  # Store shelves are landscape, so portrait = rotated

        # Brightness analysis
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # Sample center region for brightness
        stat = ImageStat.Stat(img)
        avg_brightness = sum(stat.mean) / 3

        # Check if blurry (using Laplacian variance proxy - edge density)
        # Simple proxy: variance of pixel values (higher = more detail = less blur)
        variance = sum(stat.var) / 3

        return {
            "file": img_path.name,
            "width": w,
            "height": h,
            "megapixels": round(w * h / 1e6, 2),
            "is_rotated": is_rotated,
            "avg_brightness": round(avg_brightness, 1),
            "variance": round(variance, 1),
            "likely_blurry": variance < 1500,  # heuristic threshold
            "aspect_ratio": round(w / h, 2),
        }
    except Exception as e:
        return {"file": img_path.name, "error": str(e)}


def analyze_training_lighting():
    """Analyze lighting conditions across training images."""
    results = []
    train_dir = TRAIN_IMAGES
    if not train_dir.exists():
        return results

    for img_path in sorted(train_dir.iterdir())[:50]:  # sample 50
        if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png'):
            continue
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            stat = ImageStat.Stat(img)
            avg_brightness = sum(stat.mean) / 3
            w, h = img.size
            results.append({
                "file": img_path.name,
                "brightness": round(avg_brightness, 1),
                "width": w,
                "height": h,
            })
        except:
            pass
    return results


def compute_difficulty_score(cat_id, class_counts, class_names, class_images,
                              class_bboxes, total_images):
    """
    Composite difficulty score (0-100, higher = harder/more important to improve).
    Factors:
    - Annotation scarcity (fewer = harder)
    - Image diversity (fewer unique images = harder)
    - Bbox size (smaller = harder to detect)
    - Confusability (more similar products in same group = harder)
    """
    count = class_counts.get(cat_id, 0)
    n_images = len(class_images.get(cat_id, set()))

    # Scarcity score (0-40): fewer annotations = higher score
    if count == 0:
        scarcity = 40
    elif count <= 5:
        scarcity = 35
    elif count <= 15:
        scarcity = 28
    elif count <= 30:
        scarcity = 20
    elif count <= 60:
        scarcity = 12
    elif count <= 100:
        scarcity = 6
    else:
        scarcity = max(0, 3 - (count - 100) / 100)

    # Image diversity score (0-20): fewer unique images = higher
    if n_images == 0:
        diversity = 20
    elif n_images <= 2:
        diversity = 18
    elif n_images <= 5:
        diversity = 14
    elif n_images <= 10:
        diversity = 8
    else:
        diversity = max(0, 4 - (n_images - 10) / 5)

    # Size score (0-20): smaller bboxes = higher (harder to detect)
    bboxes = class_bboxes.get(cat_id, [])
    if bboxes:
        median_area = sorted([b["rel_area"] for b in bboxes])[len(bboxes)//2]
        if median_area < 0.001:
            size_score = 20
        elif median_area < 0.005:
            size_score = 14
        elif median_area < 0.01:
            size_score = 8
        elif median_area < 0.02:
            size_score = 4
        else:
            size_score = 1
    else:
        size_score = 15  # unknown = assume hard

    # Confusability score (0-20): based on product group density
    name = class_names.get(cat_id, "").upper()
    confuse = 5  # default
    if any(k in name for k in ["KNEKKE", "WASA", "SIGDAL", "LEKSAND"]):
        confuse = 18  # 30+ similar crispbreads
    elif any(k in name for k in ["EGG", "GÅRDSEGG"]):
        confuse = 16  # 15+ egg brands, identical cartons
    elif any(k in name for k in ["KAFFE", "COFFEE", "EVERGOOD", "ALI ", "FRIELE"]):
        confuse = 15  # many similar coffee bags
    elif any(k in name for k in ["TE ", "TEA ", "TWININGS", "LIPTON", "PUKKA"]):
        confuse = 14  # many similar tea boxes
    elif any(k in name for k in ["MÜSLI", "MUSLI", "GRANOLA"]):
        confuse = 12
    elif any(k in name for k in ["MARGARIN", "SMØR", "FLORA", "MELANGE"]):
        confuse = 12
    elif any(k in name for k in ["SANDWICH"]):
        confuse = 10

    total = scarcity + diversity + size_score + confuse
    return {
        "total": round(min(total, 100), 1),
        "scarcity": round(scarcity, 1),
        "diversity": round(diversity, 1),
        "size": round(size_score, 1),
        "confusability": round(confuse, 1),
    }


def main():
    # Load COCO
    with open(COCO_PATH) as f:
        coco = json.load(f)

    class_names = load_class_names()
    images = {img["id"]: img for img in coco["images"]}
    annotations = coco["annotations"]

    class_counts = Counter()
    class_bboxes = defaultdict(list)
    class_images = defaultdict(set)

    for ann in annotations:
        cat_id = ann["category_id"]
        img_id = ann["image_id"]
        img = images[img_id]
        bbox = ann["bbox"]

        class_counts[cat_id] += 1
        class_images[cat_id].add(img_id)
        class_bboxes[cat_id].append({
            "w": bbox[2], "h": bbox[3],
            "rel_area": (bbox[2] * bbox[3]) / (img["width"] * img["height"]),
            "cx": (bbox[0] + bbox[2]/2) / img["width"],
            "cy": (bbox[1] + bbox[3]/2) / img["height"],
        })

    # ============================================================
    # 1. STORE PHOTOS QUALITY AUDIT
    # ============================================================
    print("=" * 80)
    print("1. EXISTING STORE PHOTOS QUALITY AUDIT")
    print("=" * 80)

    store_quality = []
    if STORE_PHOTOS.exists():
        for p in sorted(STORE_PHOTOS.iterdir()):
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                quality = analyze_image_quality(p)
                store_quality.append(quality)

    rotated = sum(1 for q in store_quality if q.get("is_rotated", False))
    blurry = sum(1 for q in store_quality if q.get("likely_blurry", False))
    dark = sum(1 for q in store_quality if q.get("avg_brightness", 128) < 80)

    print(f"\nTotal store photos: {len(store_quality)}")
    print(f"Rotated (portrait orientation): {rotated}/{len(store_quality)}")
    print(f"Likely blurry (low variance): {blurry}/{len(store_quality)}")
    print(f"Dark (brightness < 80): {dark}/{len(store_quality)}")

    print(f"\n--- Problem photos ---")
    for q in store_quality:
        issues = []
        if q.get("is_rotated"): issues.append("ROTATED")
        if q.get("likely_blurry"): issues.append("BLURRY")
        if q.get("avg_brightness", 128) < 80: issues.append("DARK")
        if issues:
            print(f"  {q['file']}: {', '.join(issues)} (brightness={q.get('avg_brightness','?')}, variance={q.get('variance','?')})")

    # ============================================================
    # 2. TRAINING IMAGE LIGHTING ANALYSIS
    # ============================================================
    print(f"\n{'='*80}")
    print("2. TRAINING DATA LIGHTING CONDITIONS")
    print("=" * 80)

    train_lighting = analyze_training_lighting()
    if train_lighting:
        brightnesses = [t["brightness"] for t in train_lighting]
        print(f"Sample size: {len(train_lighting)} images")
        print(f"Brightness range: {min(brightnesses):.0f} - {max(brightnesses):.0f}")
        print(f"Average brightness: {sum(brightnesses)/len(brightnesses):.0f}")
        print(f"Median brightness: {sorted(brightnesses)[len(brightnesses)//2]:.0f}")

        # Categorize
        very_bright = sum(1 for b in brightnesses if b > 160)
        normal = sum(1 for b in brightnesses if 80 <= b <= 160)
        dim = sum(1 for b in brightnesses if b < 80)
        print(f"\nLighting distribution:")
        print(f"  Bright (>160): {very_bright} ({100*very_bright/len(brightnesses):.0f}%)")
        print(f"  Normal (80-160): {normal} ({100*normal/len(brightnesses):.0f}%)")
        print(f"  Dim (<80): {dim} ({100*dim/len(brightnesses):.0f}%)")

        # Resolution distribution
        resolutions = [(t["width"], t["height"]) for t in train_lighting]
        print(f"\nResolution clusters:")
        res_counter = Counter()
        for w, h in resolutions:
            bucket = f"{(w//500)*500}-{(w//500)*500+499}px"
            res_counter[bucket] += 1
        for bucket, count in res_counter.most_common():
            print(f"  {bucket}: {count} images")

    # ============================================================
    # 3. COMPOSITE DIFFICULTY SCORES
    # ============================================================
    print(f"\n{'='*80}")
    print("3. COMPOSITE DIFFICULTY SCORES (higher = needs more attention)")
    print("=" * 80)

    difficulty_scores = {}
    for cat_id in range(356):
        difficulty_scores[cat_id] = compute_difficulty_score(
            cat_id, class_counts, class_names, class_images,
            class_bboxes, len(images)
        )

    # Sort by total difficulty
    ranked = sorted(difficulty_scores.items(), key=lambda x: -x[1]["total"])

    print(f"\n{'Rank':<5} {'CatID':<6} {'Total':<7} {'Scarce':<8} {'Divers':<8} {'Size':<6} {'Confuse':<8} {'Count':<6} {'Name'}")
    print("-" * 110)
    for rank, (cat_id, scores) in enumerate(ranked[:60], 1):
        name = class_names.get(cat_id, "?")[:40]
        count = class_counts.get(cat_id, 0)
        print(f"{rank:<5} {cat_id:<6} {scores['total']:<7} {scores['scarcity']:<8} {scores['diversity']:<8} {scores['size']:<6} {scores['confusability']:<8} {count:<6} {name}")

    # ============================================================
    # 4. COVERAGE GAP ANALYSIS (store photos vs training)
    # ============================================================
    print(f"\n{'='*80}")
    print("4. STORE PHOTO COVERAGE ANALYSIS")
    print("=" * 80)

    # Which aisles are covered by existing store photos?
    # Based on visual inspection: existing photos are from a Kiwi store,
    # showing crispbread/flatbread aisle and snacks/biscuits aisle
    # Many are rotated (portrait) and some are blurry

    print(f"""
EXISTING STORE PHOTOS (44 images from Kiwi):
  - Aisles covered: crispbread/flatbread, snacks/biscuits, some cereals
  - Quality issues: {rotated} rotated, {blurry} likely blurry
  - Missing aisles: COFFEE, TEA, EGGS, BUTTER/MARGARINE, HOT DRINKS

GAPS TO FILL ON NEXT VISIT:
  Priority 1: Coffee aisle (21 weak classes, ZERO store photos)
  Priority 2: Tea aisle (22 weak classes, ZERO store photos)
  Priority 3: Egg cooler (15 weak classes, ZERO store photos)
  Priority 4: Butter/margarine cooler (15 weak classes, ZERO store photos)
  Priority 5: Cereal aisle (16 weak classes, partial coverage)
  Priority 6: Better quality re-shoots of crispbread (straight-on, not rotated)
""")

    # ============================================================
    # 5. STORE-SPECIFIC RECOMMENDATIONS
    # ============================================================
    print(f"{'='*80}")
    print("5. STORE VISIT CHECKLIST")
    print("=" * 80)

    print("""
BEFORE YOU GO:
  - Phone at MAX resolution
  - Enough storage space (each photo ~5-10MB at max res)
  - Phone held LANDSCAPE (horizontal), NOT portrait
  - No flash

AT THE STORE (visit Meny or SPAR if possible, Kiwi already covered):

  STOP 1: COFFEE AISLE (biggest gap, 21 weak classes)
    - Wide shot of entire ground coffee section (Evergood, Ali, Friele, Jacobs)
    - Wide shot of whole bean section
    - Wide shot of coffee capsule section (COTW, Dolce Gusto, Tassimo)
    - Close-up of Jacobs Utvalgte section (Tropisk Aroma, Excelso, Bla Java)
    - Close-up of Nescafe instant section
    - Close-up of filter/accessories (Kaffefilter Presskanne)
    Total: ~8-10 photos

  STOP 2: TEA AISLE (22 weak classes)
    - Wide shot of full tea section
    - Close-up of Twinings range (Earl Grey, English Breakfast, Chai, fruit teas)
    - Close-up of Lipton range (Green Tea, Icetea powders)
    - Close-up of Pukka range (Chai, Lemon Ginger, Feel New, Clean Matcha)
    - Close-up of Confecta/organic section
    Total: ~6-8 photos

  STOP 3: EGG COOLER (15 weak classes)
    - Wide shot of entire egg section
    - Close-up of each egg brand shelf level
    - IMPORTANT: many local farm brands (Leka, Galaavolden, Torresvik, Fana, Sunnmorsegg)
    - These may only be at specific regional stores!
    Total: ~4-6 photos

  STOP 4: BUTTER/MARGARINE COOLER (15 weak classes)
    - Wide shot of full section
    - Close-up of Soft Flora / Vita Hjertego area
    - Close-up of Melange / Bremykt area
    - Close-up of Tine Meierismoer / Smor Usaltet
    Total: ~4-5 photos

  STOP 5: CEREAL/BREAKFAST (16 weak classes)
    - Wide shot (some already in store photos but rotated/blurry)
    - Close-up of granola section (Bare Bra, Start!, Synnove)
    - Close-up of muesli section (AXA variants)
    - Close-up of specialty (Lion, Weetos, Alpen)
    Total: ~4-5 photos

  STOP 6: CRISPBREAD (re-shoot, 15 weak classes)
    - Wide shot LANDSCAPE, straight-on (fix the rotated ones)
    - Close-up of Wasa Sandwich varieties
    - Close-up of gluten-free section (Sigdal, Schar, Brisk)
    Total: ~3-4 photos

  STOP 7: MISCELLANEOUS (if you spot them)
    - Jarlsberg cheese (only 1 annotation!)
    - Entrecote / Short Ribs meat section (1 annotation each!)
    - Nutella Biscuits
    - Gifflar Kanel / Bringebaer (Paagen)
    - Torres Truffle chips
    - Oreo O's cereal
    Total: ~3-5 photos

TOTAL TARGET: ~35-45 new photos, 25-30 minutes in store
""")

    # ============================================================
    # BUILD REPORT
    # ============================================================
    report = {
        "store_photo_audit": {
            "total": len(store_quality),
            "rotated": rotated,
            "blurry": blurry,
            "dark": dark,
            "details": store_quality,
        },
        "training_lighting": {
            "sample_size": len(train_lighting),
            "stats": {
                "avg": round(sum(t["brightness"] for t in train_lighting) / max(len(train_lighting), 1), 1),
                "min": min((t["brightness"] for t in train_lighting), default=0),
                "max": max((t["brightness"] for t in train_lighting), default=0),
            } if train_lighting else {},
        },
        "difficulty_ranking": [
            {
                "rank": rank,
                "cat_id": cat_id,
                "name": class_names.get(cat_id, "?"),
                "count": class_counts.get(cat_id, 0),
                "scores": scores,
            }
            for rank, (cat_id, scores) in enumerate(ranked, 1)
        ],
        "coverage_gaps": {
            "coffee": "ZERO store photos, 21 weak classes",
            "tea": "ZERO store photos, 22 weak classes",
            "eggs": "ZERO store photos, 15 weak classes",
            "butter_margarine": "ZERO store photos, 15 weak classes",
            "cereal": "Partial coverage (blurry/rotated), 16 weak classes",
            "crispbread": "Has coverage but quality issues, 15 weak classes",
        },
    }

    with open(REPORT_OUT, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\nFull V2 report saved to: {REPORT_OUT}")


if __name__ == "__main__":
    main()

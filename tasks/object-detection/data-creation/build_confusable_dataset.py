"""
Build specialized training data for confusable category pairs.
For each confusable pair, extract crops showing distinguishing features
and create hard-example training images.
"""
import json
import random
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "confusable_training"
SEED = 42

random.seed(SEED)


def main():
    # Load triplets
    trips = json.load(open(DATA_DIR / "hard_negative_pairs" / "triplets.json"))

    # Load category names
    ann = json.load(open(DATA_DIR / "coco_dataset" / "train" / "annotations.json"))
    cat_names = {c["id"]: c["name"] for c in ann["categories"]}

    # Find top confusable pairs
    pair_max_sim = {}
    for t in trips:
        pair = tuple(sorted([t["anchor_cat"], t["negative_cat"]]))
        pair_max_sim[pair] = max(pair_max_sim.get(pair, 0), t["similarity"])

    top_pairs = sorted(pair_max_sim.items(), key=lambda x: -x[1])[:50]

    print(f"Building confusable training for {len(top_pairs)} pairs...")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # For each pair, create comparison images and extra training data
    pair_data = []
    for (cat_a, cat_b), sim in top_pairs:
        name_a = cat_names.get(cat_a, f"cat_{cat_a}")
        name_b = cat_names.get(cat_b, f"cat_{cat_b}")

        pair_data.append({
            "cat_a": cat_a,
            "cat_b": cat_b,
            "name_a": name_a,
            "name_b": name_b,
            "similarity": sim,
        })

    # Save pair analysis
    with open(OUTPUT_DIR / "confusable_pairs.json", "w") as f:
        json.dump(pair_data, f, indent=2, ensure_ascii=False)

    # Extract crops for confusable pairs from mega dataset
    mega_labels = DATA_DIR / "mega_dataset" / "train" / "labels"
    mega_images = DATA_DIR / "mega_dataset" / "train" / "images"

    # Find images containing confusable categories
    confusable_cats = set()
    for (a, b), _ in top_pairs:
        confusable_cats.add(a)
        confusable_cats.add(b)

    cat_images = defaultdict(list)
    for lf in mega_labels.glob("*.txt"):
        for line in open(lf):
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                if cid in confusable_cats:
                    cat_images[cid].append({
                        "stem": lf.stem,
                        "bbox": [float(p) for p in parts[1:5]],
                    })

    # Extract crops for each confusable category
    crops_dir = OUTPUT_DIR / "crops"
    total_crops = 0

    for cat_id in sorted(confusable_cats):
        entries = cat_images[cat_id][:50]  # Max 50 per category
        cat_crop_dir = crops_dir / str(cat_id)
        cat_crop_dir.mkdir(parents=True, exist_ok=True)

        for i, entry in enumerate(entries):
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                c = mega_images / (entry["stem"] + ext)
                if c.exists() or c.is_symlink():
                    img_path = c
                    break

            if not img_path:
                continue

            try:
                real = img_path.resolve()
                img = cv2.imread(str(real))
                if img is None:
                    continue
            except:
                continue

            h, w = img.shape[:2]
            cx, cy, bw, bh = entry["bbox"]

            x1 = max(0, int((cx - bw / 2) * w) - 5)
            y1 = max(0, int((cy - bh / 2) * h) - 5)
            x2 = min(w, int((cx + bw / 2) * w) + 5)
            y2 = min(h, int((cy + bh / 2) * h) + 5)

            if x2 - x1 < 20 or y2 - y1 < 20:
                continue

            crop = img[y1:y2, x1:x2]
            cv2.imwrite(str(cat_crop_dir / f"crop_{i:04d}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            total_crops += 1

    print(f"Extracted {total_crops} crops for {len(confusable_cats)} confusable categories")
    print(f"Pairs analysis: {OUTPUT_DIR / 'confusable_pairs.json'}")
    print(f"Crops: {crops_dir}")


if __name__ == "__main__":
    main()

"""
Boost weak categories in mega_dataset to reduce class imbalance.
Target: bring all categories to at least 500 annotations.
Uses augmentation on existing images containing weak categories.
"""
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
import cv2
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
MEGA_DIR = DATA_DIR / "mega_dataset"
TARGET_MIN = 500
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def augment_image(img):
    result = img.copy()
    # Color jitter
    hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:,:,0] = (hsv[:,:,0] + random.uniform(-15, 15)) % 180
    hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.7, 1.3), 0, 255)
    hsv[:,:,2] = np.clip(hsv[:,:,2] * random.uniform(0.7, 1.3), 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    # Slight blur
    if random.random() < 0.4:
        result = cv2.GaussianBlur(result, (3, 3), 0)
    # Noise
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return result


def main():
    train_dir = MEGA_DIR / "train"
    img_dir = train_dir / "images"
    lbl_dir = train_dir / "labels"

    # Count current annotations
    counts = Counter()
    cat_images = defaultdict(list)

    for lf in lbl_dir.glob("*.txt"):
        stem = lf.stem
        cats_in_img = set()
        for line in open(lf):
            parts = line.strip().split()
            if parts:
                cid = int(parts[0])
                counts[cid] += 1
                cats_in_img.add(cid)
        for cid in cats_in_img:
            cat_images[cid].append(stem)

    weak_cats = {cid: cnt for cid, cnt in counts.items() if cnt < TARGET_MIN}
    print(f"Categories under {TARGET_MIN}: {len(weak_cats)}")

    aug_count = 0
    for cat_id, current in sorted(weak_cats.items(), key=lambda x: x[1]):
        needed = TARGET_MIN - current
        sources = cat_images[cat_id]
        if not sources:
            continue

        generated = 0
        for attempt in range(needed * 3):
            if generated >= needed:
                break

            stem = random.choice(sources)
            # Find image
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = img_dir / (stem + ext)
                if candidate.exists() or candidate.is_symlink():
                    img_path = candidate
                    break

            if not img_path:
                continue

            try:
                real_path = img_path.resolve()
                img = cv2.imread(str(real_path))
                if img is None:
                    continue
            except:
                continue

            aug_img = augment_image(img)
            aug_name = f"megaboost_{cat_id}_{generated:04d}"

            cv2.imwrite(str(img_dir / f"{aug_name}.jpg"), aug_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Copy labels
            src_lbl = lbl_dir / (stem + ".txt")
            if src_lbl.exists():
                (lbl_dir / f"{aug_name}.txt").write_text(src_lbl.read_text())

            generated += 1
            aug_count += 1

        if generated > 0 and generated >= 10:
            print(f"  cat {cat_id}: {current} -> {current + generated}")

    # Recount
    new_counts = Counter()
    for lf in lbl_dir.glob("*.txt"):
        for line in open(lf):
            parts = line.strip().split()
            if parts:
                new_counts[int(parts[0])] += 1

    vals = sorted(new_counts.values())
    print(f"\nBoosted {aug_count} images")
    print(f"New min/cat: {vals[0]}")
    print(f"New balance ratio: {vals[-1]/vals[0]:.1f}x")
    print(f"Total train images: {len(list(img_dir.iterdir()))}")


if __name__ == "__main__":
    main()

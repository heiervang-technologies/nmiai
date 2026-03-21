"""
Build V7 balanced dataset: targeted augmentation for weak categories.
- Boost all categories under 150 annotations to at least 150
- Use copy-paste, color jitter, scale augmentation on weak-category images
- Maintain zero data leakage with large_clean_split val
"""
import json
import random
import shutil
from pathlib import Path
from collections import Counter, defaultdict
import cv2
import numpy as np

DATA_DIR = Path(__file__).parent / "data"
SOURCE_DIR = DATA_DIR / "large_clean_split"
OUTPUT_DIR = DATA_DIR / "yolo_v7_balanced"
TARGET_MIN = 150
SEED = 42

random.seed(SEED)
np.random.seed(SEED)


def load_labels(label_dir):
    """Load all labels, return {filename_stem: [(cls, cx, cy, w, h), ...]}"""
    labels = {}
    for lf in label_dir.glob("*.txt"):
        boxes = []
        for line in open(lf):
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])))
        labels[lf.stem] = boxes
    return labels


def augment_image(img, level=1):
    """Apply random augmentations. level 1=light, 2=medium, 3=heavy"""
    h, w = img.shape[:2]
    result = img.copy()
    
    # Color jitter
    if random.random() < 0.7:
        hsv = cv2.cvtColor(result, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + random.uniform(-10, 10)) % 180
        hsv[:,:,1] = np.clip(hsv[:,:,1] * random.uniform(0.8, 1.2), 0, 255)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * random.uniform(0.8, 1.2), 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Gaussian blur
    if random.random() < 0.3:
        ksize = random.choice([3, 5])
        result = cv2.GaussianBlur(result, (ksize, ksize), 0)
    
    # Noise
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(5, 15), result.shape).astype(np.int16)
        result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Random crop (keeping all boxes) - slight zoom
    if level >= 2 and random.random() < 0.5:
        crop_frac = random.uniform(0.85, 0.95)
        new_w = int(w * crop_frac)
        new_h = int(h * crop_frac)
        x_off = random.randint(0, w - new_w)
        y_off = random.randint(0, h - new_h)
        result = result[y_off:y_off+new_h, x_off:x_off+new_w]
        result = cv2.resize(result, (w, h))
        # Return crop params for box adjustment
        return result, (x_off/w, y_off/h, new_w/w, new_h/h)
    
    return result, None


def adjust_boxes(boxes, crop_params):
    """Adjust box coordinates for crop augmentation."""
    if crop_params is None:
        return boxes
    
    x_off, y_off, crop_w, crop_h = crop_params
    adjusted = []
    for cls, cx, cy, w, h in boxes:
        new_cx = (cx - x_off) / crop_w
        new_cy = (cy - y_off) / crop_h
        new_w = w / crop_w
        new_h = h / crop_h
        # Check if box is still mostly in frame
        if 0.1 < new_cx < 0.9 and 0.1 < new_cy < 0.9:
            new_cx = max(0, min(1, new_cx))
            new_cy = max(0, min(1, new_cy))
            new_w = max(0.01, min(1, new_w))
            new_h = max(0.01, min(1, new_h))
            adjusted.append((cls, new_cx, new_cy, new_w, new_h))
    return adjusted


def main():
    print("Building V7 balanced dataset...")
    
    # Load source labels
    train_labels = load_labels(SOURCE_DIR / "train" / "labels")
    print(f"Source: {len(train_labels)} training images")
    
    # Count per category
    cat_counts = Counter()
    cat_images = defaultdict(list)  # cat_id -> [stem, ...]
    for stem, boxes in train_labels.items():
        for cls, *_ in boxes:
            cat_counts[cls] += 1
            if stem not in cat_images[cls]:
                cat_images[cls].append(stem)
    
    # Find weak categories
    weak_cats = {cid: cnt for cid, cnt in cat_counts.items() if cnt < TARGET_MIN}
    print(f"Weak categories (<{TARGET_MIN} ann): {len(weak_cats)}")
    
    # Setup output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)
    
    # Copy val as-is (symlinks)
    val_count = 0
    for img in (SOURCE_DIR / "val" / "images").iterdir():
        dst = OUTPUT_DIR / "val" / "images" / img.name
        if not dst.exists():
            dst.symlink_to(img.resolve())
        lsrc = SOURCE_DIR / "val" / "labels" / (img.stem + ".txt")
        ldst = OUTPUT_DIR / "val" / "labels" / (img.stem + ".txt")
        if lsrc.exists() and not ldst.exists():
            shutil.copy2(lsrc, ldst)
        val_count += 1
    print(f"Val: {val_count} images (unchanged)")
    
    # Copy all existing train (symlinks)
    train_count = 0
    for img in (SOURCE_DIR / "train" / "images").iterdir():
        dst = OUTPUT_DIR / "train" / "images" / img.name
        if not dst.exists():
            dst.symlink_to(img.resolve())
        lsrc = SOURCE_DIR / "train" / "labels" / (img.stem + ".txt")
        ldst = OUTPUT_DIR / "train" / "labels" / (img.stem + ".txt")
        if lsrc.exists() and not ldst.exists():
            shutil.copy2(lsrc, ldst)
        train_count += 1
    print(f"Copied {train_count} existing train images")
    
    # Generate targeted augmentations for weak categories
    aug_count = 0
    for cat_id, current_count in sorted(weak_cats.items(), key=lambda x: x[1]):
        needed = TARGET_MIN - current_count
        source_stems = cat_images[cat_id]
        if not source_stems:
            continue
        
        generated = 0
        attempts = 0
        while generated < needed and attempts < needed * 3:
            attempts += 1
            stem = random.choice(source_stems)
            
            # Find the image
            img_path = None
            for ext in [".jpg", ".jpeg", ".png"]:
                candidate = SOURCE_DIR / "train" / "images" / (stem + ext)
                if candidate.exists():
                    img_path = candidate
                    break
                # Follow symlinks
                if candidate.is_symlink():
                    img_path = candidate
                    break
            
            if img_path is None:
                continue
            
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            boxes = train_labels.get(stem, [])
            if not boxes:
                continue
            
            # Augment
            level = 2 if needed > 50 else 1
            aug_img, crop_params = augment_image(img, level)
            aug_boxes = adjust_boxes(boxes, crop_params)
            
            if not aug_boxes:
                continue
            
            # Save
            aug_name = f"v7aug_{cat_id}_{generated:04d}"
            cv2.imwrite(str(OUTPUT_DIR / "train" / "images" / f"{aug_name}.jpg"), aug_img, 
                       [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            with open(OUTPUT_DIR / "train" / "labels" / f"{aug_name}.txt", "w") as f:
                for cls, cx, cy, w, h in aug_boxes:
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
            
            generated += 1
            aug_count += 1
        
        if generated > 0:
            print(f"  cat {cat_id}: {current_count} -> {current_count + generated} (+{generated})")
    
    print(f"\nGenerated {aug_count} targeted augmentations")
    
    # Write dataset.yaml
    import yaml
    ann = json.load(open(DATA_DIR / "coco_dataset" / "train" / "annotations.json"))
    names = {c["id"]: c["name"] for c in ann["categories"]}
    
    yaml_data = {
        "path": str(OUTPUT_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names,
    }
    with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
        f.write("# V7 balanced dataset - targeted augmentation for weak categories\n")
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    # Final stats
    final_train = len(list((OUTPUT_DIR / "train" / "images").iterdir()))
    final_val = len(list((OUTPUT_DIR / "val" / "images").iterdir()))
    
    # Recount
    final_counts = Counter()
    for lf in (OUTPUT_DIR / "train" / "labels").glob("*.txt"):
        for line in open(lf):
            parts = line.strip().split()
            if parts:
                final_counts[int(parts[0])] += 1
    
    vals = sorted(final_counts.values())
    still_weak = sum(1 for v in vals if v < TARGET_MIN)
    
    print(f"\nFINAL V7 DATASET:")
    print(f"  Train: {final_train} images")
    print(f"  Val: {final_val} images")
    print(f"  Min annotations/cat: {vals[0]}")
    print(f"  Categories still under {TARGET_MIN}: {still_weak}")
    print(f"  dataset.yaml: {OUTPUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

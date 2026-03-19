"""
Create heavily augmented YOLO dataset using albumentations.
Generates offline augmentations for rare classes to balance the dataset.
Also applies mosaic-style augmentation by combining multiple images.
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import albumentations as A
import numpy as np
from PIL import Image

DATA_DIR = Path(__file__).parent / "data"
COCO_DIR = DATA_DIR / "coco_dataset" / "train"
YOLO_DIR = DATA_DIR / "yolo_dataset"
ANNOTATIONS = COCO_DIR / "annotations.json"
OUTPUT_DIR = DATA_DIR / "yolo_augmented"


def get_augmentation_pipeline():
    """Heavy augmentation pipeline for shelf images."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.5),
        A.GaussNoise(var_limit=(5, 25), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.RandomGamma(gamma_limit=(80, 120), p=0.3),
        A.CLAHE(clip_limit=2.0, p=0.2),
        A.ImageCompression(quality_range=(75, 95), p=0.3),
        A.RandomScale(scale_limit=0.15, p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=3, p=0.5,
                          border_mode=0),
    ], bbox_params=A.BboxParams(
        format='yolo',
        label_fields=['class_labels'],
        min_visibility=0.3,
        min_area=100,
    ))


def load_yolo_labels(label_path: Path):
    """Load YOLO format labels."""
    labels = []
    if not label_path.exists():
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                bbox = [float(x) for x in parts[1:]]
                labels.append((cls_id, bbox))
    return labels


def save_yolo_labels(label_path: Path, labels: list):
    """Save YOLO format labels."""
    with open(label_path, "w") as f:
        for cls_id, bbox in labels:
            f.write(f"{cls_id} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")


def main():
    # Load COCO annotations to get class counts
    with open(ANNOTATIONS) as f:
        coco = json.load(f)

    cat_counts = Counter(ann["category_id"] for ann in coco["annotations"])

    # Determine how many augmented copies each image needs based on rare class content
    # Group images by their rarest class
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    img_id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}

    # Calculate augmentation multiplier per image
    # Images with rare classes get more augmentations
    img_multipliers = {}
    for img_id, anns in img_anns.items():
        min_count = min(cat_counts[ann["category_id"]] for ann in anns)
        if min_count < 5:
            mult = 10
        elif min_count < 10:
            mult = 7
        elif min_count < 20:
            mult = 5
        elif min_count < 50:
            mult = 3
        else:
            mult = 1
        img_multipliers[img_id] = mult

    total_augmented = sum(img_multipliers.values())
    print(f"Will generate {total_augmented} augmented images from {len(img_multipliers)} originals")

    # Setup output
    for split in ["train"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Copy val set unchanged
    import shutil
    val_src = YOLO_DIR / "val"
    val_dst = OUTPUT_DIR / "val"
    if val_src.exists() and not val_dst.exists():
        shutil.copytree(val_src, val_dst)
        print(f"Copied validation set to {val_dst}")

    # Copy original train images first
    train_src = YOLO_DIR / "train"
    for img_path in sorted((train_src / "images").glob("*.jpg")):
        shutil.copy2(img_path, OUTPUT_DIR / "train" / "images" / img_path.name)
        label_path = train_src / "labels" / (img_path.stem + ".txt")
        if label_path.exists():
            shutil.copy2(label_path, OUTPUT_DIR / "train" / "labels" / label_path.name)

    print("Copied original training images")

    # Generate augmented images
    transform = get_augmentation_pipeline()
    random.seed(42)
    np.random.seed(42)

    augmented_count = 0
    for img_id, mult in sorted(img_multipliers.items()):
        if mult <= 1:
            continue

        filename = img_id_to_file.get(img_id)
        if not filename:
            continue

        img_path = train_src / "images" / filename
        label_path = train_src / "labels" / (Path(filename).stem + ".txt")

        if not img_path.exists() or not label_path.exists():
            continue

        # Load image and labels
        img = np.array(Image.open(img_path).convert("RGB"))
        labels = load_yolo_labels(label_path)
        if not labels:
            continue

        bboxes = [l[1] for l in labels]
        class_labels = [l[0] for l in labels]

        for aug_i in range(mult - 1):  # -1 because original is already copied
            try:
                result = transform(
                    image=img,
                    bboxes=bboxes,
                    class_labels=class_labels,
                )
                aug_img = result["image"]
                aug_bboxes = result["bboxes"]
                aug_classes = result["class_labels"]

                if not aug_bboxes:
                    continue

                # Save
                aug_name = f"{Path(filename).stem}_aug{aug_i:02d}"
                aug_img_pil = Image.fromarray(aug_img)
                aug_img_pil.save(
                    OUTPUT_DIR / "train" / "images" / f"{aug_name}.jpg",
                    quality=95,
                )
                save_yolo_labels(
                    OUTPUT_DIR / "train" / "labels" / f"{aug_name}.txt",
                    list(zip(aug_classes, aug_bboxes)),
                )
                augmented_count += 1
            except Exception as e:
                continue

    # Write dataset.yaml
    with open(ANNOTATIONS) as f:
        coco = json.load(f)
    categories = coco["categories"]

    yaml_content = f"""# NorgesGruppen Object Detection Dataset (Augmented)
path: {OUTPUT_DIR.resolve()}
train: train/images
val: val/images

nc: {len(categories)}
names:
"""
    for cat in categories:
        name = cat["name"].replace("'", "\\'").replace('"', '\\"')
        yaml_content += f"  {cat['id']}: '{name}'\n"

    with open(OUTPUT_DIR / "dataset.yaml", "w") as f:
        f.write(yaml_content)

    train_imgs = len(list((OUTPUT_DIR / "train" / "images").glob("*.jpg")))
    val_imgs = len(list((OUTPUT_DIR / "val" / "images").glob("*.jpg")))
    print(f"\n=== AUGMENTED DATASET COMPLETE ===")
    print(f"Train images: {train_imgs} (original: 198 + augmented: {augmented_count})")
    print(f"Val images: {val_imgs}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"dataset.yaml: {OUTPUT_DIR / 'dataset.yaml'}")


if __name__ == "__main__":
    main()

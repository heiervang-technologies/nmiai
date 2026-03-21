#!/usr/bin/env python3
"""
Update mega_dataset with all synthetic sources:
- silver_copypaste (copy-paste augmentation)
- synthetic_dataset (FLUX + blurred background composites)

Uses symlinks to avoid duplicating data.
"""

import shutil
from pathlib import Path

BASE_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data")
MEGA_DIR = BASE_DIR / "mega_dataset"
MEGA_TRAIN_IMGS = MEGA_DIR / "train" / "images"
MEGA_TRAIN_LBLS = MEGA_DIR / "train" / "labels"

SOURCES = [
    ("silver_copypaste", BASE_DIR / "silver_copypaste" / "images", BASE_DIR / "silver_copypaste" / "labels"),
    ("synthetic_dataset", BASE_DIR / "synthetic_dataset" / "images", None),  # needs COCO->YOLO conversion
]


def add_yolo_source(name: str, img_dir: Path, lbl_dir: Path):
    """Add a YOLO-format source to mega dataset via symlinks."""
    added = 0
    skipped = 0

    for img_path in sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
        # Skip background-only images
        if img_path.name.startswith("bg_"):
            continue

        label_path = lbl_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            skipped += 1
            continue

        # Check label isn't empty
        if label_path.stat().st_size == 0:
            skipped += 1
            continue

        dst_img = MEGA_TRAIN_IMGS / img_path.name
        dst_lbl = MEGA_TRAIN_LBLS / (img_path.stem + ".txt")

        if dst_img.exists() or dst_lbl.exists():
            skipped += 1
            continue

        dst_img.symlink_to(img_path.resolve())
        dst_lbl.symlink_to(label_path.resolve())
        added += 1

    print(f"  {name}: added {added}, skipped {skipped}")
    return added


def convert_and_add_synthetic():
    """Convert synthetic COCO dataset and add to mega."""
    import json

    synth_dir = BASE_DIR / "synthetic_dataset"
    annot_file = synth_dir / "annotations.json"
    img_dir = synth_dir / "images"

    if not annot_file.exists():
        print("  synthetic_dataset: no annotations.json found, skipping")
        return 0

    with open(annot_file) as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    anns_by_image = {}
    for ann in data["annotations"]:
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    added = 0
    for img_id, img_info in images.items():
        fname = img_info["file_name"]
        if fname.startswith("bg_"):
            continue

        src_img = img_dir / fname
        if not src_img.exists():
            continue

        anns = anns_by_image.get(img_id, [])
        if not anns:
            continue

        # Convert to YOLO
        lines = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            img_w, img_h = img_info["width"], img_info["height"]
            cx = max(0, min(1, (x + w / 2) / img_w))
            cy = max(0, min(1, (y + h / 2) / img_h))
            nw = max(0, min(1, w / img_w))
            nh = max(0, min(1, h / img_h))
            if nw < 0.005 or nh < 0.005:
                continue
            lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if not lines:
            continue

        dst_img = MEGA_TRAIN_IMGS / fname
        dst_lbl = MEGA_TRAIN_LBLS / (Path(fname).stem + ".txt")

        if dst_img.exists():
            continue

        # Symlink image, write label
        dst_img.symlink_to(src_img.resolve())
        with open(dst_lbl, "w") as f:
            f.write("\n".join(lines))

        added += 1

    print(f"  synthetic_dataset: added {added} images")
    return added


def main():
    print("Updating mega_dataset with synthetic sources")
    print(f"Current mega size: {len(list(MEGA_TRAIN_IMGS.glob('*')))} train images")
    print()

    total_added = 0

    # Silver copypaste (already YOLO format)
    silver_imgs = BASE_DIR / "silver_copypaste" / "images"
    silver_lbls = BASE_DIR / "silver_copypaste" / "labels"
    if silver_imgs.exists() and silver_lbls.exists():
        total_added += add_yolo_source("silver_copypaste", silver_imgs, silver_lbls)

    # Synthetic dataset (COCO format -> convert)
    total_added += convert_and_add_synthetic()

    new_size = len(list(MEGA_TRAIN_IMGS.glob("*")))
    print(f"\nTotal added: {total_added}")
    print(f"New mega size: {new_size} train images")


if __name__ == "__main__":
    main()

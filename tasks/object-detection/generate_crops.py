"""Generate crop images from YOLO dataset annotations for DINOv2 probe training.

Cuts bounding box crops from training images and saves them with category labels.
"""
import json
from pathlib import Path
from PIL import Image

TASK_DIR = Path(__file__).resolve().parent
# Use V5 augmented dataset (biggest dataset we have)
YOLO_DIR = TASK_DIR / "data-creation" / "data" / "yolo_augmented_v5"
COCO_DIR = TASK_DIR / "data-creation" / "data" / "coco_dataset"
OUTPUT_DIR = TASK_DIR / "vlm-approach" / "cached_dataset" / "crops"
OUTPUT_JSON = TASK_DIR / "data-creation" / "data" / "extra_crops" / "all_crops.json"

def main():
    # Load COCO annotations to get bboxes
    coco_file = COCO_DIR / "train" / "annotations.json"
    with open(coco_file) as f:
        coco = json.load(f)

    # Build image_id -> filename mapping
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    img_dir = COCO_DIR / "train" / "images"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    samples = []
    crop_id = 300000  # Start after existing crops
    skipped = 0

    for ann in coco["annotations"]:
        img_file = id_to_file.get(ann["image_id"])
        if not img_file:
            continue
        img_path = img_dir / img_file
        if not img_path.exists():
            continue

        x, y, w, h = ann["bbox"]
        cat_id = ann["category_id"]

        # Skip tiny boxes
        if w < 10 or h < 10:
            skipped += 1
            continue

        crop_path = OUTPUT_DIR / f"{crop_id}.jpg"
        if not crop_path.exists():
            try:
                img = Image.open(img_path)
                x1, y1 = max(0, int(x)), max(0, int(y))
                x2, y2 = min(img.width, int(x + w)), min(img.height, int(y + h))
                if x2 <= x1 or y2 <= y1:
                    skipped += 1
                    continue
                crop = img.crop((x1, y1, x2, y2))
                crop.save(crop_path, quality=90)
            except Exception:
                skipped += 1
                continue

        samples.append({
            "crop_path": str(crop_path),
            "category_id": cat_id,
        })
        crop_id += 1

        if len(samples) % 5000 == 0:
            print(f"  Generated {len(samples)} crops...")

    # Also include existing crops
    existing_json = TASK_DIR / "data-creation" / "data" / "extra_crops" / "clean_combined_samples.json"
    if existing_json.exists():
        existing = json.load(open(existing_json))
        # Only include ones that actually exist
        existing_valid = [s for s in existing if Path(s["crop_path"]).exists()]
        print(f"Existing valid crops: {len(existing_valid)}")
        samples = existing_valid + samples

    with open(OUTPUT_JSON, "w") as f:
        json.dump(samples, f)
    print(f"\nTotal crops: {len(samples)} (skipped {skipped})")
    print(f"Saved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

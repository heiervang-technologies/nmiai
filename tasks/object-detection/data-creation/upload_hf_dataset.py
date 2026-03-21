"""
Upload object detection dataset to HuggingFace with two splits:
- competition: COCO images with annotations (248 images)
- store_photos: Markus's personal store photos + video frames (1118 images, unlabeled)

Uses HF datasets library so images render in the viewer for manual inspection.
"""
import json
from pathlib import Path
from datasets import Dataset, DatasetDict, Features, Value, Image, Sequence
from huggingface_hub import HfApi

DATA_DIR = Path(__file__).parent / "data"
REPO_ID = "marksverdhei/nmiai-object-detection"


def build_competition_split():
    """Build competition split from COCO data with annotations."""
    coco_dir = DATA_DIR / "coco_dataset" / "train"
    ann_path = coco_dir / "annotations.json"
    img_dir = coco_dir / "images"

    with open(ann_path) as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    # Group annotations by image
    from collections import defaultdict
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    img_map = {img["id"]: img for img in coco["images"]}

    records = []
    for img_id, info in sorted(img_map.items()):
        img_path = img_dir / info["file_name"]
        if not img_path.exists():
            continue

        anns = img_anns.get(img_id, [])

        # COCO format annotations
        bboxes = []
        category_ids = []
        category_names = []
        areas = []

        for ann in anns:
            bboxes.append(ann["bbox"])  # [x, y, w, h]
            category_ids.append(ann["category_id"])
            category_names.append(cat_names.get(ann["category_id"], "unknown"))
            areas.append(ann.get("area", 0))

        records.append({
            "image": str(img_path),
            "image_id": img_id,
            "file_name": info["file_name"],
            "width": info["width"],
            "height": info["height"],
            "num_objects": len(anns),
            "bbox": json.dumps(bboxes),
            "category_ids": json.dumps(category_ids),
            "category_names": json.dumps(category_names),
            "areas": json.dumps(areas),
        })

    print(f"Competition split: {len(records)} images")

    ds = Dataset.from_dict({
        "image": [r["image"] for r in records],
        "image_id": [r["image_id"] for r in records],
        "file_name": [r["file_name"] for r in records],
        "width": [r["width"] for r in records],
        "height": [r["height"] for r in records],
        "num_objects": [r["num_objects"] for r in records],
        "bbox": [r["bbox"] for r in records],
        "category_ids": [r["category_ids"] for r in records],
        "category_names": [r["category_names"] for r in records],
        "areas": [r["areas"] for r in records],
    })

    # Cast image column
    ds = ds.cast_column("image", Image())
    return ds


def build_store_photos_split():
    """Build store photos split from Markus's personal photos + video frames."""
    store_dir = DATA_DIR / "store_photos"
    frames_dir = store_dir / "video_frames"

    records = []

    # Direct photos
    for img_path in sorted(store_dir.glob("*.jpg")):
        records.append({
            "image": str(img_path),
            "file_name": img_path.name,
            "source": "phone_photo",
            "labeled": False,
        })

    # Video frames
    if frames_dir.exists():
        for img_path in sorted(frames_dir.glob("*.jpg")):
            records.append({
                "image": str(img_path),
                "file_name": img_path.name,
                "source": "video_frame",
                "labeled": False,
            })

    print(f"Store photos split: {len(records)} images ({sum(1 for r in records if r['source']=='phone_photo')} photos, {sum(1 for r in records if r['source']=='video_frame')} frames)")

    ds = Dataset.from_dict({
        "image": [r["image"] for r in records],
        "file_name": [r["file_name"] for r in records],
        "source": [r["source"] for r in records],
        "labeled": [r["labeled"] for r in records],
    })

    ds = ds.cast_column("image", Image())
    return ds


def main():
    print("Building HuggingFace dataset...")

    competition_ds = build_competition_split()
    store_ds = build_store_photos_split()

    dataset_dict = DatasetDict({
        "competition": competition_ds,
        "store_photos": store_ds,
    })

    print(f"\nDataset summary:")
    print(f"  competition: {len(competition_ds)} images with annotations")
    print(f"  store_photos: {len(store_ds)} images (unlabeled)")

    print(f"\nPushing to {REPO_ID} (private)...")
    dataset_dict.push_to_hub(
        REPO_ID,
        private=True,
    )

    print(f"\nDone! Dataset at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()

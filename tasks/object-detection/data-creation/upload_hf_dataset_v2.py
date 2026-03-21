"""
Upload object detection dataset to HuggingFace with two splits:
- competition: COCO images with annotations (248 images)
- store_photos: Markus's personal store photos + video frames (unlabeled)

Uses parquet + HfApi for compatibility. Images embedded as bytes for viewer rendering.
"""
import json
import io
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image as PILImage

DATA_DIR = Path(__file__).parent / "data"
REPO_ID = "marksverdhei/nmiai-object-detection"
LOCAL_PARQUET = Path(__file__).parent / "hf_upload_tmp"


def image_to_bytes(img_path, max_size=1200):
    """Load image, resize if too large, return JPEG bytes."""
    img = PILImage.open(img_path)
    img = img.convert("RGB")

    # Resize large images for HF viewer
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def build_competition_parquet():
    """Build competition split parquet."""
    coco_dir = DATA_DIR / "coco_dataset" / "train"
    ann_path = coco_dir / "annotations.json"
    img_dir = coco_dir / "images"

    with open(ann_path) as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}

    from collections import defaultdict
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    rows = {"image": [], "image_id": [], "file_name": [], "width": [], "height": [],
            "num_objects": [], "annotations": []}

    for img_id in sorted(img_map.keys()):
        info = img_map[img_id]
        img_path = img_dir / info["file_name"]
        if not img_path.exists():
            continue

        anns = img_anns.get(img_id, [])
        ann_data = []
        for ann in anns:
            ann_data.append({
                "bbox": ann["bbox"],
                "category_id": ann["category_id"],
                "category_name": cat_names.get(ann["category_id"], "unknown"),
                "area": ann.get("area", 0),
            })

        try:
            img_bytes = image_to_bytes(img_path)
        except Exception as e:
            print(f"  Skip {img_path.name}: {e}")
            continue

        rows["image"].append({"bytes": img_bytes, "path": info["file_name"]})
        rows["image_id"].append(img_id)
        rows["file_name"].append(info["file_name"])
        rows["width"].append(info["width"])
        rows["height"].append(info["height"])
        rows["num_objects"].append(len(anns))
        rows["annotations"].append(json.dumps(ann_data))

    print(f"Competition: {len(rows['image'])} images")
    return rows


def build_store_parquet():
    """Build store photos split parquet."""
    store_dir = DATA_DIR / "store_photos"
    frames_dir = store_dir / "video_frames"

    rows = {"image": [], "file_name": [], "source": [], "labeled": []}

    # Direct photos
    for img_path in sorted(store_dir.glob("*.jpg")):
        try:
            img_bytes = image_to_bytes(img_path)
        except Exception as e:
            print(f"  Skip {img_path.name}: {e}")
            continue
        rows["image"].append({"bytes": img_bytes, "path": img_path.name})
        rows["file_name"].append(img_path.name)
        rows["source"].append("phone_photo")
        rows["labeled"].append(False)

    # Video frames
    if frames_dir.exists():
        for img_path in sorted(frames_dir.glob("*.jpg")):
            try:
                img_bytes = image_to_bytes(img_path)
            except Exception as e:
                continue
            rows["image"].append({"bytes": img_bytes, "path": img_path.name})
            rows["file_name"].append(img_path.name)
            rows["source"].append("video_frame")
            rows["labeled"].append(False)

    print(f"Store photos: {len(rows['image'])} images")
    return rows


def main():
    api = HfApi()

    # Create repo if needed
    try:
        create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
        print(f"Repo ready: {REPO_ID}")
    except Exception as e:
        print(f"Repo exists or error: {e}")

    LOCAL_PARQUET.mkdir(exist_ok=True)

    # Build and upload competition split
    print("\nBuilding competition split...")
    comp = build_competition_parquet()

    # Use datasets library for proper image format
    from datasets import Dataset, Features, Value, Image as HFImage

    comp_ds = Dataset.from_dict(comp)
    comp_ds = comp_ds.cast_column("image", HFImage())

    print("Saving competition parquet...")
    comp_path = LOCAL_PARQUET / "competition"
    comp_path.mkdir(exist_ok=True)
    comp_ds.to_parquet(comp_path / "data-00000-of-00001.parquet")

    # Upload
    print("Uploading competition split...")
    api.upload_file(
        path_or_fileobj=str(comp_path / "data-00000-of-00001.parquet"),
        path_in_repo="data/competition-00000-of-00001.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    # Build and upload store photos split
    print("\nBuilding store photos split...")
    store = build_store_parquet()

    store_ds = Dataset.from_dict(store)
    store_ds = store_ds.cast_column("image", HFImage())

    print("Saving store photos parquet...")
    store_path = LOCAL_PARQUET / "store_photos"
    store_path.mkdir(exist_ok=True)
    store_ds.to_parquet(store_path / "data-00000-of-00001.parquet")

    print("Uploading store_photos split...")
    api.upload_file(
        path_or_fileobj=str(store_path / "data-00000-of-00001.parquet"),
        path_in_repo="data/store_photos-00000-of-00001.parquet",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    # Upload README
    readme = f"""---
dataset_info:
  - config_name: competition
    features:
      - name: image
        dtype: image
      - name: image_id
        dtype: int64
      - name: file_name
        dtype: string
      - name: width
        dtype: int64
      - name: height
        dtype: int64
      - name: num_objects
        dtype: int64
      - name: annotations
        dtype: string
    splits:
      - name: competition
        num_examples: {len(comp['image'])}
  - config_name: store_photos
    features:
      - name: image
        dtype: image
      - name: file_name
        dtype: string
      - name: source
        dtype: string
      - name: labeled
        dtype: bool
    splits:
      - name: store_photos
        num_examples: {len(store['image'])}
configs:
  - config_name: competition
    data_files:
      - split: competition
        path: data/competition-*.parquet
  - config_name: store_photos
    data_files:
      - split: store_photos
        path: data/store_photos-*.parquet
license: other
task_categories:
  - object-detection
---

# NM i AI 2026 - Object Detection Dataset

Private dataset for The Vector Space team.

## Splits

- **competition**: {len(comp['image'])} COCO images with full annotations (356 Norwegian grocery categories)
- **store_photos**: {len(store['image'])} personal store photos + video frames (unlabeled)
"""

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    print(f"\nDone! Dataset at: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()

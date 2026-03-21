"""
Upload to HF using raw parquet + HfApi (no datasets library serialization).
Images embedded as bytes in parquet for HF viewer rendering.
"""
import json
import io
from pathlib import Path
from collections import defaultdict
from huggingface_hub import HfApi, create_repo
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image as PILImage

DATA_DIR = Path(__file__).parent / "data"
REPO_ID = "marksverdhei/nmiai-object-detection"
TMP = Path(__file__).parent / "hf_parquet_tmp"


def image_to_bytes(img_path, max_size=1200):
    img = PILImage.open(img_path).convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), PILImage.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def build_competition():
    coco_dir = DATA_DIR / "coco_dataset" / "train"
    with open(coco_dir / "annotations.json") as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

    # Build columns
    image_bytes_col = []
    image_path_col = []
    image_id_col = []
    file_name_col = []
    width_col = []
    height_col = []
    num_objects_col = []
    annotations_col = []

    for img_id in sorted(img_map.keys()):
        info = img_map[img_id]
        img_path = coco_dir / "images" / info["file_name"]
        if not img_path.exists():
            continue

        try:
            img_b = image_to_bytes(img_path)
        except Exception as e:
            print(f"  Skip {info['file_name']}: {e}")
            continue

        anns = img_anns.get(img_id, [])
        ann_data = [{"bbox": a["bbox"], "category_id": a["category_id"],
                     "category_name": cat_names.get(a["category_id"], "?"),
                     "area": a.get("area", 0)} for a in anns]

        image_bytes_col.append(img_b)
        image_path_col.append(info["file_name"])
        image_id_col.append(img_id)
        file_name_col.append(info["file_name"])
        width_col.append(info["width"])
        height_col.append(info["height"])
        num_objects_col.append(len(anns))
        annotations_col.append(json.dumps(ann_data))

    # Build arrow table with image struct {bytes, path} for HF viewer
    image_struct = pa.StructArray.from_arrays(
        [pa.array(image_bytes_col, type=pa.binary()), pa.array(image_path_col, type=pa.string())],
        names=["bytes", "path"]
    )

    table = pa.table({
        "image": image_struct,
        "image_id": pa.array(image_id_col, type=pa.int64()),
        "file_name": pa.array(file_name_col, type=pa.string()),
        "width": pa.array(width_col, type=pa.int64()),
        "height": pa.array(height_col, type=pa.int64()),
        "num_objects": pa.array(num_objects_col, type=pa.int64()),
        "annotations": pa.array(annotations_col, type=pa.string()),
    })

    print(f"Competition: {len(image_bytes_col)} images")
    return table, len(image_bytes_col)


def build_store():
    store_dir = DATA_DIR / "store_photos"
    frames_dir = store_dir / "video_frames"

    image_bytes_col = []
    image_path_col = []
    file_name_col = []
    source_col = []
    labeled_col = []

    for img_path in sorted(store_dir.glob("*.jpg")):
        try:
            img_b = image_to_bytes(img_path)
        except:
            continue
        image_bytes_col.append(img_b)
        image_path_col.append(img_path.name)
        file_name_col.append(img_path.name)
        source_col.append("phone_photo")
        labeled_col.append(False)

    if frames_dir.exists():
        for img_path in sorted(frames_dir.glob("*.jpg")):
            try:
                img_b = image_to_bytes(img_path)
            except:
                continue
            image_bytes_col.append(img_b)
            image_path_col.append(img_path.name)
            file_name_col.append(img_path.name)
            source_col.append("video_frame")
            labeled_col.append(False)

    image_struct = pa.StructArray.from_arrays(
        [pa.array(image_bytes_col, type=pa.binary()), pa.array(image_path_col, type=pa.string())],
        names=["bytes", "path"]
    )

    table = pa.table({
        "image": image_struct,
        "file_name": pa.array(file_name_col, type=pa.string()),
        "source": pa.array(source_col, type=pa.string()),
        "labeled": pa.array(labeled_col, type=pa.bool_()),
    })

    photos = sum(1 for s in source_col if s == "phone_photo")
    frames = sum(1 for s in source_col if s == "video_frame")
    print(f"Store: {len(image_bytes_col)} images ({photos} photos, {frames} frames)")
    return table, len(image_bytes_col)


def main():
    api = HfApi()
    create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    TMP.mkdir(exist_ok=True)

    # Competition split
    print("Building competition split...")
    comp_table, comp_n = build_competition()
    comp_path = TMP / "competition-00000-of-00001.parquet"
    pq.write_table(comp_table, comp_path)
    print(f"Uploading competition ({comp_path.stat().st_size // 1024 // 1024}MB)...")
    api.upload_file(
        path_or_fileobj=str(comp_path),
        path_in_repo="data/competition-00000-of-00001.parquet",
        repo_id=REPO_ID, repo_type="dataset",
    )

    # Store split
    print("\nBuilding store photos split...")
    store_table, store_n = build_store()
    store_path = TMP / "store_photos-00000-of-00001.parquet"
    pq.write_table(store_table, store_path)
    print(f"Uploading store_photos ({store_path.stat().st_size // 1024 // 1024}MB)...")
    api.upload_file(
        path_or_fileobj=str(store_path),
        path_in_repo="data/store_photos-00000-of-00001.parquet",
        repo_id=REPO_ID, repo_type="dataset",
    )

    # README
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
        num_examples: {comp_n}
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
        num_examples: {store_n}
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

- **competition**: {comp_n} COCO images with full annotations (356 Norwegian grocery categories)
- **store_photos**: {store_n} personal store photos + video frames (unlabeled)
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID, repo_type="dataset",
    )

    print(f"\nDone! https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()

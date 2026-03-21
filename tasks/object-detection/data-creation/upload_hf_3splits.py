"""
Upload to HF with THREE splits:
- competition: COCO images with annotations
- store_photos: Markus's direct phone photos (39 images, higher quality)
- video_frames: Extracted video frames (1079, need extensive correction)
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


def build_image_parquet(image_paths, extra_cols=None):
    """Build parquet from list of image paths with optional extra columns."""
    image_bytes_col = []
    image_path_col = []
    file_name_col = []

    for img_path in image_paths:
        try:
            img_b = image_to_bytes(img_path)
        except Exception as e:
            print(f"  Skip {img_path.name}: {e}")
            continue
        image_bytes_col.append(img_b)
        image_path_col.append(img_path.name)
        file_name_col.append(img_path.name)

    image_struct = pa.StructArray.from_arrays(
        [pa.array(image_bytes_col, type=pa.binary()), pa.array(image_path_col, type=pa.string())],
        names=["bytes", "path"]
    )

    cols = {"image": image_struct, "file_name": pa.array(file_name_col, type=pa.string())}

    if extra_cols:
        for k, v in extra_cols.items():
            # Trim to actual successful count
            cols[k] = v[:len(file_name_col)]

    return pa.table(cols), len(image_bytes_col)


def build_competition():
    coco_dir = DATA_DIR / "coco_dataset" / "train"
    with open(coco_dir / "annotations.json") as f:
        coco = json.load(f)

    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    img_map = {img["id"]: img for img in coco["images"]}
    img_anns = defaultdict(list)
    for ann in coco["annotations"]:
        img_anns[ann["image_id"]].append(ann)

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
        except:
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


def main():
    api = HfApi()
    create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    TMP.mkdir(exist_ok=True)

    store_dir = DATA_DIR / "store_photos"
    frames_dir = store_dir / "video_frames"

    # 1. Competition
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

    # 2. Store photos (direct phone photos only)
    print("\nBuilding store_photos split (direct photos only)...")
    photo_paths = sorted(store_dir.glob("*.jpg"))
    photo_table, photo_n = build_image_parquet(photo_paths)
    photo_path = TMP / "store_photos-00000-of-00001.parquet"
    pq.write_table(photo_table, photo_path)
    print(f"Uploading store_photos ({photo_path.stat().st_size // 1024 // 1024}MB)...")
    api.upload_file(
        path_or_fileobj=str(photo_path),
        path_in_repo="data/store_photos-00000-of-00001.parquet",
        repo_id=REPO_ID, repo_type="dataset",
    )

    # 3. Video frames (separate - needs correction)
    print("\nBuilding video_frames split...")
    frame_paths = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
    frame_table, frame_n = build_image_parquet(frame_paths)
    frame_path = TMP / "video_frames-00000-of-00001.parquet"
    pq.write_table(frame_table, frame_path)
    print(f"Uploading video_frames ({frame_path.stat().st_size // 1024 // 1024}MB)...")
    api.upload_file(
        path_or_fileobj=str(frame_path),
        path_in_repo="data/video_frames-00000-of-00001.parquet",
        repo_id=REPO_ID, repo_type="dataset",
    )

    # Delete old combined store_photos parquet if it exists
    try:
        api.delete_file(
            path_in_repo="data/store_photos-00000-of-00001.parquet",
            repo_id=REPO_ID, repo_type="dataset",
        )
    except:
        pass

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
    splits:
      - name: store_photos
        num_examples: {photo_n}
  - config_name: video_frames
    features:
      - name: image
        dtype: image
      - name: file_name
        dtype: string
    splits:
      - name: video_frames
        num_examples: {frame_n}
configs:
  - config_name: competition
    data_files:
      - split: competition
        path: data/competition-*.parquet
  - config_name: store_photos
    data_files:
      - split: store_photos
        path: data/store_photos-*.parquet
  - config_name: video_frames
    data_files:
      - split: video_frames
        path: data/video_frames-*.parquet
license: other
task_categories:
  - object-detection
---

# NM i AI 2026 - Object Detection Dataset

Private dataset for The Vector Space team.

## Splits

- **competition**: {comp_n} COCO images with full annotations (356 Norwegian grocery categories)
- **store_photos**: {photo_n} direct phone photos from store visits (higher quality, ready for labeling)
- **video_frames**: {frame_n} extracted video frames (need extensive correction before use)
"""
    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID, repo_type="dataset",
    )

    print(f"\nDone! https://huggingface.co/datasets/{REPO_ID}")
    print(f"  competition: {comp_n} images")
    print(f"  store_photos: {photo_n} images")
    print(f"  video_frames: {frame_n} images")


if __name__ == "__main__":
    main()

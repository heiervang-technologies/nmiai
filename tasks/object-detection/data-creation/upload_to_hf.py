"""
Upload all object detection data to a private HuggingFace dataset.

Structure on HF:
  heiertech/nmiai-object-detection/
  ├── competition_data/
  │   ├── coco_dataset/          # 248 training images + annotations.json
  │   └── product_images/        # 345 product reference folders
  ├── yolo_datasets/
  │   ├── yolo_v1/               # Original 80/20 split
  │   └── yolo_v3/               # Best dataset (aug + PL)
  ├── store_photos/
  │   ├── photos/                # 39 phone photos
  │   └── video_frames/          # 1079 extracted frames
  ├── external/
  │   └── skus_on_shelves_pl/    # 27K Polish shelf images + annotations
  ├── embeddings/
  │   ├── ref_embeddings.pth
  │   ├── reference_embeddings.npz
  │   ├── category_embeddings.npz
  │   └── *.json mappings
  └── README.md
"""
import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "heiertech/nmiai-object-detection"
DATA_DIR = Path(__file__).parent / "data"
OUTPUTS_DIR = Path(__file__).parent / "outputs"


def create_readme():
    return """---
license: mit
task_categories:
- object-detection
language:
- 'no'
tags:
- grocery
- shelf-detection
- norgesgruppen
- nmiai-2026
pretty_name: NM i AI 2026 - Object Detection Dataset
private: true
---

# NM i AI 2026 - Object Detection Dataset

Competition data and supplementary datasets for the NorgesGruppen grocery product detection task.

## Contents

### competition_data/
- **coco_dataset/**: 248 training images with 22,731 COCO-format annotations across 356 product categories
- **product_images/**: 345 product reference folders with multi-angle studio photos (by barcode/EAN)

### yolo_datasets/
- **yolo_v1/**: Original 80/20 split (198 train, 50 val), dataset.yaml included
- **yolo_v3/**: Best augmented dataset (2,565 train, 50 val) - includes albumentations augmentation + Polish shelf data

### store_photos/
- **photos/**: 39 high-res photos from Norwegian Kiwi store (4096x3072)
- **video_frames/**: 1,079 keyframes extracted from 4 store walkthrough videos

### external/
- **skus_on_shelves_pl/**: 27,244 Polish shelf images with 2M+ COCO annotations (CC-BY-4.0)

### embeddings/
- **ref_embeddings.pth**: DINOv2 ViT-S/14 embeddings for 344 products + 356 categories
- **barcode_category_mapping.json**: Visual similarity mapping between product barcodes and category IDs
- **category_index.json**: Category ID to name mapping
- **pl_to_ng_category_mapping.json**: Polish to Norwegian category mapping

## Key Stats

| Dataset | Images | Annotations | Categories |
|---------|--------|-------------|------------|
| Competition (COCO) | 248 | 22,731 | 356 |
| YOLO v1 | 198+50 | 17,924+4,807 | 356 |
| YOLO v3 (augmented) | 2,565+50 | 176,211 | 356 |
| Store photos | 39+1,079 | unlabeled | - |
| SKUs on Shelves PL | 27,244 | 2,054,882 | 7,942 |

## Usage

```python
from huggingface_hub import snapshot_download
path = snapshot_download("heiertech/nmiai-object-detection", repo_type="dataset")
```
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pl", action="store_true", help="Skip PL dataset (11GB)")
    parser.add_argument("--skip-video-frames", action="store_true", help="Skip video frames (577MB)")
    parser.add_argument("--dry-run", action="store_true", help="Just print what would be uploaded")
    args = parser.parse_args()

    api = HfApi()

    # Create repo if needed
    print(f"Creating/verifying repo: {REPO_ID}")
    create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)

    # Upload plan
    uploads = []

    # 1. Competition data - COCO dataset
    coco_dir = DATA_DIR / "coco_dataset"
    if coco_dir.exists():
        uploads.append((coco_dir, "competition_data/coco_dataset"))

    # 2. Competition data - Product images
    prod_dir = DATA_DIR / "product_images"
    if prod_dir.exists():
        uploads.append((prod_dir, "competition_data/product_images"))

    # 3. YOLO v1 (original split)
    yolo_v1 = DATA_DIR / "yolo_dataset"
    if yolo_v1.exists():
        uploads.append((yolo_v1, "yolo_datasets/yolo_v1"))

    # 4. YOLO v3 (best augmented) - need to resolve symlinks
    yolo_v3 = DATA_DIR / "yolo_augmented_v3"
    if yolo_v3.exists():
        uploads.append((yolo_v3, "yolo_datasets/yolo_v3"))

    # 5. Store photos
    store_dir = DATA_DIR / "store_photos"
    if store_dir.exists():
        # Photos
        for f in sorted(store_dir.glob("*.jpg")):
            uploads.append((f, f"store_photos/photos/{f.name}"))
        for f in sorted(store_dir.glob("*.mp4")):
            uploads.append((f, f"store_photos/videos/{f.name}"))
        # Video frames
        if not args.skip_video_frames:
            vf_dir = store_dir / "video_frames"
            if vf_dir.exists():
                uploads.append((vf_dir, "store_photos/video_frames"))

    # 6. External - PL dataset
    if not args.skip_pl:
        pl_dir = DATA_DIR / "external" / "skus_on_shelves_pl" / "extracted"
        if pl_dir.exists():
            uploads.append((pl_dir, "external/skus_on_shelves_pl"))

    # 7. Embeddings and mappings
    if OUTPUTS_DIR.exists():
        uploads.append((OUTPUTS_DIR, "embeddings"))
    pth_file = DATA_DIR / "ref_embeddings.pth"
    if pth_file.exists():
        uploads.append((pth_file, "embeddings/ref_embeddings.pth"))

    # 8. Dataset analysis
    analysis = Path(__file__).parent / "dataset_analysis.json"
    if analysis.exists():
        uploads.append((analysis, "dataset_analysis.json"))

    print(f"\nUpload plan: {len(uploads)} items")
    for src, dest in uploads:
        if src.is_dir():
            count = sum(1 for _ in src.rglob("*") if _.is_file())
            size = sum(f.stat().st_size for f in src.rglob("*") if f.is_file()) / (1024*1024)
            print(f"  {dest}/ ({count} files, {size:.0f} MB)")
        else:
            size = src.stat().st_size / (1024*1024)
            print(f"  {dest} ({size:.1f} MB)")

    if args.dry_run:
        print("\nDry run - nothing uploaded.")
        return

    # Upload README first
    print("\nUploading README...")
    api.upload_file(
        path_or_fileobj=create_readme().encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    # Upload each item
    for i, (src, dest) in enumerate(uploads):
        print(f"\n[{i+1}/{len(uploads)}] Uploading {dest}...")
        try:
            if src.is_dir():
                api.upload_folder(
                    folder_path=str(src),
                    path_in_repo=dest,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    ignore_patterns=["*.pyc", "__pycache__", ".DS_Store"],
                )
            else:
                api.upload_file(
                    path_or_fileobj=str(src),
                    path_in_repo=dest,
                    repo_id=REPO_ID,
                    repo_type="dataset",
                )
            print(f"  Done")
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\n=== UPLOAD COMPLETE ===")
    print(f"Dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()

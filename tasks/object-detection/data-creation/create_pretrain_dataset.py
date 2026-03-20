"""
Create a private HuggingFace dataset card pointing to standard CV pre-training datasets.

These datasets are all available via ultralytics auto-download or HuggingFace:
- Objects365 (already have)
- Open Images V7
- LVIS
- MS COCO 2017
- PASCAL VOC

We don't need to re-host the data - just create YAML configs and a dataset card
that references the official sources.
"""
import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo

REPO_ID = "heiertech/cv-pretrain-datasets"


def create_readme():
    return """---
license: mit
task_categories:
- object-detection
tags:
- pre-training
- object-detection
- coco
- voc
- lvis
- open-images
- objects365
pretty_name: CV Pre-training Dataset Collection
private: true
---

# CV Pre-training Dataset Collection

Standard computer vision datasets for object detection pre-training.
All datasets auto-download via ultralytics or torchvision.

## Datasets

### 1. Objects365 (v2)
- **Size**: 2M images, 365 categories, 30M+ bounding boxes
- **Ultralytics**: `data="Objects365.yaml"`
- **Download**: Auto via ultralytics
- **Use**: Primary pre-training dataset for dense object detection

### 2. MS COCO 2017
- **Size**: 118K train, 5K val, 80 categories
- **Ultralytics**: `data="coco.yaml"`
- **Download**: Auto via ultralytics
- **Use**: Standard detection benchmark, excellent for general pre-training

### 3. Open Images V7
- **Size**: 1.9M images, 600 categories, 16M bounding boxes
- **Ultralytics**: `data="open-images-v7.yaml"`
- **Download**: Auto via ultralytics (uses fiftyone)
- **Use**: Large-scale diverse detection, many grocery-relevant categories

### 4. LVIS v1
- **Size**: 164K images, 1203 categories (long-tail distribution)
- **Ultralytics**: `data="lvis-v1.yaml"`  (uses COCO images)
- **Download**: Auto via ultralytics
- **Use**: Long-tail category distribution, good for rare class handling

### 5. PASCAL VOC (2007+2012)
- **Size**: 16.5K train, 4.9K val, 20 categories
- **Ultralytics**: `data="VOC.yaml"`
- **Download**: Auto via ultralytics
- **Use**: Classic benchmark, quick fine-tuning baseline

### 6. SKU-110K (Retail Shelf Detection)
- **Size**: 11.7K images, 1.7M annotations, 1 class (object)
- **Ultralytics**: `data="SKU-110K.yaml"`
- **Download**: Auto via ultralytics
- **Use**: Dense shelf detection pre-training (directly relevant!)

## Usage with Ultralytics YOLO

```python
from ultralytics import YOLO

model = YOLO("yolo11x.pt")

# Pre-train on Objects365
model.train(data="Objects365.yaml", epochs=10, imgsz=640)

# Fine-tune on COCO
model.train(data="coco.yaml", epochs=20, imgsz=640)

# Fine-tune on our competition data
model.train(data="path/to/yolo_v3/dataset.yaml", epochs=100, imgsz=1280)
```

## YAML Configs

All YAML configs are included in this repo for reference,
but ultralytics will auto-download datasets when you reference them by name.
"""


def create_yaml_configs():
    """Create YAML config files for each dataset."""
    configs = {}

    configs["objects365.yaml"] = """# Objects365 Dataset
# Auto-downloaded by ultralytics
# https://docs.ultralytics.com/datasets/detect/objects365/
path: ../datasets/Objects365
train: images/train
val: images/val

nc: 365
download: https://docs.ultralytics.com/datasets/detect/objects365/
"""

    configs["coco2017.yaml"] = """# MS COCO 2017 Dataset
# Auto-downloaded by ultralytics
# https://docs.ultralytics.com/datasets/detect/coco/
path: ../datasets/coco
train: images/train2017
val: images/val2017
test: images/test2017

nc: 80
download: https://docs.ultralytics.com/datasets/detect/coco/
"""

    configs["open-images-v7.yaml"] = """# Open Images V7 Dataset
# Auto-downloaded by ultralytics (uses fiftyone)
# https://docs.ultralytics.com/datasets/detect/open-images-v7/
path: ../datasets/open-images-v7
train: images/train
val: images/validation

nc: 600
download: https://docs.ultralytics.com/datasets/detect/open-images-v7/
"""

    configs["lvis-v1.yaml"] = """# LVIS v1 Dataset (uses COCO images)
# Auto-downloaded by ultralytics
# https://docs.ultralytics.com/datasets/detect/lvis/
path: ../datasets/coco
train: images/train2017
val: images/val2017

nc: 1203
download: https://docs.ultralytics.com/datasets/detect/lvis/
"""

    configs["pascal-voc.yaml"] = """# PASCAL VOC 2007+2012 Dataset
# Auto-downloaded by ultralytics
# https://docs.ultralytics.com/datasets/detect/voc/
path: ../datasets/VOC
train: images/train
val: images/val

nc: 20
download: https://docs.ultralytics.com/datasets/detect/voc/
"""

    configs["sku-110k.yaml"] = """# SKU-110K Dense Shelf Detection Dataset
# Auto-downloaded by ultralytics
# https://docs.ultralytics.com/datasets/detect/sku-110k/
path: ../datasets/SKU-110K
train: train/images
val: val/images
test: test/images

nc: 1
names:
  0: object
download: https://docs.ultralytics.com/datasets/detect/sku-110k/
"""

    return configs


def main():
    api = HfApi()

    print(f"Creating repo: {REPO_ID}")
    create_repo(REPO_ID, repo_type="dataset", private=True, exist_ok=True)

    # Upload README
    print("Uploading README...")
    api.upload_file(
        path_or_fileobj=create_readme().encode(),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )

    # Upload YAML configs
    configs = create_yaml_configs()
    for name, content in configs.items():
        print(f"Uploading {name}...")
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo=f"configs/{name}",
            repo_id=REPO_ID,
            repo_type="dataset",
        )

    print(f"\n=== DONE ===")
    print(f"Dataset: https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()

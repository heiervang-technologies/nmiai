"""
Build YOLO V4 dataset:
1. V3 augmented data (2565 train images with labels)
2. Pseudo-labeled store photos (39 images)
3. Pseudo-labeled video frames (1079 frames)
4. Original COCO 248 images (already in V3)

Pseudo-labeling uses the best YOLOv8x model with high confidence threshold.
"""
import json
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO

DATA_DIR = Path(__file__).parent / "data"
V3_DIR = DATA_DIR / "yolo_augmented_v3"
STORE_PHOTOS = DATA_DIR / "store_photos"
V4_DIR = DATA_DIR / "yolo_augmented_v4"
COCO_ANN = DATA_DIR / "coco_dataset" / "train" / "annotations.json"

BEST_MODEL = Path("/home/me/ht/nmiai/tasks/object-detection/yolo-approach/runs/yolov8x_v3_1280/weights/best.pt")
CONF_THRESHOLD = 0.5  # High confidence only for pseudo-labels
IOU_THRESHOLD = 0.5
IMGSZ = 1280


def pseudo_label_images(model, image_paths: list[Path], output_images_dir: Path, output_labels_dir: Path, prefix: str = ""):
    """Run YOLO inference and save pseudo-labels in YOLO format."""
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_detections = 0

    for img_path in image_paths:
        # Run inference
        results = model.predict(
            str(img_path),
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            imgsz=IMGSZ,
            verbose=False,
        )

        if not results or len(results[0].boxes) == 0:
            continue

        result = results[0]
        boxes = result.boxes

        # Get image dimensions
        img_h, img_w = result.orig_shape

        # Convert to YOLO format
        labels = []
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            # xyxy to xywh normalized
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            # Clamp
            cx = max(0, min(1, cx))
            cy = max(0, min(1, cy))
            w = max(0, min(1, w))
            h = max(0, min(1, h))
            labels.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if not labels:
            continue

        # Save image (symlink to save space)
        out_name = f"{prefix}{img_path.stem}.jpg"
        dst_img = output_images_dir / out_name
        if not dst_img.exists():
            dst_img.symlink_to(img_path.resolve())

        # Save labels
        label_path = output_labels_dir / f"{prefix}{img_path.stem}.txt"
        label_path.write_text("\n".join(labels) + "\n")

        total_images += 1
        total_detections += len(labels)

    return total_images, total_detections


def main():
    print("=== Building YOLO V4 Dataset ===\n")

    # Create output dirs
    for split in ["train", "val"]:
        (V4_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (V4_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    # Step 1: Copy V3 data (symlinks)
    print("[1/4] Linking V3 train data...")
    v3_train_count = 0
    for img_path in (V3_DIR / "train" / "images").iterdir():
        dst = V4_DIR / "train" / "images" / img_path.name
        if not dst.exists():
            real = img_path.resolve()
            dst.symlink_to(real)
        label_src = V3_DIR / "train" / "labels" / (img_path.stem + ".txt")
        label_dst = V4_DIR / "train" / "labels" / (img_path.stem + ".txt")
        if label_src.exists() and not label_dst.exists():
            shutil.copy2(label_src, label_dst)
        v3_train_count += 1
    print(f"  Linked {v3_train_count} V3 train images")

    print("[2/4] Linking V3 val data...")
    v3_val_count = 0
    for img_path in (V3_DIR / "val" / "images").iterdir():
        dst = V4_DIR / "val" / "images" / img_path.name
        if not dst.exists():
            real = img_path.resolve()
            dst.symlink_to(real)
        label_src = V3_DIR / "val" / "labels" / (img_path.stem + ".txt")
        label_dst = V4_DIR / "val" / "labels" / (img_path.stem + ".txt")
        if label_src.exists() and not label_dst.exists():
            shutil.copy2(label_src, label_dst)
        v3_val_count += 1
    print(f"  Linked {v3_val_count} V3 val images")

    # Step 2: Pseudo-label store photos
    print("\n[3/4] Pseudo-labeling store photos...")
    print(f"  Loading model: {BEST_MODEL.name}")
    model = YOLO(str(BEST_MODEL))

    store_photos = sorted(STORE_PHOTOS.glob("*.jpg"))
    print(f"  {len(store_photos)} store photos to label")
    sp_imgs, sp_dets = pseudo_label_images(
        model, store_photos,
        V4_DIR / "train" / "images",
        V4_DIR / "train" / "labels",
        prefix="store_",
    )
    print(f"  Pseudo-labeled: {sp_imgs} images, {sp_dets} detections")

    # Step 3: Pseudo-label video frames
    print("\n[4/4] Pseudo-labeling video frames...")
    video_frames = sorted((STORE_PHOTOS / "video_frames").glob("*.jpg"))
    print(f"  {len(video_frames)} video frames to label")
    vf_imgs, vf_dets = pseudo_label_images(
        model, video_frames,
        V4_DIR / "train" / "images",
        V4_DIR / "train" / "labels",
        prefix="vframe_",
    )
    print(f"  Pseudo-labeled: {vf_imgs} images, {vf_dets} detections")

    # Step 4: Write dataset.yaml
    with open(COCO_ANN) as f:
        coco = json.load(f)
    names = {c["id"]: c["name"] for c in coco["categories"]}

    yaml_data = {
        "path": str(V4_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(names),
        "names": names,
    }
    yaml_path = V4_DIR / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write("# NorgesGruppen OD - V4 (V3 + pseudo-labeled store photos + video frames)\n")
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    # Summary
    total_train = len(list((V4_DIR / "train" / "images").iterdir()))
    total_val = len(list((V4_DIR / "val" / "images").iterdir()))

    print(f"\n=== YOLO V4 DATASET COMPLETE ===")
    print(f"Output: {V4_DIR}")
    print(f"Train: {total_train} images")
    print(f"  - V3 base: {v3_train_count}")
    print(f"  - Store photos (pseudo): {sp_imgs} ({sp_dets} detections)")
    print(f"  - Video frames (pseudo): {vf_imgs} ({vf_dets} detections)")
    print(f"Val: {total_val} images (unchanged)")
    print(f"dataset.yaml: {yaml_path}")
    print(f"\nConf threshold for pseudo-labels: {CONF_THRESHOLD}")


if __name__ == "__main__":
    main()

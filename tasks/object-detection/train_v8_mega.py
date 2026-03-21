#!/usr/bin/env python3
"""
V8 MEGA DATASET Training - Optimized for 6,440+ images.

Key changes from V3:
- 2.5x more data -> can train longer with less overfitting
- Reduced augmentation (real data diversity replaces synthetic aug)
- Lower copy_paste since silver already includes copy-paste augmented images
- Higher patience (data is more diverse, convergence may be slower)
- cosine LR schedule via lrf
- max_det=300 for dense shelf scenes (up to 235 products per image)

Run on TITAN GPU 1:
  CUDA_VISIBLE_DEVICES=1 python train_v8_mega.py
"""
from ultralytics import YOLO
from pathlib import Path


def main():
    # Fresh start from COCO pretrained
    model = YOLO("yolov8x.pt")

    data_yaml = str(Path(__file__).parent / "data-creation" / "data" / "mega_dataset" / "dataset.yaml")

    results = model.train(
        data=data_yaml,
        epochs=120,
        imgsz=1280,
        batch=4,  # 3090 24GB with imgsz=1280
        optimizer="AdamW",
        lr0=0.0008,  # Slightly lower LR for bigger dataset
        lrf=0.01,    # Cosine decay to 1% of lr0
        warmup_epochs=5,
        weight_decay=0.0005,
        patience=30,  # More patience - diverse data needs time

        # Reduced augmentation - real data diversity replaces synthetic
        mosaic=0.8,      # Down from 1.0 - we have real diverse images now
        mixup=0.05,      # Down from 0.15 - less needed with 6k images
        copy_paste=0.1,  # Down from 0.3 - silver already has copy-paste
        degrees=3.0,     # Down from 5.0 - shelf products are upright
        translate=0.1,
        scale=0.4,       # Down from 0.5
        fliplr=0.5,
        flipud=0.0,      # Shelves don't flip vertically
        hsv_h=0.015,
        hsv_s=0.5,       # Down from 0.7 - preserve product colors
        hsv_v=0.3,       # Down from 0.4
        erasing=0.15,    # Down from 0.3

        # Detection
        iou=0.5,
        max_det=300,     # Dense shelves

        # Performance
        workers=8,
        device=0,
        amp=True,

        # Project
        project=str(Path(__file__).parent / "yolo-approach" / "runs"),
        name="yolov8x_v8_mega",
        exist_ok=True,

        # Saving
        save=True,
        save_period=20,
        plots=True,
        val=True,

        # Close mosaic at epoch 100 for fine-tuning
        close_mosaic=20,
    )

    print("V8 MEGA Training complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()

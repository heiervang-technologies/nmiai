"""
Train YOLO11x on the augmented v3 dataset.

Usage (on titan):
    source ~/yolo_env/bin/activate
    python train_yolo11x.py
"""

from ultralytics import YOLO

DATA_YAML = "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/yolo_augmented_v3/dataset.yaml"

model = YOLO("yolo11x.pt")

model.train(
    data=DATA_YAML,
    epochs=200,
    imgsz=1280,
    batch=4,
    device=0,
    workers=8,
    project="runs/yolo11",
    name="yolo11x-v3",
    patience=30,
    save=True,
    save_period=20,
    val=True,
    plots=True,
    lr0=0.01,
    lrf=0.01,
    optimizer="SGD",
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    cos_lr=True,
    close_mosaic=10,
    mosaic=1.0,
    mixup=0.1,
    copy_paste=0.1,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    exist_ok=True,
)

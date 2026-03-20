"""
Train RT-DETR-L on the augmented v3 dataset.
Fixed: ensures classification head is properly rebuilt for 356 classes.

Usage (on titan):
    source ~/yolo_env/bin/activate
    python train_rtdetr.py
"""

from ultralytics import RTDETR

DATA_YAML = "/home/me/ht/nmiai/tasks/object-detection/data-creation/data/yolo_augmented_v3/dataset.yaml"

# Load from config YAML (not .pt) to build fresh architecture,
# then transfer pretrained backbone weights. This ensures the
# classification head is properly initialized for 356 classes.
model = RTDETR("rtdetr-l.yaml").load("rtdetr-l.pt")

# Verify all parameters are trainable
frozen = sum(1 for p in model.model.parameters() if not p.requires_grad)
total = sum(1 for p in model.model.parameters())
print(f"Parameters: {total} total, {frozen} frozen")

# Verify nc
decoder = model.model.model[28]
print(f"Decoder nc: {decoder.nc}")
print(f"Score head shape: {decoder.dec_score_head[0].weight.shape}")

model.train(
    data=DATA_YAML,
    epochs=120,
    imgsz=1280,
    batch=4,
    device=0,
    workers=8,
    project="runs/rtdetr",
    name="rtdetr-l-v3-fixed",
    patience=20,
    save=True,
    save_period=20,
    val=True,
    plots=True,
    lr0=0.0001,
    lrf=0.01,
    optimizer="AdamW",
    weight_decay=0.0001,
    warmup_epochs=5,
    cos_lr=True,
    close_mosaic=0,
    exist_ok=True,
)

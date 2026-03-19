"""Train YOLOv8x on v3 augmented dataset (2565 images incl. Polish shelf data)."""
from ultralytics import YOLO
from pathlib import Path


def main():
    # Start from COCO pretrained (not from current best - fresh start on bigger data)
    model = YOLO("yolov8x.pt")

    results = model.train(
        data="/home/me/ht/nmiai/tasks/object-detection/data-creation/data/yolo_augmented_v3/dataset.yaml",
        epochs=150,
        imgsz=1280,
        batch=4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.0005,
        patience=40,

        # Augmentation - slightly less aggressive since we have more data
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.2,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.2,

        # Detection
        max_det=300,

        # Performance
        workers=8,
        device=0,
        amp=True,
        project=str(Path(__file__).parent / "runs"),
        name="yolov8x_v3_1280",
        exist_ok=True,

        # Save
        save=True,
        save_period=25,
        plots=True,
        val=True,
    )

    print("V3 Training complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()

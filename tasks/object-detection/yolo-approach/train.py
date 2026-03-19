"""Train YOLOv8x on the NorgesGruppen shelf detection dataset."""
from ultralytics import YOLO
from pathlib import Path

def main():
    model = YOLO("yolov8x.pt")

    results = model.train(
        data=str(Path(__file__).parent / "data.yaml"),
        epochs=200,
        imgsz=1280,
        batch=4,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        warmup_epochs=5,
        weight_decay=0.0005,
        patience=50,

        # Augmentation
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.3,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.3,

        # Detection
        iou=0.5,
        max_det=300,

        # Performance
        workers=8,
        device=0,
        amp=True,
        project=str(Path(__file__).parent / "runs"),
        name="yolov8x_1280_v1",
        exist_ok=True,

        # Save
        save=True,
        save_period=25,  # Save checkpoint every 25 epochs
        plots=True,
        val=True,
    )

    print("Training complete!")
    print(f"Best model: {results.save_dir}/weights/best.pt")


if __name__ == "__main__":
    main()

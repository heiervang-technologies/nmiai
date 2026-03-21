"""
Pseudo-label ALL unlabeled data with our best YOLO model.
Run this when GPU is free after training.
Sources:
- 39 store photos
- 1,079 video frames
Output: YOLO format in data/pseudo_labeled_all/
"""
import sys
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = DATA_DIR / "pseudo_labeled_all"


def main():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("YOLO not available")
        return

    # Find best model
    model_paths = [
        Path("/home/me/ht/nmiai/runs/detect/yolov8x_v5_b2/weights/best.pt"),
        Path("/home/me/ht/nmiai/runs/detect/yolov8x_v4_12802/weights/best.pt"),
        Path("/home/me/ht/nmiai/runs/detect/yolov8x_v6_clean/weights/best.pt"),
    ]
    model_path = None
    for mp in model_paths:
        if mp.exists():
            model_path = mp
            break

    if not model_path:
        print("No model found")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "images").mkdir(exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(exist_ok=True)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    # Gather all unlabeled images
    store_dir = DATA_DIR / "store_photos"
    frames_dir = store_dir / "video_frames"

    images = []
    for img_path in sorted(store_dir.glob("*.jpg")):
        images.append(img_path)
    if frames_dir.exists():
        for img_path in sorted(frames_dir.glob("*.jpg")):
            images.append(img_path)

    print(f"Pseudo-labeling {len(images)} images...")

    conf = float(sys.argv[1]) if len(sys.argv) > 1 else 0.4
    labeled = 0

    for i, img_path in enumerate(images):
        results = model.predict(str(img_path), conf=conf, iou=0.5, imgsz=1280, verbose=False)
        if not results or len(results[0].boxes) == 0:
            continue

        result = results[0]
        img_h, img_w = result.orig_shape

        labels = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = ((x1 + x2) / 2) / img_w
            cy = ((y1 + y2) / 2) / img_h
            w = (x2 - x1) / img_w
            h = (y2 - y1) / img_h
            labels.append(f"{cls_id} {max(0,min(1,cx)):.6f} {max(0,min(1,cy)):.6f} {max(0,min(1,w)):.6f} {max(0,min(1,h)):.6f}")

        if labels:
            dst = OUTPUT_DIR / "images" / img_path.name
            if not dst.exists():
                dst.symlink_to(img_path.resolve())
            (OUTPUT_DIR / "labels" / f"{img_path.stem}.txt").write_text("\n".join(labels) + "\n")
            labeled += 1

        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(images)}: {labeled} labeled so far")

    print(f"\nDone! Labeled {labeled}/{len(images)} images")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

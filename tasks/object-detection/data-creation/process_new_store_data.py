"""
Process new store photos/videos as they arrive.
1. Copy new files from phone (if mounted)
2. Extract video frames
3. Pseudo-label with YOLO (when GPU available)
4. Add to external val and training datasets
"""
import json
import shutil
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).parent / "data"
STORE_DIR = DATA_DIR / "store_photos"
FRAMES_DIR = STORE_DIR / "video_frames_new"
EXTERNAL_VAL = DATA_DIR / "external_val"
CLEAN_DIR = DATA_DIR / "yolo_clean_with_val"
CLEAN_NV = DATA_DIR / "yolo_clean_no_val"
LARGE_SPLIT = DATA_DIR / "large_clean_split"


def extract_new_video_frames():
    """Extract frames from any new videos."""
    import cv2
    import numpy as np

    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    # Find videos not yet processed
    existing_prefixes = set()
    for f in FRAMES_DIR.glob("*.jpg"):
        prefix = f.stem.rsplit("_frame_", 1)[0]
        existing_prefixes.add(prefix)

    new_videos = []
    for vid in sorted(STORE_DIR.glob("*.mp4")):
        if vid.stem not in existing_prefixes:
            new_videos.append(vid)

    if not new_videos:
        print("No new videos to process")
        return 0

    total_frames = 0
    for vid_path in new_videos:
        print(f"Processing {vid_path.name}...")
        cap = cv2.VideoCapture(str(vid_path))
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        stride = max(1, int(fps * 0.5))  # Every 0.5 seconds

        prev_gray = None
        frame_idx = 0
        saved = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride == 0:
                # Scene change detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small = cv2.resize(gray, (320, 240))

                save = True
                if prev_gray is not None:
                    diff = float(cv2.absdiff(prev_gray, small).mean())
                    if diff < 5:  # Too similar to previous
                        save = False

                if save:
                    out_path = FRAMES_DIR / f"{vid_path.stem}_frame_{saved:04d}.jpg"
                    cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved += 1
                    prev_gray = small

            frame_idx += 1

        cap.release()
        total_frames += saved
        print(f"  Saved {saved} frames from {total} total ({fps:.0f} fps)")

    return total_frames


def pseudo_label_images(image_paths, output_dir, prefix="new_", conf=0.5):
    """Pseudo-label images with YOLO. Returns count of labeled images."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("YOLO not available, skipping pseudo-labeling")
        return 0

    # Find best model
    model_paths = [
        Path("/home/me/ht/nmiai/runs/detect/yolov8x_v5_b2/weights/best.pt"),
        Path("/home/me/ht/nmiai/runs/detect/yolov8x_v4_12802/weights/best.pt"),
    ]
    model_path = None
    for mp in model_paths:
        if mp.exists():
            model_path = mp
            break

    if not model_path:
        print("No YOLO model found")
        return 0

    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels").mkdir(parents=True, exist_ok=True)

    print(f"Loading {model_path.name}...")
    model = YOLO(str(model_path))

    labeled = 0
    for img_path in image_paths:
        name = f"{prefix}{img_path.stem}.jpg"

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
            dst = output_dir / "images" / name
            if not dst.exists():
                dst.symlink_to(img_path.resolve())
            (output_dir / "labels" / f"{prefix}{img_path.stem}.txt").write_text("\n".join(labels) + "\n")
            labeled += 1

    return labeled


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract-frames", action="store_true")
    parser.add_argument("--pseudo-label", action="store_true")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    print(f"Store photos: {len(list(STORE_DIR.glob('*.jpg')))} photos, {len(list(STORE_DIR.glob('*.mp4')))} videos")

    if args.extract_frames:
        n = extract_new_video_frames()
        print(f"Extracted {n} new frames")

    if args.pseudo_label:
        # Pseudo-label store photos
        photos = sorted(STORE_DIR.glob("*.jpg"))
        print(f"\nPseudo-labeling {len(photos)} store photos...")
        n = pseudo_label_images(photos, EXTERNAL_VAL, prefix="store_", conf=args.conf)
        print(f"Labeled {n} store photos")

        # Pseudo-label new video frames
        if FRAMES_DIR.exists():
            frames = sorted(FRAMES_DIR.glob("*.jpg"))
            print(f"\nPseudo-labeling {len(frames)} new frames...")
            n2 = pseudo_label_images(frames, EXTERNAL_VAL, prefix="nframe_", conf=args.conf)
            print(f"Labeled {n2} new frames")


if __name__ == "__main__":
    main()

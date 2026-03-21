#!/usr/bin/env python3
"""
Pseudo-label remaining video frames using best YOLO model.

Detects products in unlabeled video frames and generates YOLO-format labels.
Uses the trained YOLO model for both detection and classification.

After running:
  python build_mega_dataset.py --add-pseudolabels

Usage:
  python pseudo_label_video_frames.py [--conf 0.3] [--device 0] [--batch 16]
"""

import argparse
import json
import shutil
from pathlib import Path

from ultralytics import YOLO

BASE = Path(__file__).parent / "data-creation" / "data"
VIDEO_FRAMES = BASE / "store_photos" / "video_frames"
STORE_PHOTOS = BASE / "store_photos"
SILVER_EXISTING = BASE / "silver_augmented_dataset" / "train" / "images"
OUTPUT_VFRAMES = BASE / "pseudo_labels" / "video_frames"
OUTPUT_STORE = BASE / "pseudo_labels" / "store_photos"

# Best available model
MODEL_PATHS = [
    Path(__file__).parent / "yolo-approach" / "best.pt",
    Path(__file__).parent / "yolo-approach" / "runs" / "detect" / "train" / "weights" / "best.pt",
]


def find_model():
    for p in MODEL_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError(f"No model found at: {MODEL_PATHS}")


def get_already_labeled():
    """Get set of filenames already in silver dataset."""
    existing = set()
    if SILVER_EXISTING.exists():
        for p in SILVER_EXISTING.iterdir():
            existing.add(p.name)
    return existing


def pseudo_label(src_dir, out_dir, model, conf_thresh, device, batch_size,
                 prefix="", existing=None, extensions=('.jpg', '.jpeg', '.png')):
    """Run YOLO inference and save YOLO-format labels."""
    out_img = out_dir / "images"
    out_lbl = out_dir / "labels"
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    # Collect unlabeled images
    images_to_label = []
    for p in sorted(src_dir.iterdir()):
        if p.suffix.lower() not in extensions:
            continue
        # Skip if already in silver
        target_name = f"{prefix}_{p.name}" if prefix else p.name
        if existing and (target_name in existing or p.name in existing):
            continue
        images_to_label.append(p)

    print(f"  Images to pseudo-label: {len(images_to_label)}")
    if not images_to_label:
        return 0

    total_detections = 0
    # Process in batches
    for i in range(0, len(images_to_label), batch_size):
        batch = images_to_label[i:i+batch_size]
        batch_paths = [str(p) for p in batch]

        results = model.predict(
            batch_paths,
            conf=conf_thresh,
            iou=0.5,
            device=device,
            verbose=False,
            imgsz=1280,
        )

        for img_path, result in zip(batch, results):
            boxes = result.boxes
            if len(boxes) == 0:
                continue

            # Save image
            target_name = f"{prefix}_{img_path.name}" if prefix else img_path.name
            shutil.copy2(img_path, out_img / target_name)

            # Save YOLO label
            label_name = Path(target_name).stem + ".txt"
            with open(out_lbl / label_name, "w") as f:
                for box in boxes:
                    cls = int(box.cls.item())
                    # Convert to YOLO format (center_x, center_y, w, h) normalized
                    xywhn = box.xywhn[0]
                    cx, cy, w, h = xywhn[0].item(), xywhn[1].item(), xywhn[2].item(), xywhn[3].item()
                    f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                    total_detections += 1

        print(f"  Processed {min(i+batch_size, len(images_to_label))}/{len(images_to_label)} images, {total_detections} detections so far")

    return total_detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--device", default="0", help="Device (0 for GPU, cpu for CPU)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    args = parser.parse_args()

    model_path = find_model()
    print(f"Model: {model_path}")
    print(f"Confidence threshold: {args.conf}")
    print(f"Device: {args.device}")

    model = YOLO(str(model_path))
    existing = get_already_labeled()
    print(f"Already labeled (in silver): {len(existing)}")

    # Pseudo-label video frames
    print(f"\n=== Pseudo-labeling video frames ===")
    if VIDEO_FRAMES.exists():
        vf_detections = pseudo_label(
            VIDEO_FRAMES, OUTPUT_VFRAMES, model, args.conf, args.device,
            args.batch, prefix="plvf", existing=existing
        )
        print(f"  Total video frame detections: {vf_detections}")
    else:
        print(f"  Video frames not found at {VIDEO_FRAMES}")

    # Pseudo-label remaining store photos
    print(f"\n=== Pseudo-labeling store photos ===")
    store_detections = pseudo_label(
        STORE_PHOTOS, OUTPUT_STORE, model, args.conf, args.device,
        args.batch, prefix="plsp", existing=existing,
        extensions=('.jpg', '.jpeg', '.png')
    )
    print(f"  Total store photo detections: {store_detections}")

    print(f"\n=== DONE ===")
    print(f"Video frame pseudo-labels: {OUTPUT_VFRAMES}")
    print(f"Store photo pseudo-labels: {OUTPUT_STORE}")
    print(f"\nTo add to mega dataset:")
    print(f"  python build_mega_dataset.py --add-pseudolabels")


if __name__ == "__main__":
    main()

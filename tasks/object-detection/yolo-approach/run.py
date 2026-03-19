"""NM i AI 2026 - Object Detection Submission (YOLO Approach)"""
import json
import argparse
from pathlib import Path
import torch
import numpy as np
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to images directory')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load model - weights file next to this script
    model_path = Path(__file__).parent / 'best.pt'
    model = YOLO(str(model_path))

    # Find all images
    data_dir = Path(args.data)
    images = sorted(list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.jpeg')) + list(data_dir.glob('*.png')))
    print(f"Found {len(images)} images in {data_dir}")

    results_list = []

    # Run inference in batches
    batch_size = 8
    for i in range(0, len(images), batch_size):
        batch_paths = images[i:i + batch_size]
        batch_sources = [str(p) for p in batch_paths]

        results = model.predict(
            source=batch_sources,
            conf=0.001,
            iou=0.45,
            max_det=300,
            imgsz=1280,
            device=device,
            half=True,
            verbose=False,
        )

        for img_path, r in zip(batch_paths, results):
            predictions = []
            boxes = r.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy()

                for j in range(len(boxes)):
                    x1, y1, x2, y2 = xyxy[j]
                    w = x2 - x1
                    h = y2 - y1
                    predictions.append({
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'category_id': int(classes[j]),
                        'confidence': float(confs[j])
                    })

            results_list.append({
                'image_id': img_path.name,
                'predictions': predictions
            })

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_list, indent=2))
    print(f"Written {len(results_list)} results to {output_path}")


if __name__ == '__main__':
    main()

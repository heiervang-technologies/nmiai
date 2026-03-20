"""NM i AI 2026 - Object Detection Submission (ONNX Runtime, no ultralytics)"""
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# YOLOv8 ONNX post-processing constants
CONF_THRESH = 0.001
IOU_THRESH = 0.45
MAX_DET = 300
INPUT_SIZE = 1280
NUM_CLASSES = 356


def letterbox(img, new_shape=INPUT_SIZE):
    """Resize and pad image to square, preserving aspect ratio."""
    h, w = img.shape[:2]
    ratio = new_shape / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad to square
    dw = (new_shape - new_w) / 2
    dh = (new_shape - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_padded, ratio, (dw, dh)


def preprocess(img):
    """Preprocess image for YOLOv8 ONNX input."""
    img_lb, ratio, (dw, dh) = letterbox(img, INPUT_SIZE)
    # BGR -> RGB, HWC -> CHW, normalize to [0, 1], add batch dim
    blob = img_lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, 0)
    return blob, ratio, dw, dh


def nms_per_class(boxes, scores, class_ids, iou_thresh=IOU_THRESH, max_det=MAX_DET):
    """Apply per-class NMS using OpenCV."""
    if len(boxes) == 0:
        return [], [], []

    # Convert to list format for cv2.dnn.NMSBoxesBatched
    # boxes as [x, y, w, h] for cv2 NMS
    boxes_xywh = []
    for b in boxes:
        x1, y1, x2, y2 = b
        boxes_xywh.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])

    indices = cv2.dnn.NMSBoxesBatched(
        boxes_xywh,
        scores.tolist(),
        class_ids.tolist(),
        CONF_THRESH,
        iou_thresh
    )

    if len(indices) == 0:
        return [], [], []

    indices = indices.flatten()[:max_det]
    return boxes[indices], scores[indices], class_ids[indices]


def postprocess(output, ratio, dw, dh, orig_h, orig_w):
    """Post-process YOLOv8 ONNX output to detections."""
    # YOLOv8 output: [1, 4+nc, num_boxes] -> transpose to [num_boxes, 4+nc]
    preds = output[0].squeeze(0).T  # [num_boxes, 4+356]

    # Split box coords and class scores
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    class_scores = preds[:, 4:]  # [num_boxes, 356]

    # Get best class and confidence per box
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    # Filter by confidence
    mask = confidences > CONF_THRESH
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    # Convert center to corner format (xyxy)
    x1 = cx - bw / 2
    y1 = cy - bh / 2
    x2 = cx + bw / 2
    y2 = cy + bh / 2

    # Undo letterbox: subtract padding, divide by ratio
    x1 = (x1 - dw) / ratio
    y1 = (y1 - dh) / ratio
    x2 = (x2 - dw) / ratio
    y2 = (y2 - dh) / ratio

    # Clip to image bounds
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # Per-class NMS
    boxes, confidences, class_ids = nms_per_class(boxes, confidences, class_ids)

    if len(boxes) == 0:
        return []

    # Convert to COCO format [x, y, w, h]
    predictions = []
    for i in range(len(boxes)):
        bx1, by1, bx2, by2 = boxes[i]
        predictions.append({
            'bbox': [float(bx1), float(by1), float(bx2 - bx1), float(by2 - by1)],
            'category_id': int(class_ids[i]),
            'score': float(confidences[i])
        })

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to images directory')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()

    # Load ONNX model
    model_path = Path(__file__).parent / 'best.onnx'
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name

    # Find all images
    data_dir = Path(args.input)
    images = sorted(list(data_dir.glob('*.jpg')) + list(data_dir.glob('*.jpeg')) + list(data_dir.glob('*.png')))
    print(f"Found {len(images)} images in {data_dir}")

    results_list = []

    for img_path in images:
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            results_list.append({'image_id': img_path.name, 'predictions': []})
            continue

        orig_h, orig_w = img.shape[:2]

        # Preprocess
        blob, ratio, dw, dh = preprocess(img)

        # Run inference
        outputs = session.run(None, {input_name: blob})

        # Post-process
        predictions = postprocess(outputs, ratio, dw, dh, orig_h, orig_w)

        results_list.append({
            'image_id': img_path.name,
            'predictions': predictions
        })

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results_list))
    print(f"Written {len(results_list)} results to {output_path}")


if __name__ == '__main__':
    main()

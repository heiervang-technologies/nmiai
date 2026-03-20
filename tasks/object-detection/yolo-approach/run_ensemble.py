"""NM i AI 2026 - Object Detection: 2-Model Ensemble with WBF
YOLO11x + YOLO26x, both trained on V3 augmented dataset.
Uses ensemble_boxes (pre-installed in sandbox) for Weighted Boxes Fusion.
"""
import argparse
import json
import pathlib
import numpy as np
import cv2
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion


# Constants
INPUT_SIZE = 1280
CONF_THRESH = 0.001
IOU_THRESH = 0.45
WBF_IOU_THRESH = 0.55
MAX_DET = 300
NUM_CLASSES = 356


def letterbox(img, new_shape=INPUT_SIZE):
    """Resize and pad image to square, preserving aspect ratio."""
    h, w = img.shape[:2]
    ratio = new_shape / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (new_shape - new_w) / 2
    dh = (new_shape - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_padded, ratio, dw, dh


def preprocess(img):
    """Preprocess image for YOLO ONNX input."""
    img_lb, ratio, dw, dh = letterbox(img, INPUT_SIZE)
    blob = img_lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    blob = np.expand_dims(blob, 0)
    return blob, ratio, dw, dh


def decode_yolo_output(output, ratio, dw, dh, orig_h, orig_w):
    """Decode raw YOLO ONNX output to boxes, scores, class_ids."""
    preds = output[0].squeeze(0)
    # YOLOv8/11/26 output: [4+nc, num_boxes] -> transpose
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    class_scores = preds[:, 4:]

    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)

    mask = confidences > CONF_THRESH
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(cx) == 0:
        return np.array([]), np.array([]), np.array([])

    # Convert center to corner (xyxy)
    x1 = (cx - bw / 2 - dw) / ratio
    y1 = (cy - bh / 2 - dh) / ratio
    x2 = (cx + bw / 2 - dw) / ratio
    y2 = (cy + bh / 2 - dh) / ratio

    # Clip
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    return boxes, confidences, class_ids


def run_model(session, input_name, blob, ratio, dw, dh, orig_h, orig_w):
    """Run one ONNX model and return decoded predictions."""
    outputs = session.run(None, {input_name: blob})
    return decode_yolo_output(outputs, ratio, dw, dh, orig_h, orig_w)


def ensemble_wbf(all_boxes, all_scores, all_labels, orig_h, orig_w,
                 iou_thresh=WBF_IOU_THRESH, conf_thresh=CONF_THRESH):
    """Apply Weighted Boxes Fusion across multiple models."""
    if not all_boxes or all(len(b) == 0 for b in all_boxes):
        return np.array([]), np.array([]), np.array([])

    # Normalize boxes to [0, 1] for WBF
    norm_boxes = []
    norm_scores = []
    norm_labels = []
    for boxes, scores, labels in zip(all_boxes, all_scores, all_labels):
        if len(boxes) == 0:
            norm_boxes.append(np.array([]).reshape(0, 4))
            norm_scores.append(np.array([]))
            norm_labels.append(np.array([]))
            continue
        nb = boxes.copy()
        nb[:, 0] /= orig_w
        nb[:, 1] /= orig_h
        nb[:, 2] /= orig_w
        nb[:, 3] /= orig_h
        nb = np.clip(nb, 0, 1)
        norm_boxes.append(nb.tolist())
        norm_scores.append(scores.tolist())
        norm_labels.append(labels.tolist())

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        norm_boxes, norm_scores, norm_labels,
        iou_thr=iou_thresh,
        skip_box_thr=conf_thresh,
    )

    # Denormalize
    if len(fused_boxes) > 0:
        fused_boxes[:, 0] *= orig_w
        fused_boxes[:, 1] *= orig_h
        fused_boxes[:, 2] *= orig_w
        fused_boxes[:, 3] *= orig_h

    return fused_boxes[:MAX_DET], fused_scores[:MAX_DET], fused_labels[:MAX_DET]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    base_dir = pathlib.Path(__file__).parent
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load both models
    models = []
    for name in ['yolo11x_v3.onnx', 'yolo26x_v3.onnx']:
        path = base_dir / name
        session = ort.InferenceSession(str(path), providers=providers)
        input_name = session.get_inputs()[0].name
        models.append((session, input_name))

    images_dir = pathlib.Path(args.input)
    image_files = sorted(
        list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
    )

    results = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        orig_h, orig_w = img.shape[:2]
        blob, ratio, dw, dh = preprocess(img)

        # Run both models
        all_boxes = []
        all_scores = []
        all_labels = []
        for session, input_name in models:
            boxes, scores, labels = run_model(session, input_name, blob, ratio, dw, dh, orig_h, orig_w)
            all_boxes.append(boxes if len(boxes) > 0 else np.array([]).reshape(0, 4))
            all_scores.append(scores if len(scores) > 0 else np.array([]))
            all_labels.append(labels if len(labels) > 0 else np.array([]))

        # Fuse with WBF
        fused_boxes, fused_scores, fused_labels = ensemble_wbf(
            all_boxes, all_scores, all_labels, orig_h, orig_w
        )

        # Convert to flat COCO format
        for i in range(len(fused_boxes)):
            x1, y1, x2, y2 = fused_boxes[i]
            results.append({
                'image_id': int(img_path.stem.replace('img_', '')),
                'bbox': [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                'category_id': int(fused_labels[i]),
                'score': round(float(fused_scores[i]), 4),
            })

    output_path = pathlib.Path(args.output)
    with open(str(output_path), 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()

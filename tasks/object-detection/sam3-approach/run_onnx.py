"""
Submission run.py for ONNX-based inference.
No blocked imports (os, subprocess, socket, ctypes).
Uses only: pathlib, json, numpy, cv2, torch, onnxruntime.

Usage:
    python run.py --data /data/images/ --output /output.json
"""

import argparse
import json
import pathlib
import numpy as np
import cv2
import onnxruntime as ort
import torch
import torchvision


SCRIPT_DIR = pathlib.Path(__file__).parent
MODEL_PATH = SCRIPT_DIR / "best.onnx"
IMGSZ = 1280
CONF_THRESH = 0.001
IOU_THRESH = 0.45
MAX_DET = 300


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize and pad image maintaining aspect ratio."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return img, r, (dw, dh)


def preprocess(img_path, imgsz):
    """Load and preprocess image."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None, None
    orig_shape = img.shape[:2]
    img_lb, ratio, pad = letterbox(img, imgsz)
    img_lb = img_lb[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
    img_lb = np.ascontiguousarray(img_lb, dtype=np.float32) / 255.0
    img_lb = np.expand_dims(img_lb, 0)
    return img_lb, orig_shape, ratio, pad


def nms_per_class(boxes_xyxy, scores, class_ids, iou_thresh, max_det):
    """Per-class NMS using torchvision."""
    if len(boxes_xyxy) == 0:
        return np.array([]), np.array([]), np.array([])

    boxes_t = torch.from_numpy(boxes_xyxy).float()
    scores_t = torch.from_numpy(scores).float()
    class_ids_t = torch.from_numpy(class_ids).long()

    # Offset boxes by class to do per-class NMS in one call
    offsets = class_ids_t.float() * 4096.0
    boxes_offset = boxes_t + offsets.unsqueeze(1)

    keep = torchvision.ops.nms(boxes_offset, scores_t, iou_thresh)
    if len(keep) > max_det:
        keep = keep[:max_det]

    return (
        boxes_xyxy[keep.numpy()],
        scores[keep.numpy()],
        class_ids[keep.numpy()],
    )


def postprocess(output, orig_shape, ratio, pad, conf_thresh, iou_thresh, max_det):
    """Post-process YOLO ONNX output to COCO format predictions.

    YOLO ONNX output shape: [1, 4+nc, num_boxes]
    First 4 rows: cx, cy, w, h (center format)
    Remaining rows: class scores
    """
    preds = output[0]  # [1, 4+nc, num_boxes] or [1, num_boxes, 4+nc]

    if preds.ndim == 3:
        preds = preds[0]  # [4+nc, num_boxes] or [num_boxes, 4+nc]

    # YOLO exports as [4+nc, num_boxes] — transpose if needed
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T  # -> [num_boxes, 4+nc]

    if len(preds) == 0:
        return []

    # Split boxes and class scores
    # Check if end2end format (6 columns: x1,y1,x2,y2,conf,cls)
    if preds.shape[1] == 6:
        boxes_xyxy = preds[:, :4]
        confs = preds[:, 4]
        class_ids = preds[:, 5].astype(int)
        # Filter by confidence
        mask = confs > conf_thresh
        boxes_xyxy = boxes_xyxy[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]
    else:
        # Standard YOLO format: [num_boxes, 4+nc]
        # First 4: cx, cy, w, h
        cx = preds[:, 0]
        cy = preds[:, 1]
        w = preds[:, 2]
        h = preds[:, 3]
        class_scores = preds[:, 4:]

        # Get max class score and class id
        confs = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        # Filter by confidence
        mask = confs > conf_thresh
        cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
        confs = confs[mask]
        class_ids = class_ids[mask]

        # Convert cx,cy,w,h to x1,y1,x2,y2
        boxes_xyxy = np.stack(
            [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1
        )

    if len(boxes_xyxy) == 0:
        return []

    # NMS
    boxes_xyxy, confs, class_ids = nms_per_class(
        boxes_xyxy, confs, class_ids, iou_thresh, max_det
    )

    if len(boxes_xyxy) == 0:
        return []

    # Scale boxes back to original image coordinates
    dw, dh = pad
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dw) / ratio
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dh) / ratio

    # Clip to image bounds
    oh, ow = orig_shape
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, ow)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, oh)

    # Convert xyxy -> COCO [x, y, w, h]
    predictions = []
    for i in range(len(boxes_xyxy)):
        x1, y1, x2, y2 = boxes_xyxy[i]
        bw, bh = x2 - x1, y2 - y1
        if bw < 2 or bh < 2:
            continue
        predictions.append(
            {
                "bbox": [
                    round(float(x1), 2),
                    round(float(y1), 2),
                    round(float(bw), 2),
                    round(float(bh), 2),
                ],
                "category_id": int(class_ids[i]),
                "score": round(float(confs[i]), 4),
            }
        )

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    images_dir = pathlib.Path(args.input)
    output_path = pathlib.Path(args.output)

    # Load ONNX model
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(MODEL_PATH), providers=providers)
    input_name = session.get_inputs()[0].name

    # Get all test images
    image_files = sorted(images_dir.glob("*.jpg"))

    results = []
    for img_path in image_files:
        img_input, orig_shape, ratio, pad = preprocess(img_path, IMGSZ)
        if img_input is None:
            results.append({"image_id": img_path.name, "predictions": []})
            continue

        outputs = session.run(None, {input_name: img_input})
        predictions = postprocess(
            outputs, orig_shape, ratio, pad, CONF_THRESH, IOU_THRESH, MAX_DET
        )

        results.append(
            {
                "image_id": img_path.name,
                "predictions": predictions,
            }
        )

    with open(output_path, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

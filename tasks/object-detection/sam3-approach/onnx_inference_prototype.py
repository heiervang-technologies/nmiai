"""
Pure ONNX Runtime inference for YOLO26 — no ultralytics dependency.

This proves we can train YOLO26x locally with ultralytics 8.4.24,
export to ONNX, and run inference in the sandbox using only
onnxruntime-gpu (pre-installed) + numpy + cv2 + torch.

No os/subprocess/socket imports needed.

Usage (local test):
    uv run python onnx_inference_prototype.py --model best.onnx --data /data/images/ --output /output.json
"""

import argparse
import json
import pathlib
import numpy as np
import torch
import cv2


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """Resize and pad image to target shape maintaining aspect ratio."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def preprocess(img_path, imgsz=640):
    """Load and preprocess image for ONNX model."""
    img = cv2.imread(str(img_path))
    orig_shape = img.shape[:2]
    img_lb, ratio, pad = letterbox(img, imgsz)
    img_lb = img_lb[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img_lb = np.ascontiguousarray(img_lb, dtype=np.float32) / 255.0
    img_lb = np.expand_dims(img_lb, 0)  # add batch dim
    return img_lb, orig_shape, ratio, pad


def postprocess_yolo26(output, orig_shape, ratio, pad, conf_thresh=0.001, iou_thresh=0.5, nc=356):
    """Post-process YOLO26 ONNX output (end2end mode with NMS-free).

    YOLO26 uses end2end=True, so output is already post-NMS.
    Output shape: [batch, num_detections, 4+nc] or [batch, num_detections, 6]
    """
    # YOLO26 end2end output: [batch, max_det, 6] = [x1,y1,x2,y2,conf,class]
    # Or standard YOLO output: [batch, num_boxes, 4+nc]
    preds = output[0]  # first output tensor

    if preds.ndim == 3:
        preds = preds[0]  # remove batch dim

    # Determine output format
    if preds.shape[-1] == 6:
        # End2end format: [x1, y1, x2, y2, conf, class_id]
        boxes_xyxy = preds[:, :4]
        confs = preds[:, 4]
        class_ids = preds[:, 5].astype(int)
    else:
        # Standard format: [x1, y1, x2, y2, class_scores...]
        boxes_xyxy = preds[:, :4]
        class_scores = preds[:, 4:]
        confs = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

    # Filter by confidence
    mask = confs > conf_thresh
    boxes_xyxy = boxes_xyxy[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]

    if len(boxes_xyxy) == 0:
        return [], [], []

    # Scale boxes back to original image coordinates
    dw, dh = pad
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dw) / ratio
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dh) / ratio

    # Clip to image bounds
    h, w = orig_shape
    boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, w)
    boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, h)

    # Convert xyxy to COCO format [x, y, w, h]
    boxes_coco = np.zeros_like(boxes_xyxy)
    boxes_coco[:, 0] = boxes_xyxy[:, 0]
    boxes_coco[:, 1] = boxes_xyxy[:, 1]
    boxes_coco[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
    boxes_coco[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]

    # Filter out tiny boxes
    valid = (boxes_coco[:, 2] > 2) & (boxes_coco[:, 3] > 2)
    boxes_coco = boxes_coco[valid]
    confs = confs[valid]
    class_ids = class_ids[valid]

    return boxes_coco.tolist(), confs.tolist(), class_ids.tolist()


def run_inference(model_path, images_dir, output_path, imgsz=640, conf_thresh=0.001):
    """Run ONNX inference on all images."""
    import onnxruntime as ort

    # Create ONNX session with GPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name

    # Get all test images
    images_dir = pathlib.Path(images_dir)
    image_files = sorted(images_dir.glob("*.jpg"))

    results = []
    for img_path in image_files:
        img_input, orig_shape, ratio, pad = preprocess(img_path, imgsz)
        outputs = session.run(None, {input_name: img_input})

        boxes, confs, class_ids = postprocess_yolo26(
            outputs, orig_shape, ratio, pad, conf_thresh=conf_thresh
        )

        predictions = []
        for box, conf, cls_id in zip(boxes, confs, class_ids):
            predictions.append(
                {
                    "bbox": [round(v, 2) for v in box],
                    "category_id": int(cls_id),
                    "score": round(float(conf), 4),
                }
            )

        results.append(
            {
                "image_id": img_path.name,
                "predictions": predictions,
            }
        )

    # Write output
    output_path = pathlib.Path(output_path)
    with open(output_path, "w") as f:
        json.dump(results, f)

    print(f"Inference complete: {len(results)} images, {sum(len(r['predictions']) for r in results)} total detections")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to ONNX model")
    parser.add_argument("--input", required=True, help="Path to images directory")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    args = parser.parse_args()
    run_inference(args.model, args.input, args.output, args.imgsz, args.conf)

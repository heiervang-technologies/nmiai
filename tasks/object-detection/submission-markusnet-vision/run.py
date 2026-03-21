"""
YOLO detection + MarkusNet vision ONNX classification.
All models via onnxruntime - no pure PyTorch reimplementation.

Usage: python run.py --input /data/images --output /predictions.json
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

SCRIPT_DIR = Path(__file__).parent
NUM_CLASSES = 356
CONF_THRESH = 0.001
NMS_IOU_THRESH = 0.45
MAX_DET = 300

# Preprocessing: SQUARE 256x256 (like DINOv2). Produces 16*16=256 patches.
FIXED_H, FIXED_W = 256, 256
MERGE_SIZE = 2
PATCH_SIZE = 16
TEMPORAL_PATCH = 2


def letterbox(image, new_shape, color=(114, 114, 114)):
    h, w = image.shape[:2]
    ratio = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(round(w * ratio)), int(round(h * ratio))
    dw, dh = (new_shape[1] - nw) / 2, (new_shape[0] - nh) / 2
    if (w, h) != (nw, nh):
        image = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


def nms(boxes, scores, thresh):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        order = order[np.where(iou <= thresh)[0] + 1]
    return keep


def detect_yolo(img_bgr, session, inp_name, H, W):
    oh, ow = img_bgr.shape[:2]
    lb, ratio, (dw, dh) = letterbox(img_bgr, (H, W))
    tensor = cv2.cvtColor(lb, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None]

    out = session.run(None, {inp_name: tensor})[0][0].T
    boxes, scores_all = out[:, :4], out[:, 4:]
    cls_ids = np.argmax(scores_all, 1)
    confs = np.array([scores_all[j, cls_ids[j]] for j in range(len(cls_ids))])

    mask = confs > CONF_THRESH
    boxes, cls_ids, confs = boxes[mask], cls_ids[mask], confs[mask]
    if len(boxes) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    xyxy = np.zeros_like(boxes)
    xyxy[:, 0] = (boxes[:, 0] - boxes[:, 2] / 2 - dw) / ratio
    xyxy[:, 1] = (boxes[:, 1] - boxes[:, 3] / 2 - dh) / ratio
    xyxy[:, 2] = (boxes[:, 0] + boxes[:, 2] / 2 - dw) / ratio
    xyxy[:, 3] = (boxes[:, 1] + boxes[:, 3] / 2 - dh) / ratio
    xyxy[:, [0, 2]] = np.clip(xyxy[:, [0, 2]], 0, ow)
    xyxy[:, [1, 3]] = np.clip(xyxy[:, [1, 3]], 0, oh)

    fb, fs, fc = [], [], []
    for c in range(NUM_CLASSES):
        m = cls_ids == c
        if not np.any(m):
            continue
        cb, cs = xyxy[m], confs[m]
        keep = nms(cb, cs, NMS_IOU_THRESH)
        fb.append(cb[keep])
        fs.append(cs[keep])
        fc.extend([c] * len(keep))
    if not fb:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)
    fb, fs = np.concatenate(fb), np.concatenate(fs)
    if len(fs) > MAX_DET:
        top = np.argsort(fs)[::-1][:MAX_DET]
        fb, fs, fc = fb[top], fs[top], [fc[j] for j in top]
    return fb, fs, np.array(fc)


def preprocess_crop(crop_pil):
    """Preprocess a PIL crop to pixel_values [280, 1536] matching the ONNX input.

    Resizes to FIXED_H x FIXED_W (320x224), normalizes with mean/std=0.5,
    creates temporal frames, patches, and reorders to 2x2 merge groups.
    """
    img = np.array(crop_pil.convert("RGB"))
    resized = cv2.resize(img, (FIXED_W, FIXED_H), interpolation=cv2.INTER_LINEAR)

    # Normalize: (pixel/255 - 0.5) / 0.5
    tensor = resized.astype(np.float32) / 255.0
    tensor = (tensor - 0.5) / 0.5  # [H, W, 3]
    tensor = np.transpose(tensor, (2, 0, 1))  # [3, H, W]

    # Stack temporal frames
    frames = np.stack([tensor, tensor], axis=0)  # [2, 3, H, W]

    h_p = FIXED_H // PATCH_SIZE  # 20
    w_p = FIXED_W // PATCH_SIZE  # 14
    pH, pW = PATCH_SIZE, PATCH_SIZE
    pT = TEMPORAL_PATCH

    # Reshape to patches (row-major order)
    x = frames.reshape(1, pT, 3, h_p, pH, w_p, pW)
    x = np.transpose(x, (0, 3, 5, 2, 1, 4, 6))  # [1, h_p, w_p, C, T, pH, pW]
    x = x.reshape(h_p * w_p, 3 * pT * pH * pW)  # [280, 1536]

    # Reorder to 2x2 spatial merge groups (matching transformers processor)
    merge = MERGE_SIZE
    x = x.reshape(h_p, w_p, -1)
    x = x.reshape(h_p // merge, merge, w_p // merge, merge, -1)
    x = np.transpose(x, (0, 2, 1, 3, 4))  # [h_p/2, w_p/2, 2, 2, 1536]
    x = x.reshape(h_p * w_p, -1)  # [280, 1536]

    return x


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    # Load YOLO
    yolo_path = SCRIPT_DIR / "best.onnx"
    yolo_sess = ort.InferenceSession(str(yolo_path), providers=providers)
    yolo_inp = yolo_sess.get_inputs()[0]
    H, W = yolo_inp.shape[2], yolo_inp.shape[3]

    # Load MarkusNet vision encoder
    vision_path = SCRIPT_DIR / "markusnet_vision_int8.onnx"
    vision_sess = ort.InferenceSession(str(vision_path), providers=providers)

    # Load probe classifier
    probe_path = SCRIPT_DIR / "markusnet_probe.onnx"
    probe_sess = ort.InferenceSession(str(probe_path), providers=providers)

    # Process images
    images_dir = Path(args.input)
    results = []

    for img_path in sorted(images_dir.glob("*.jpg")):
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        oh, ow = img_bgr.shape[:2]
        img_id = int(img_path.stem.replace("img_", ""))

        boxes, scores, yolo_cls = detect_yolo(img_bgr, yolo_sess, yolo_inp.name, H, W)
        if len(boxes) == 0:
            continue

        img_pil = Image.open(img_path).convert("RGB")

        for j in range(len(boxes)):
            x1, y1, x2, y2 = boxes[j]
            x1i, y1i = max(0, int(x1)), max(0, int(y1))
            x2i, y2i = min(ow, int(x2)), min(oh, int(y2))

            if x2i > x1i and y2i > y1i:
                crop = img_pil.crop((x1i, y1i, x2i, y2i))
            else:
                crop = Image.new("RGB", (32, 32), (128, 128, 128))

            # Vision encoder
            pv = preprocess_crop(crop)  # [280, 1536]
            features = vision_sess.run(None, {"pixel_values": pv})[0]  # [1, 768]

            # Probe classifier
            logits = probe_sess.run(None, {"features": features})[0]  # [1, 356]
            cat_id = int(np.argmax(logits[0]))

            results.append({
                "image_id": img_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "category_id": cat_id,
                "score": float(scores[j]),
            })

    with open(args.output, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

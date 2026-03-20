"""NM i AI 2026 - Object Detection: Single YOLO + DINOv2 Classification
Single best.onnx detector + DINOv2 ViT-S/14 linear probe classifier.
NMS IoU=0.45, conf=0.001, max_det=300.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_data_config


SCRIPT_DIR = Path(__file__).parent
MODEL_FILE = "best.onnx"
DINO_WEIGHTS = "dinov2_vits14.pth"
LINEAR_PROBE = "linear_probe.pth"
REF_EMBEDDINGS = "ref_embeddings.pth"

NUM_CLASSES = 356
CONF_THRESH = 0.001
NMS_IOU_THRESH = 0.45
MAX_DET = 300
CROP_BATCH_SIZE = 128


def load_onnx_session(model_path: Path, providers: list):
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_info = session.get_inputs()[0]
    return session, input_info.name, tuple(input_info.shape)


def letterbox(image, new_shape, color=(114, 114, 114)):
    height, width = image.shape[:2]
    ratio = min(new_shape[0] / height, new_shape[1] / width)
    resized_wh = (int(round(width * ratio)), int(round(height * ratio)))
    pad_w = (new_shape[1] - resized_wh[0]) / 2
    pad_h = (new_shape[0] - resized_wh[1]) / 2
    if (width, height) != resized_wh:
        image = cv2.resize(image, resized_wh, interpolation=cv2.INTER_LINEAR)
    top = int(round(pad_h - 0.1))
    bottom = int(round(pad_h + 0.1))
    left = int(round(pad_w - 0.1))
    right = int(round(pad_w + 0.1))
    bordered = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return bordered, ratio, (pad_w, pad_h)


def preprocess_image(image_bgr, input_shape):
    target_h, target_w = int(input_shape[2]), int(input_shape[3])
    letterboxed, ratio, pad = letterbox(image_bgr, (target_h, target_w))
    image_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = image_rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
    return tensor, ratio, pad


def compute_iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + areas - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms_by_class(boxes, scores, labels, iou_thresh):
    if len(boxes) == 0:
        return boxes, scores, labels
    kept_boxes, kept_scores, kept_labels = [], [], []
    for class_id in np.unique(labels):
        class_mask = labels == class_id
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        order = np.argsort(class_scores)[::-1]
        while len(order) > 0:
            current = order[0]
            kept_boxes.append(class_boxes[current])
            kept_scores.append(class_scores[current])
            kept_labels.append(class_id)
            if len(order) == 1:
                break
            iou = compute_iou(class_boxes[current], class_boxes[order[1:]])
            order = order[1:][iou <= iou_thresh]
    kept_scores = np.asarray(kept_scores, dtype=np.float32)
    ranking = np.argsort(kept_scores)[::-1]
    return (
        np.asarray(kept_boxes, dtype=np.float32)[ranking],
        kept_scores[ranking],
        np.asarray(kept_labels, dtype=np.int64)[ranking],
    )


def decode_yolo_output(output, ratio, pad, original_shape):
    prediction = output[0]
    if prediction.ndim == 3:
        prediction = prediction[0]
    if prediction.shape[0] < prediction.shape[1]:
        prediction = prediction.T
    box_cxcywh = prediction[:, :4]
    class_scores = prediction[:, 4:]
    labels = np.argmax(class_scores, axis=1)
    scores = np.max(class_scores, axis=1)
    mask = scores > CONF_THRESH
    box_cxcywh, labels, scores = box_cxcywh[mask], labels[mask], scores[mask]
    if len(box_cxcywh) == 0:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)
    boxes = np.empty_like(box_cxcywh)
    boxes[:, 0] = box_cxcywh[:, 0] - box_cxcywh[:, 2] / 2
    boxes[:, 1] = box_cxcywh[:, 1] - box_cxcywh[:, 3] / 2
    boxes[:, 2] = box_cxcywh[:, 0] + box_cxcywh[:, 2] / 2
    boxes[:, 3] = box_cxcywh[:, 1] + box_cxcywh[:, 3] / 2
    pad_w, pad_h = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio
    original_h, original_w = original_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_h)
    valid = (boxes[:, 2] - boxes[:, 0] > 2) & (boxes[:, 3] - boxes[:, 1] > 2)
    boxes, scores, labels = boxes[valid], scores[valid], labels[valid]
    if len(boxes) == 0:
        return np.empty((0, 4), np.float32), np.empty(0, np.float32), np.empty(0, np.int64)
    boxes, scores, labels = nms_by_class(boxes, scores, labels, NMS_IOU_THRESH)
    if len(scores) > MAX_DET:
        top = np.argsort(scores)[::-1][:MAX_DET]
        boxes, scores, labels = boxes[top], scores[top], labels[top]
    return boxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int64)


def load_dino(device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    state_dict = torch.load(SCRIPT_DIR / DINO_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)
    return model, transform


def load_linear_probe(device):
    probe_path = SCRIPT_DIR / LINEAR_PROBE
    if not probe_path.exists():
        return None
    state = torch.load(probe_path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "weight" not in state:
        return None
    num_classes, embed_dim = state["weight"].shape
    probe = torch.nn.Linear(embed_dim, num_classes)
    probe.load_state_dict(state)
    return probe.eval().to(device)


def classify_boxes(image_bgr, boxes, detector_labels, detector_scores, dino_model, transform, linear_probe, device):
    if len(boxes) == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    category_ids, confidences = [], []

    for start in range(0, len(boxes), CROP_BATCH_SIZE):
        batch_boxes = boxes[start:start + CROP_BATCH_SIZE]
        batch_det_labels = detector_labels[start:start + CROP_BATCH_SIZE]
        batch_det_scores = detector_scores[start:start + CROP_BATCH_SIZE]
        crops = []
        for box in batch_boxes:
            x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
            x2, y2 = min(image_pil.width, int(box[2])), min(image_pil.height, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                crops.append(Image.new("RGB", (32, 32), (114, 114, 114)))
            else:
                crops.append(image_pil.crop((x1, y1, x2, y2)))

        tensors = torch.stack([transform(crop) for crop in crops]).to(device)
        autocast_on = device.type == "cuda"
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_on):
                embeddings = dino_model(tensors)
            embeddings = F.normalize(embeddings.float(), dim=-1)

            if linear_probe is not None:
                probe_logits = linear_probe(embeddings)
                probe_prob = F.softmax(probe_logits, dim=-1)
                # Build detector prior
                prior = torch.zeros((len(batch_boxes), NUM_CLASSES), device=device)
                for i in range(len(batch_boxes)):
                    prior[i, int(batch_det_labels[i])] = float(batch_det_scores[i])
                row_sum = prior.sum(dim=-1, keepdim=True)
                prior = prior / row_sum.clamp(min=1e-8)
                # Weighted combination: probe 0.85, detector prior 0.15
                combined = probe_prob * 0.85 + prior * 0.15
            else:
                # Fallback to detector labels
                combined = torch.zeros((len(batch_boxes), NUM_CLASSES), device=device)
                for i in range(len(batch_boxes)):
                    combined[i, int(batch_det_labels[i])] = 1.0

            batch_conf, batch_cls = combined.max(dim=-1)

        category_ids.extend(batch_cls.cpu().tolist())
        confidences.extend(batch_conf.cpu().tolist())

    return np.asarray(category_ids, np.int64), np.asarray(confidences, np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if "CUDAExecutionProvider" in ort.get_available_providers()
                 else ["CPUExecutionProvider"])

    session, input_name, input_shape = load_onnx_session(SCRIPT_DIR / MODEL_FILE, providers)
    dino_model, transform = load_dino(device)
    linear_probe = load_linear_probe(device)

    input_dir = Path(args.input)
    output_path = Path(args.output)
    image_paths = sorted(
        list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png"))
    )

    results = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        tensor, ratio, pad = preprocess_image(image_bgr, input_shape)
        outputs = session.run(None, {input_name: tensor})
        boxes, scores, labels = decode_yolo_output(outputs, ratio, pad, image_bgr.shape[:2])
        if len(boxes) == 0:
            continue

        category_ids, class_conf = classify_boxes(
            image_bgr, boxes, labels, scores, dino_model, transform, linear_probe, device
        )

        final_scores = np.clip(scores * (0.70 + 0.30 * class_conf), 0.0, 1.0)
        image_id = int(image_path.stem.replace("img_", ""))
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            results.append({
                "image_id": image_id,
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "category_id": int(category_ids[idx]),
                "score": round(float(final_scores[idx]), 4),
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

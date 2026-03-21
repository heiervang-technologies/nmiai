"""NM i AI 2026 - Object Detection: 2-Model WBF Ensemble + DINOv2 Classification
Runs two YOLO ONNX detectors, fuses with Weighted Boxes Fusion, then
classifies with DINOv2 ViT-S/14 linear probe.

Weight files (3):
  1. yolo_a.onnx  - primary YOLO detector (souped or best)
  2. yolo_b.onnx  - secondary YOLO detector (different arch/data)
  3. dino_with_probe.pth - DINOv2 weights + linear probe (combined)
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

# Model files
YOLO_A = "yolo_a.onnx"
YOLO_B = "yolo_b.onnx"
DINO_PROBE_WEIGHTS = "dino_with_probe.pth"

# Detection config
NUM_CLASSES = 356
CONF_THRESH = 0.001
NMS_IOU_THRESH = 0.45
WBF_IOU_THRESH = 0.60
MAX_DET = 300
CROP_BATCH_SIZE = 128

# Ensemble config
MODEL_WEIGHTS = [1.5, 1.0]  # V5 weighted higher (stronger model)
ENABLE_TTA_FLIP = True       # Horizontal flip TTA per model


# ─── YOLO helpers ────────────────────────────────────────────────────────────

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
    bordered = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=color)
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
    return boxes[valid], scores[valid], labels[valid]


# ─── WBF (inline, no external deps) ─────────────────────────────────────────

def wbf_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def weighted_boxes_fusion(boxes_list, scores_list, labels_list,
                          weights=None, iou_thr=0.55, skip_box_thr=0.0):
    """Weighted Boxes Fusion. Boxes must be normalized [0,1]."""
    n_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * n_models

    all_boxes = []
    for model_idx in range(n_models):
        for i in range(len(boxes_list[model_idx])):
            score = scores_list[model_idx][i]
            if score < skip_box_thr:
                continue
            all_boxes.append({
                "box": np.array(boxes_list[model_idx][i], dtype=np.float64),
                "score": float(score) * weights[model_idx],
                "label": int(labels_list[model_idx][i]),
                "model_idx": model_idx,
            })

    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    all_boxes.sort(key=lambda x: -x["score"])

    label_groups = {}
    for b in all_boxes:
        label_groups.setdefault(b["label"], []).append(b)

    fused_boxes, fused_scores, fused_labels = [], [], []

    for label, group_boxes in label_groups.items():
        clusters = []
        for b in group_boxes:
            matched = False
            for cluster in clusters:
                # IoU with current fused centroid of cluster
                total_w = sum(c["score"] for c in cluster)
                fused = np.zeros(4)
                for c in cluster:
                    fused += c["box"] * c["score"]
                fused /= total_w if total_w > 0 else 1.0
                if wbf_iou(b["box"], fused) > iou_thr:
                    cluster.append(b)
                    matched = True
                    break
            if not matched:
                clusters.append([b])

        for cluster in clusters:
            total_w = sum(c["score"] for c in cluster)
            if total_w == 0:
                continue
            fused_box = np.zeros(4)
            for c in cluster:
                fused_box += c["box"] * c["score"]
            fused_box /= total_w
            # box_and_model_avg: average over boxes found, not all models
            score = total_w / max(1, len(cluster))
            fused_boxes.append(fused_box)
            fused_scores.append(score)
            fused_labels.append(label)

    if not fused_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    scores_arr = np.array(fused_scores)
    order = np.argsort(scores_arr)[::-1]
    return (
        np.array(fused_boxes)[order],
        scores_arr[order],
        np.array(fused_labels, dtype=int)[order],
    )


# ─── Multi-model detection with WBF ─────────────────────────────────────────

def run_single_model(session, input_name, input_shape, image_bgr):
    """Run one YOLO model on an image, optionally with flip TTA."""
    tensor, ratio, pad = preprocess_image(image_bgr, input_shape)
    outputs = session.run(None, {input_name: tensor})
    boxes, scores, labels = decode_yolo_output(outputs, ratio, pad, image_bgr.shape[:2])

    if ENABLE_TTA_FLIP:
        flipped = cv2.flip(image_bgr, 1)
        tensor_f, ratio_f, pad_f = preprocess_image(flipped, input_shape)
        outputs_f = session.run(None, {input_name: tensor_f})
        boxes_f, scores_f, labels_f = decode_yolo_output(outputs_f, ratio_f, pad_f,
                                                          flipped.shape[:2])
        if len(boxes_f) > 0:
            img_w = image_bgr.shape[1]
            flipped_back = boxes_f.copy()
            flipped_back[:, 0] = img_w - boxes_f[:, 2]
            flipped_back[:, 2] = img_w - boxes_f[:, 0]
            boxes_f = flipped_back
            if len(boxes) > 0:
                boxes = np.concatenate([boxes, boxes_f])
                scores = np.concatenate([scores, scores_f])
                labels = np.concatenate([labels, labels_f])
            else:
                boxes, scores, labels = boxes_f, scores_f, labels_f
            # NMS on merged TTA output for this model
            boxes, scores, labels = nms_by_class(boxes, scores, labels, NMS_IOU_THRESH)

    return boxes, scores, labels


def detect_ensemble(models, image_bgr):
    """Run all models and fuse with WBF."""
    orig_h, orig_w = image_bgr.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []

    for session, input_name, input_shape in models:
        boxes, scores, labels = run_single_model(session, input_name, input_shape, image_bgr)
        # Normalize to [0, 1] for WBF
        if len(boxes) > 0:
            norm = boxes.copy()
            norm[:, [0, 2]] /= orig_w
            norm[:, [1, 3]] /= orig_h
            norm = np.clip(norm, 0, 1)
            all_boxes.append(norm.tolist())
            all_scores.append(scores.tolist())
            all_labels.append(labels.tolist())
        else:
            all_boxes.append([])
            all_scores.append([])
            all_labels.append([])

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=MODEL_WEIGHTS,
        iou_thr=WBF_IOU_THRESH,
        skip_box_thr=CONF_THRESH,
    )

    # Denormalize back to pixel coords
    if len(fused_boxes) > 0:
        fused_boxes[:, [0, 2]] *= orig_w
        fused_boxes[:, [1, 3]] *= orig_h

    # Limit detections
    if len(fused_scores) > MAX_DET:
        top = np.argsort(fused_scores)[::-1][:MAX_DET]
        fused_boxes = fused_boxes[top]
        fused_scores = fused_scores[top]
        fused_labels = fused_labels[top]

    return fused_boxes, fused_scores, fused_labels


# ─── DINOv2 classification ──────────────────────────────────────────────────

def load_dino_and_probe(device):
    """Load DINOv2 + linear probe from combined checkpoint."""
    combined = torch.load(SCRIPT_DIR / DINO_PROBE_WEIGHTS, map_location=device,
                          weights_only=True)

    # Separate probe keys from DINOv2 keys
    probe_state = {}
    dino_state = {}
    for k, v in combined.items():
        if k.startswith("probe."):
            probe_state[k[6:]] = v  # strip "probe." prefix
        else:
            dino_state[k] = v

    # Load DINOv2
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    model.load_state_dict(dino_state, strict=False)
    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)

    # Load linear probe
    num_classes, embed_dim = probe_state["weight"].shape
    probe = torch.nn.Linear(embed_dim, num_classes)
    probe.load_state_dict(probe_state)
    probe.eval().to(device)

    return model, transform, probe


def classify_boxes(image_bgr, boxes, detector_labels, detector_scores,
                   dino_model, transform, linear_probe, device):
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
            embeddings = embeddings.float()

            probe_logits = linear_probe(embeddings)
            probe_prob = F.softmax(probe_logits, dim=-1)
            # Build detector prior
            prior = torch.zeros((len(batch_boxes), NUM_CLASSES), device=device)
            for i in range(len(batch_boxes)):
                prior[i, int(batch_det_labels[i])] = float(batch_det_scores[i])
            row_sum = prior.sum(dim=-1, keepdim=True)
            prior = prior / row_sum.clamp(min=1e-8)
            combined = probe_prob * 0.85 + prior * 0.15

            batch_conf, batch_cls = combined.max(dim=-1)

        category_ids.extend(batch_cls.cpu().tolist())
        confidences.extend(batch_conf.cpu().tolist())

    return np.asarray(category_ids, np.int64), np.asarray(confidences, np.float32)


# ─── Main ────────────────────────────────────────────────────────────────────

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

    # Load YOLO models
    models = []
    for name in [YOLO_A, YOLO_B]:
        path = SCRIPT_DIR / name
        session = ort.InferenceSession(str(path), providers=providers)
        input_info = session.get_inputs()[0]
        input_shape = tuple(input_info.shape)
        models.append((session, input_info.name, input_shape))
    print(f"Loaded {len(models)} YOLO models")

    # Load DINOv2 + probe
    dino_model, transform, linear_probe = load_dino_and_probe(device)
    print("Loaded DINOv2 + linear probe")

    # Process images
    input_dir = Path(args.input)
    image_paths = sorted(
        list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
        + list(input_dir.glob("*.png"))
    )

    results = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        # Ensemble detection
        boxes, scores, labels = detect_ensemble(models, image_bgr)
        if len(boxes) == 0:
            continue

        # Classification
        category_ids, class_conf = classify_boxes(
            image_bgr, boxes, labels, scores,
            dino_model, transform, linear_probe, device
        )

        final_scores = np.clip(scores * (0.70 + 0.30 * class_conf), 0.0, 1.0)
        image_id = int(image_path.stem.replace("img_", ""))
        # Category aliases: merge umlaut spelling variants
        ALIASES = {59: 61, 170: 260, 36: 201}
        for idx, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            cat_id = int(category_ids[idx])
            cat_id = ALIASES.get(cat_id, cat_id)
            results.append({
                "image_id": image_id,
                "bbox": [round(float(x1), 2), round(float(y1), 2),
                         round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                "category_id": cat_id,
                "score": round(float(final_scores[idx]), 4),
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(results, f)
    print(f"Wrote {len(results)} detections to {output_path}")


if __name__ == "__main__":
    main()

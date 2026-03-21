"""NM i AI 2026 - Object Detection: YOLO + Dual Classifier Ensemble
Single YOLO detector + DINOv2 linear probe + MarkusNet vision probe.
Two independent classifiers, averaged for better classification accuracy.

Weight files (3):
  1. best.onnx          - YOLO detector (~132 MB)
  2. dino_with_probe.pth - DINOv2 ViT-S/14 + linear probe combined (~85 MB)
  3. markusnet_nf4.pt   - MarkusNet NF4 quantized (vision+text+cls) (~181 MB)
     OR markusnet_vision_int8.onnx + markusnet_probe.onnx (~86 MB)
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_data_config


SCRIPT_DIR = Path(__file__).parent

# Model files
YOLO_MODEL = "best.onnx"
DINO_PROBE_WEIGHTS = "dino_with_probe.pth"
# MarkusNet vision ONNX (probe weights embedded in dino_with_probe.pth)
MARKUS_VISION_ONNX = "markusnet_vision_int8.onnx"

# Detection config
NUM_CLASSES = 356
CONF_THRESH = 0.001
NMS_IOU_THRESH = 0.45
MAX_DET = 300
CROP_BATCH_SIZE = 128

# Classifier blending weights
DINO_WEIGHT = 0.6       # DINOv2 is more reliable (92% mAP)
MARKUS_WEIGHT = 0.4     # MarkusNet adds diversity (89.7% accuracy)
DETECTOR_PRIOR = 0.10   # Small weight for detector's own class prediction
CLASSIFIER_WEIGHT = 0.90  # Combined classifier weight (1 - DETECTOR_PRIOR)

# MarkusNet vision preprocessing
QWEN_IMAGE_SIZE = 448
QWEN_MEAN = [0.48145466, 0.4578275, 0.40821073]
QWEN_STD = [0.26862954, 0.26130258, 0.27577711]
VIS_PATCH_SIZE = 16
VIS_SPATIAL_MERGE = 2
FACTOR = VIS_PATCH_SIZE * VIS_SPATIAL_MERGE  # 32


# ─── YOLO Detection ─────────────────────────────────────────────────────────

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


def preprocess_yolo(image_bgr, input_shape):
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
    return boxes, scores, labels


def detect_with_tta(session, input_name, input_shape, image_bgr):
    """Run YOLO detection with horizontal flip TTA."""
    tensor, ratio, pad = preprocess_yolo(image_bgr, input_shape)
    outputs = session.run(None, {input_name: tensor})
    boxes, scores, labels = decode_yolo_output(outputs, ratio, pad, image_bgr.shape[:2])

    # Horizontal flip TTA
    flipped = cv2.flip(image_bgr, 1)
    tensor_f, ratio_f, pad_f = preprocess_yolo(flipped, input_shape)
    outputs_f = session.run(None, {input_name: tensor_f})
    boxes_f, scores_f, labels_f = decode_yolo_output(outputs_f, ratio_f, pad_f, flipped.shape[:2])
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
        boxes, scores, labels = nms_by_class(boxes, scores, labels, NMS_IOU_THRESH)
        if len(scores) > MAX_DET:
            top = np.argsort(scores)[::-1][:MAX_DET]
            boxes, scores, labels = boxes[top], scores[top], labels[top]
    return boxes, scores, labels


# ─── DINOv2 Classifier ──────────────────────────────────────────────────────

def load_dino_and_probe(device):
    combined = torch.load(SCRIPT_DIR / DINO_PROBE_WEIGHTS, map_location=device,
                          weights_only=True)
    probe_state = {}
    dino_state = {}
    markus_probe_state = {}
    for k, v in combined.items():
        if k.startswith("probe."):
            probe_state[k[6:]] = v
        elif k.startswith("markus_probe."):
            markus_probe_state[k[13:]] = v
        else:
            dino_state[k] = v

    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    model.load_state_dict(dino_state, strict=False)
    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)

    num_classes, embed_dim = probe_state["weight"].shape
    probe = torch.nn.Linear(embed_dim, num_classes)
    probe.load_state_dict(probe_state)
    probe.eval().to(device)

    # MarkusNet probe (simple linear layer)
    markus_linear = None
    if markus_probe_state:
        m_nc, m_dim = markus_probe_state["weight"].shape
        markus_linear = torch.nn.Linear(m_dim, m_nc)
        markus_linear.load_state_dict(markus_probe_state)
        markus_linear.eval().to(device)

    return model, transform, probe, markus_linear


def dino_classify(crops_pil, dino_model, transform, probe, device):
    """Classify crops using DINOv2 + linear probe. Returns (N, NUM_CLASSES) probs."""
    if not crops_pil:
        return np.empty((0, NUM_CLASSES), dtype=np.float32)

    all_probs = []
    for start in range(0, len(crops_pil), CROP_BATCH_SIZE):
        batch = crops_pil[start:start + CROP_BATCH_SIZE]
        tensors = torch.stack([transform(crop) for crop in batch]).to(device)
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=device.type == "cuda"):
                embeddings = dino_model(tensors)
            embeddings = F.normalize(embeddings.float(), dim=-1)
            logits = probe(embeddings)
            probs = F.softmax(logits, dim=-1)
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs, axis=0)


# ─── MarkusNet ONNX Classifier ──────────────────────────────────────────────

def preprocess_markusnet_crop(pil_img):
    """Preprocess a crop for MarkusNet vision encoder (ONNX).
    Letterbox pad + normalize, then extract patches.
    """
    img = pil_img.convert("RGB")
    orig_w, orig_h = img.size

    # Smart resize: pad to QWEN_IMAGE_SIZE maintaining aspect ratio
    target = QWEN_IMAGE_SIZE
    scale = min(target / orig_h, target / orig_w)
    new_w = max(FACTOR, round(orig_w * scale / FACTOR) * FACTOR)
    new_h = max(FACTOR, round(orig_h * scale / FACTOR) * FACTOR)

    # Clamp to max
    max_pixels = target * target
    if new_h * new_w > max_pixels:
        beta = math.sqrt((orig_h * orig_w) / max_pixels)
        new_h = max(FACTOR, math.floor(orig_h / beta / FACTOR) * FACTOR)
        new_w = max(FACTOR, math.floor(orig_w / beta / FACTOR) * FACTOR)

    img = img.resize((new_w, new_h), Image.BICUBIC)

    # To numpy, normalize
    arr = np.array(img, dtype=np.float32) / 255.0
    for c in range(3):
        arr[:, :, c] = (arr[:, :, c] - QWEN_MEAN[c]) / QWEN_STD[c]

    # Extract patches: (h_patches, w_patches, patch_h, patch_w, 3)
    h_patches = new_h // VIS_PATCH_SIZE
    w_patches = new_w // VIS_PATCH_SIZE
    t_patches = 1  # single image

    # Reshape to patches
    arr = arr.reshape(h_patches, VIS_PATCH_SIZE, w_patches, VIS_PATCH_SIZE, 3)
    arr = arr.transpose(0, 2, 1, 3, 4)  # (h_patches, w_patches, patch_h, patch_w, 3)
    patches = arr.reshape(-1, VIS_PATCH_SIZE * VIS_PATCH_SIZE * 3)  # (num_patches, 768)

    grid_thw = np.array([[t_patches, h_patches, w_patches]], dtype=np.int64)

    return patches[np.newaxis, ...].astype(np.float32), grid_thw


def load_markusnet_onnx(providers):
    """Load MarkusNet ONNX vision encoder."""
    vision_path = SCRIPT_DIR / MARKUS_VISION_ONNX
    if not vision_path.exists():
        return None
    return ort.InferenceSession(str(vision_path), providers=providers)


def markusnet_classify(crops_pil, vision_session, markus_probe, device):
    """Classify crops using MarkusNet ONNX vision + PyTorch probe. Returns (N, NUM_CLASSES) probs."""
    if not crops_pil or vision_session is None or markus_probe is None:
        return None

    all_probs = []
    for crop in crops_pil:
        try:
            pixel_values, grid_thw = preprocess_markusnet_crop(crop)

            # Vision encoder
            vision_input = vision_session.get_inputs()[0].name
            grid_input = vision_session.get_inputs()[1].name if len(vision_session.get_inputs()) > 1 else None

            inputs = {vision_input: pixel_values}
            if grid_input:
                inputs[grid_input] = grid_thw

            features = vision_session.run(None, inputs)[0]  # (1, num_tokens, hidden)

            # Pool features (mean over tokens)
            if features.ndim == 3:
                pooled = features.mean(axis=1)  # (1, hidden)
            else:
                pooled = features

            # PyTorch probe
            with torch.inference_mode():
                feat_tensor = torch.from_numpy(pooled.astype(np.float32)).to(device)
                logits = markus_probe(feat_tensor)
                probs = F.softmax(logits, dim=-1)
            all_probs.append(probs[0].cpu().numpy())
        except Exception:
            all_probs.append(np.ones(NUM_CLASSES, dtype=np.float32) / NUM_CLASSES)

    return np.array(all_probs, dtype=np.float32)


# ─── Ensemble Classification ────────────────────────────────────────────────

def ensemble_classify(crops_pil, det_labels, det_scores,
                      dino_model, dino_transform, dino_probe,
                      markus_vision, markus_probe_linear,
                      device):
    """Run both classifiers and average their predictions."""
    if len(crops_pil) == 0:
        return np.empty(0, np.int64), np.empty(0, np.float32)

    # DINOv2 classification
    dino_probs = dino_classify(crops_pil, dino_model, dino_transform, dino_probe, device)

    # MarkusNet classification (may be None if ONNX not available)
    markus_probs = markusnet_classify(crops_pil, markus_vision, markus_probe_linear, device)

    # Build detector prior
    prior = np.zeros((len(crops_pil), NUM_CLASSES), dtype=np.float32)
    for i in range(len(crops_pil)):
        prior[i, int(det_labels[i])] = float(det_scores[i])
    prior_sum = prior.sum(axis=-1, keepdims=True)
    prior = prior / np.maximum(prior_sum, 1e-8)

    # Combine classifiers
    if markus_probs is not None:
        # Weighted average of both classifiers + detector prior
        combined = (CLASSIFIER_WEIGHT * (
            DINO_WEIGHT * dino_probs + MARKUS_WEIGHT * markus_probs
        ) + DETECTOR_PRIOR * prior)
    else:
        # DINOv2 only + detector prior
        combined = CLASSIFIER_WEIGHT * dino_probs + DETECTOR_PRIOR * prior

    category_ids = np.argmax(combined, axis=-1).astype(np.int64)
    confidences = np.max(combined, axis=-1).astype(np.float32)

    return category_ids, confidences


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

    # Load YOLO
    yolo_path = SCRIPT_DIR / YOLO_MODEL
    yolo_session = ort.InferenceSession(str(yolo_path), providers=providers)
    yolo_input = yolo_session.get_inputs()[0]
    yolo_input_name = yolo_input.name
    yolo_input_shape = tuple(yolo_input.shape)
    print(f"YOLO loaded: {yolo_path.name}")

    # Load DINOv2 + probe + MarkusNet probe
    dino_model, dino_transform, dino_probe, markus_probe_linear = load_dino_and_probe(device)
    print("DINOv2 + probe loaded")
    if markus_probe_linear is not None:
        print("MarkusNet probe loaded from combined weights")

    # Load MarkusNet vision ONNX (optional)
    markus_vision = load_markusnet_onnx(providers)
    if markus_vision is not None:
        print("MarkusNet vision ONNX loaded (dual classifier mode)")
    else:
        print("MarkusNet vision ONNX not found (DINOv2 only mode)")

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

        # Detect with TTA
        boxes, scores, labels = detect_with_tta(
            yolo_session, yolo_input_name, yolo_input_shape, image_bgr
        )
        if len(boxes) == 0:
            continue

        # Extract crops
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)
        crops = []
        for box in boxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(image_pil.width, int(box[2]))
            y2 = min(image_pil.height, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                crops.append(Image.new("RGB", (32, 32), (114, 114, 114)))
            else:
                crops.append(image_pil.crop((x1, y1, x2, y2)))

        # Dual classifier ensemble
        category_ids, class_conf = ensemble_classify(
            crops, labels, scores,
            dino_model, dino_transform, dino_probe,
            markus_vision, markus_probe_linear,
            device
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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(results, f)
    print(f"Wrote {len(results)} detections to {output_path}")


if __name__ == "__main__":
    main()

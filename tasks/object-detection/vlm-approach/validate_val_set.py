"""
Validate models against the clean held-out validation set.

Evaluates:
1. DINOv2 + linear probe classification accuracy (using GT boxes)
2. YOLO ONNX ensemble detection mAP@0.5 (class-agnostic)
3. Full pipeline: YOLO detection + DINOv2 classification mAP@0.5

Self-contained -- does not import run_dense_retrieval (avoids unsloth compile overhead).

Usage:
    CUDA_VISIBLE_DEVICES=0 uv run python validate_val_set.py
"""

import gc
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import timm
import torch
import torch.nn.functional as F
from PIL import Image
from timm.data import create_transform, resolve_data_config

# Paths
SCRIPT_DIR = Path(__file__).parent
VAL_ROOT = SCRIPT_DIR.parent / "data-creation" / "data" / "clean_split" / "val"
VAL_IMAGES = VAL_ROOT / "images"
VAL_LABELS = VAL_ROOT / "labels"
YOLO_DIR = SCRIPT_DIR.parent / "titan-models"

NUM_CLASSES = 356
CONF_THRESH = 0.001
NMS_IOU_THRESH = 0.70
WBF_IOU_THRESH = 0.60
SOURCE_MATCH_IOU = 0.55
MAX_DET = 300
IOU_THRESH = 0.5


# ============================================================================
# Ground Truth Loading
# ============================================================================

def load_ground_truth():
    """Load YOLO format ground truth labels."""
    gt = {}
    label_files = sorted(VAL_LABELS.glob("*.txt"))
    for label_path in label_files:
        image_name = label_path.stem + ".jpg"
        image_path = VAL_IMAGES / image_name
        if not image_path.exists():
            image_path = VAL_IMAGES / (label_path.stem + ".png")
            if not image_path.exists():
                continue
            image_name = label_path.stem + ".png"

        img = cv2.imread(str(image_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        boxes = []
        for line in label_path.read_text().strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = max(0, (cx - bw / 2) * w)
            y1 = max(0, (cy - bh / 2) * h)
            x2 = min(w, (cx + bw / 2) * w)
            y2 = min(h, (cy + bh / 2) * h)
            if x2 > x1 and y2 > y1:
                boxes.append((cls_id, x1, y1, x2, y2))
        gt[image_name] = boxes
    return gt


# ============================================================================
# YOLO Detection Utilities (copied from run_dense_retrieval to avoid imports)
# ============================================================================

def load_onnx_session(model_path, providers):
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


def compute_iou_single(box, boxes):
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
            iou = compute_iou_single(class_boxes[current], class_boxes[order[1:]])
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
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
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
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    boxes, scores, labels = nms_by_class(boxes, scores, labels, NMS_IOU_THRESH)
    if len(scores) > MAX_DET:
        top = np.argsort(scores)[::-1][:MAX_DET]
        boxes, scores, labels = boxes[top], scores[top], labels[top]
    return boxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int64)


def weighted_boxes_fusion(boxes_list, scores_list, labels_list, iou_thr, skip_box_thr):
    weighted = []
    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        for idx in range(len(boxes)):
            if scores[idx] < skip_box_thr:
                continue
            weighted.append({"box": boxes[idx].copy(), "score": float(scores[idx]), "label": int(labels[idx])})
    if not weighted:
        return np.empty((0, 4), dtype=np.float32), np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)
    weighted.sort(key=lambda e: e["score"], reverse=True)
    grouped = {}
    for entry in weighted:
        clusters = grouped.setdefault(entry["label"], [])
        matched = False
        for cluster in clusters:
            fused_box = np.zeros(4, dtype=np.float32)
            total_score = 0.0
            for member in cluster:
                fused_box += member["box"] * member["score"]
                total_score += member["score"]
            fused_box /= max(total_score, 1e-8)
            if compute_iou_single(entry["box"], fused_box[None, :])[0] > iou_thr:
                cluster.append(entry)
                matched = True
                break
        if not matched:
            clusters.append([entry])
    fused_boxes, fused_scores, fused_labels = [], [], []
    model_count = max(1, len(boxes_list))
    for label, clusters in grouped.items():
        for cluster in clusters:
            box = np.zeros(4, dtype=np.float32)
            total_score = 0.0
            for member in cluster:
                box += member["box"] * member["score"]
                total_score += member["score"]
            box /= max(total_score, 1e-8)
            fused_boxes.append(box)
            fused_scores.append(total_score / model_count)
            fused_labels.append(label)
    order = np.argsort(np.asarray(fused_scores))[::-1][:MAX_DET]
    return (
        np.asarray(fused_boxes, dtype=np.float32)[order],
        np.asarray(fused_scores, dtype=np.float32)[order],
        np.asarray(fused_labels, dtype=np.int64)[order],
    )


def build_detector_prior(fused_boxes, source_boxes, source_scores, source_labels):
    prior = np.zeros((len(fused_boxes), NUM_CLASSES), dtype=np.float32)
    if len(fused_boxes) == 0 or len(source_boxes) == 0:
        return prior
    for idx, fused_box in enumerate(fused_boxes):
        iou = compute_iou_single(fused_box, source_boxes)
        matches = np.where(iou >= SOURCE_MATCH_IOU)[0]
        if len(matches) == 0:
            continue
        for match in matches:
            prior[idx, source_labels[match]] += source_scores[match]
        row_sum = prior[idx].sum()
        if row_sum > 0:
            prior[idx] /= row_sum
    return prior


# ============================================================================
# DINOv2 Loading and Classification
# ============================================================================

def load_dino(device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    state_dict = torch.load(SCRIPT_DIR / "dinov2_vits14.pth", map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)
    return model, transform


def load_linear_probe(device):
    probe_path = SCRIPT_DIR / "linear_probe.pth"
    if not probe_path.exists():
        return None
    state = torch.load(probe_path, map_location=device, weights_only=False)
    if not isinstance(state, dict) or "weight" not in state:
        return None
    num_classes, embed_dim = state["weight"].shape
    probe = torch.nn.Linear(embed_dim, num_classes)
    probe.load_state_dict(state)
    return probe.eval().to(device)


def load_embedding_bank(device):
    centroids = torch.zeros((NUM_CLASSES, 384), dtype=torch.float32, device=device)
    centroid_mask = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=device)
    ref_embeddings = []
    ref_categories = []

    data_bank_path = SCRIPT_DIR.parent / "data-creation" / "data" / "ref_embeddings.pth"
    if data_bank_path.exists():
        data_bank = torch.load(data_bank_path, map_location=device, weights_only=False)
        category_embeddings = F.normalize(data_bank["category_embeddings"].float().to(device), dim=-1)
        category_ids = data_bank.get("category_ids", list(range(category_embeddings.shape[0])))
        for idx, category_id in enumerate(category_ids):
            category_id = int(category_id)
            centroids[category_id] = category_embeddings[idx]
            centroid_mask[category_id] = True
        reference_embeddings = F.normalize(data_bank["reference_embeddings"].float().to(device), dim=-1)
        barcode_to_category = data_bank.get("barcode_to_category", {})
        reference_barcodes = data_bank.get("reference_barcodes", [])
        for idx, barcode in enumerate(reference_barcodes):
            barcode_key = str(barcode)
            if barcode_key not in barcode_to_category:
                continue
            ref_embeddings.append(reference_embeddings[idx])
            ref_categories.append(int(barcode_to_category[barcode_key]))

    multi_bank_path = SCRIPT_DIR / "ref_embeddings.pth"
    if multi_bank_path.exists():
        multi_bank = torch.load(multi_bank_path, map_location=device, weights_only=False)
        embedding_dict = multi_bank.get("embeddings", multi_bank)
        for category_id, embedding in embedding_dict.items():
            category_id = int(category_id)
            if not isinstance(embedding, torch.Tensor):
                continue
            embedding = F.normalize(embedding.float().to(device), dim=-1)
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)
            if not centroid_mask[category_id]:
                centroids[category_id] = F.normalize(embedding.mean(dim=0), dim=-1)
                centroid_mask[category_id] = True
            for row in embedding:
                ref_embeddings.append(row)
                ref_categories.append(category_id)

    if ref_embeddings:
        ref_matrix = torch.stack(ref_embeddings)
        ref_cat_ids = torch.tensor(ref_categories, device=device, dtype=torch.long)
    else:
        ref_matrix = torch.empty((0, 384), device=device)
        ref_cat_ids = torch.empty(0, device=device, dtype=torch.long)

    return centroids, centroid_mask, ref_matrix, ref_cat_ids


def build_ref_similarity(embeddings, ref_matrix, ref_cat_ids):
    batch_size = embeddings.shape[0]
    ref_scores = embeddings.new_full((batch_size, NUM_CLASSES), -20.0)
    if ref_matrix.numel() == 0:
        return ref_scores
    similarities = embeddings @ ref_matrix.T
    for category_id in range(NUM_CLASSES):
        mask = ref_cat_ids == category_id
        if torch.any(mask):
            ref_scores[:, category_id] = similarities[:, mask].max(dim=1).values
    return ref_scores


# ============================================================================
# mAP Utilities
# ============================================================================

def compute_iou_matrix(boxes_a, boxes_b):
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    x1 = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0:1].T)
    y1 = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1:2].T)
    x2 = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2:3].T)
    y2 = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3:4].T)
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def compute_ap(recalls, precisions):
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for rp in recall_points:
        precs = mpre[mrec >= rp]
        if len(precs) > 0:
            ap += precs.max()
    return ap / 101.0


# ============================================================================
# 1. DINOv2 Classification Evaluation
# ============================================================================

def eval_dino_classification(gt):
    print("=" * 60)
    print("DINOv2 + Linear Probe Classification (GT boxes)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model, transform = load_dino(device)
    probe = load_linear_probe(device)
    centroids, centroid_mask, ref_matrix, ref_cat_ids = load_embedding_bank(device)

    BATCH_SIZE = 128
    all_crops = []
    all_labels = []

    print("Cropping GT boxes...")
    for image_name, boxes in sorted(gt.items()):
        image_path = VAL_IMAGES / image_name
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        for cls_id, x1, y1, x2, y2 in boxes:
            crop = img_pil.crop((int(x1), int(y1), int(x2), int(y2)))
            if crop.width < 2 or crop.height < 2:
                continue
            all_crops.append(crop)
            all_labels.append(cls_id)

    print(f"Total crops: {len(all_crops)}")

    all_probe_preds = []
    all_centroid_preds = []
    all_combined_preds = []

    for start in range(0, len(all_crops), BATCH_SIZE):
        batch_crops = all_crops[start:start + BATCH_SIZE]
        tensors = torch.stack([transform(c) for c in batch_crops]).to(device)

        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            embeddings = model(tensors)
        embeddings = F.normalize(embeddings.float(), dim=-1)

        with torch.inference_mode():
            probe_logits = probe(embeddings)
            all_probe_preds.extend(probe_logits.argmax(dim=-1).cpu().tolist())

            centroid_logits = embeddings @ centroids.T
            centroid_logits[:, ~centroid_mask] = -20.0
            all_centroid_preds.extend(centroid_logits.argmax(dim=-1).cpu().tolist())

            centroid_prob = F.softmax(centroid_logits / 0.12, dim=-1)
            ref_logits = build_ref_similarity(embeddings, ref_matrix, ref_cat_ids)
            ref_prob = F.softmax(ref_logits / 0.10, dim=-1)
            probe_prob = F.softmax(probe_logits, dim=-1)
            combined = centroid_prob * 0.45 + ref_prob * 0.35 + probe_prob * 0.15
            all_combined_preds.extend(combined.argmax(dim=-1).cpu().tolist())

        if (start + BATCH_SIZE) % 512 == 0 or start + BATCH_SIZE >= len(all_crops):
            print(f"  Processed {min(start + BATCH_SIZE, len(all_crops))}/{len(all_crops)} crops")

    total = len(all_labels)
    correct_probe = sum(1 for i in range(total) if all_probe_preds[i] == all_labels[i])
    correct_centroid = sum(1 for i in range(total) if all_centroid_preds[i] == all_labels[i])
    correct_combined = sum(1 for i in range(total) if all_combined_preds[i] == all_labels[i])

    per_class_total = defaultdict(int)
    per_class_correct = defaultdict(int)
    for i in range(total):
        per_class_total[all_labels[i]] += 1
        if all_combined_preds[i] == all_labels[i]:
            per_class_correct[all_labels[i]] += 1

    print(f"\nResults ({total} crops from {len(gt)} images):")
    print(f"  Linear Probe only:      {correct_probe}/{total} = {correct_probe/max(total,1):.4f}")
    print(f"  Centroid matching only:  {correct_centroid}/{total} = {correct_centroid/max(total,1):.4f}")
    print(f"  Combined pipeline:       {correct_combined}/{total} = {correct_combined/max(total,1):.4f}")

    per_class_accs = [per_class_correct[c] / per_class_total[c] for c in per_class_total]
    mean_per_class = np.mean(per_class_accs) if per_class_accs else 0.0

    classes_perfect = sum(1 for c in per_class_total if per_class_correct[c] == per_class_total[c])
    classes_zero = sum(1 for c in per_class_total if per_class_correct[c] == 0)

    print(f"\n  Mean per-class accuracy: {mean_per_class:.4f}")
    print(f"  Classes with samples: {len(per_class_total)}")
    print(f"  Classes with 100% acc: {classes_perfect}")
    print(f"  Classes with 0% acc: {classes_zero}")

    class_acc_list = [(c, per_class_correct[c] / per_class_total[c], per_class_total[c])
                      for c in per_class_total if per_class_total[c] >= 3]
    class_acc_list.sort(key=lambda x: x[1])
    print(f"\n  Worst classes (>= 3 samples):")
    for cls_id, acc, cnt in class_acc_list[:10]:
        print(f"    Class {cls_id}: {acc:.2%} ({per_class_correct[cls_id]}/{cnt})")

    # Free GPU memory
    del model, probe, centroids, centroid_mask, ref_matrix, ref_cat_ids
    torch.cuda.empty_cache()
    gc.collect()

    return correct_combined / max(total, 1)


# ============================================================================
# 2. YOLO Detection mAP
# ============================================================================

def run_yolo_ensemble(img, yolo_models):
    """Run 2-model YOLO ensemble with flip TTA and WBF fusion."""
    all_boxes_list, all_scores_list, all_labels_list = [], [], []
    source_boxes_list, source_scores_list, source_labels_list = [], [], []

    for session, input_name, input_shape in yolo_models:
        for flip in (False, True):
            source = cv2.flip(img, 1) if flip else img
            tensor, ratio, pad = preprocess_image(source, input_shape)
            outputs = session.run(None, {input_name: tensor})
            boxes, scores, labels = decode_yolo_output(outputs, ratio, pad, source.shape[:2])
            if flip and len(boxes) > 0:
                flipped = boxes.copy()
                flipped[:, 0] = img.shape[1] - boxes[:, 2]
                flipped[:, 2] = img.shape[1] - boxes[:, 0]
                boxes = flipped
            if len(boxes) == 0:
                continue
            norm_boxes = boxes.copy()
            norm_boxes[:, [0, 2]] /= img.shape[1]
            norm_boxes[:, [1, 3]] /= img.shape[0]
            all_boxes_list.append(norm_boxes)
            all_scores_list.append(scores)
            all_labels_list.append(np.zeros(len(labels), dtype=np.int64))
            source_boxes_list.append(boxes)
            source_scores_list.append(scores)
            source_labels_list.append(labels)

    if not all_boxes_list:
        empty = np.empty((0, 4), dtype=np.float32)
        return empty, np.empty(0, dtype=np.float32), empty, np.empty(0, dtype=np.float32), np.empty(0, dtype=np.int64)

    fused_boxes, fused_scores, _ = weighted_boxes_fusion(
        all_boxes_list, all_scores_list, all_labels_list,
        iou_thr=WBF_IOU_THRESH, skip_box_thr=CONF_THRESH
    )
    fused_boxes[:, [0, 2]] *= img.shape[1]
    fused_boxes[:, [1, 3]] *= img.shape[0]

    stacked_src_boxes = np.concatenate(source_boxes_list) if source_boxes_list else np.empty((0, 4), dtype=np.float32)
    stacked_src_scores = np.concatenate(source_scores_list) if source_scores_list else np.empty(0, dtype=np.float32)
    stacked_src_labels = np.concatenate(source_labels_list) if source_labels_list else np.empty(0, dtype=np.int64)

    return fused_boxes, fused_scores, stacked_src_boxes, stacked_src_scores, stacked_src_labels


def eval_detection_map(gt, yolo_models):
    print("\n" + "=" * 60)
    print("YOLO ONNX Ensemble Detection mAP@0.5")
    print("=" * 60)

    all_det_scores = []
    total_gt = 0

    for idx, (image_name, gt_boxes_raw) in enumerate(sorted(gt.items())):
        image_path = VAL_IMAGES / image_name
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        gt_xyxy = np.array([[x1, y1, x2, y2] for _, x1, y1, x2, y2 in gt_boxes_raw], dtype=np.float32)
        total_gt += len(gt_xyxy)

        fused_boxes, fused_scores, _, _, _ = run_yolo_ensemble(img, yolo_models)

        if len(fused_boxes) == 0 or len(gt_xyxy) == 0:
            continue

        iou_matrix = compute_iou_matrix(fused_boxes, gt_xyxy)
        gt_matched = np.zeros(len(gt_xyxy), dtype=bool)
        order = np.argsort(fused_scores)[::-1]

        for det_idx in order:
            best_gt = iou_matrix[det_idx].argmax()
            if iou_matrix[det_idx, best_gt] >= IOU_THRESH and not gt_matched[best_gt]:
                gt_matched[best_gt] = True
                all_det_scores.append((float(fused_scores[det_idx]), True))
            else:
                all_det_scores.append((float(fused_scores[det_idx]), False))

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx+1}/{len(gt)} images")

    all_det_scores.sort(key=lambda x: x[0], reverse=True)
    tp = fp = 0
    recalls, precisions = [], []
    for score, is_tp in all_det_scores:
        if is_tp:
            tp += 1
        else:
            fp += 1
        recalls.append(tp / max(total_gt, 1))
        precisions.append(tp / (tp + fp))

    agnostic_ap = compute_ap(np.array(recalls), np.array(precisions))

    print(f"\nDetection Results:")
    print(f"  Total GT boxes: {total_gt}")
    print(f"  Total detections: {len(all_det_scores)}")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  Recall: {tp/max(total_gt,1):.4f}")
    print(f"  Class-agnostic mAP@0.5: {agnostic_ap:.4f}")
    return agnostic_ap


# ============================================================================
# 3. Full Pipeline mAP (detection + classification)
# ============================================================================

def classify_crops_dino(img, boxes, detector_prior, dino_model, transform, centroids,
                        centroid_mask, ref_matrix, ref_cat_ids, linear_probe, device):
    """Classify detected boxes using DINOv2 ensemble."""
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    category_ids = []
    confidences = []
    BATCH_SIZE = 128

    for start in range(0, len(boxes), BATCH_SIZE):
        batch_boxes = boxes[start:start + BATCH_SIZE]
        crops = []
        for box in batch_boxes:
            x1, y1, x2, y2 = max(0, int(box[0])), max(0, int(box[1])), min(image_pil.width, int(box[2])), min(image_pil.height, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                crops.append(Image.new("RGB", (32, 32), (114, 114, 114)))
            else:
                crops.append(image_pil.crop((x1, y1, x2, y2)))

        tensors = torch.stack([transform(crop) for crop in crops]).to(device)
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            embeddings = dino_model(tensors)
        embeddings = F.normalize(embeddings.float(), dim=-1)

        with torch.inference_mode():
            centroid_logits = embeddings @ centroids.T
            centroid_logits[:, ~centroid_mask] = -20.0
            centroid_prob = F.softmax(centroid_logits / 0.12, dim=-1)

            ref_logits = build_ref_similarity(embeddings, ref_matrix, ref_cat_ids)
            ref_prob = F.softmax(ref_logits / 0.10, dim=-1)

            combined = centroid_prob * 0.45 + ref_prob * 0.35
            weight_total = torch.full((len(batch_boxes), 1), 0.80, device=device)

            if linear_probe is not None:
                probe_prob = F.softmax(linear_probe(embeddings), dim=-1)
                combined += probe_prob * 0.15
                weight_total += 0.15

            if detector_prior.size > 0:
                prior_tensor = torch.from_numpy(detector_prior[start:start + len(batch_boxes)]).to(device)
                prior_rows = (prior_tensor.sum(dim=-1, keepdim=True) > 0).float()
                if torch.any(prior_rows > 0):
                    combined += prior_tensor * 0.10
                    weight_total += prior_rows * 0.10

            combined = combined / weight_total
            batch_conf, batch_cls = combined.max(dim=-1)

        category_ids.extend(batch_cls.cpu().tolist())
        confidences.extend(batch_conf.cpu().tolist())

    return np.asarray(category_ids, dtype=np.int64), np.asarray(confidences, dtype=np.float32)


def eval_full_pipeline_map(gt, yolo_models, dino_model, transform, linear_probe,
                           centroids, centroid_mask, ref_matrix, ref_cat_ids, device):
    print("\n" + "=" * 60)
    print("Full Pipeline: YOLO + DINOv2 Classification mAP@0.5")
    print("=" * 60)

    per_class_dets = defaultdict(list)
    per_class_gt_count = defaultdict(int)

    for idx, (image_name, gt_boxes_raw) in enumerate(sorted(gt.items())):
        image_path = VAL_IMAGES / image_name
        img = cv2.imread(str(image_path))
        if img is None:
            continue

        gt_xyxy = np.array([[x1, y1, x2, y2] for _, x1, y1, x2, y2 in gt_boxes_raw], dtype=np.float32)
        gt_classes = np.array([c for c, _, _, _, _ in gt_boxes_raw], dtype=np.int64)
        for c in gt_classes:
            per_class_gt_count[int(c)] += 1

        fused_boxes, fused_scores, src_boxes, src_scores, src_labels = run_yolo_ensemble(img, yolo_models)

        if len(fused_boxes) == 0:
            continue

        detector_prior = build_detector_prior(fused_boxes, src_boxes, src_scores, src_labels)
        pred_classes, class_conf = classify_crops_dino(
            img, fused_boxes, detector_prior,
            dino_model, transform, centroids, centroid_mask,
            ref_matrix, ref_cat_ids, linear_probe, device
        )

        final_scores = np.clip(fused_scores * (0.70 + 0.30 * class_conf), 0.0, 1.0)

        if len(gt_xyxy) == 0:
            for d in range(len(fused_boxes)):
                per_class_dets[int(pred_classes[d])].append((float(final_scores[d]), False))
            continue

        iou_matrix = compute_iou_matrix(fused_boxes, gt_xyxy)
        gt_matched = np.zeros(len(gt_xyxy), dtype=bool)
        order = np.argsort(final_scores)[::-1]

        for det_idx in order:
            pred_cls = int(pred_classes[det_idx])
            best_iou = 0
            best_gt = -1
            for gt_idx in range(len(gt_xyxy)):
                if gt_matched[gt_idx]:
                    continue
                if int(gt_classes[gt_idx]) != pred_cls:
                    continue
                if iou_matrix[det_idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[det_idx, gt_idx]
                    best_gt = gt_idx

            if best_gt >= 0 and best_iou >= IOU_THRESH:
                gt_matched[best_gt] = True
                per_class_dets[pred_cls].append((float(final_scores[det_idx]), True))
            else:
                per_class_dets[pred_cls].append((float(final_scores[det_idx]), False))

        if (idx + 1) % 20 == 0:
            print(f"  Processed {idx+1}/{len(gt)} images")

    aps = []
    for cls_id in sorted(per_class_gt_count.keys()):
        n_gt = per_class_gt_count[cls_id]
        dets = per_class_dets.get(cls_id, [])
        if n_gt == 0:
            continue
        dets.sort(key=lambda x: x[0], reverse=True)
        tp = fp = 0
        recalls, precisions = [], []
        for score, is_tp in dets:
            if is_tp:
                tp += 1
            else:
                fp += 1
            recalls.append(tp / n_gt)
            precisions.append(tp / (tp + fp))
        aps.append(compute_ap(np.array(recalls), np.array(precisions)) if recalls else 0.0)

    map_score = np.mean(aps) if aps else 0.0
    total_gt = sum(per_class_gt_count.values())
    total_tp = sum(1 for dets in per_class_dets.values() for _, is_tp in dets if is_tp)

    print(f"\nFull Pipeline Results:")
    print(f"  Classes evaluated: {len(aps)}")
    print(f"  Total GT boxes: {total_gt}")
    print(f"  Total TP (correct class + IoU >= 0.5): {total_tp}")
    print(f"  Per-class mAP@0.5: {map_score:.4f}")
    return map_score


# ============================================================================
# Main
# ============================================================================

def main():
    t0 = time.time()
    sys.stdout.reconfigure(line_buffering=True)

    print("Loading ground truth...")
    gt = load_ground_truth()
    total_boxes = sum(len(v) for v in gt.values())
    all_classes = set()
    for boxes in gt.values():
        for cls_id, *_ in boxes:
            all_classes.add(cls_id)
    print(f"Loaded {len(gt)} images with {total_boxes} GT boxes, {len(all_classes)} unique classes")

    # 1. DINOv2 classification
    cls_acc = eval_dino_classification(gt)
    torch.cuda.empty_cache()
    gc.collect()

    # Load YOLO models
    available_providers = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in available_providers
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    print(f"\nONNX providers: {providers}")

    model_files = ["yolo11x_v3.onnx", "yolo26x_v3.onnx"]
    yolo_models = []
    for mf in model_files:
        path = YOLO_DIR / mf
        if path.exists():
            yolo_models.append(load_onnx_session(path, providers))
            print(f"  Loaded {mf}")

    # 2. YOLO detection mAP (class-agnostic)
    det_map = eval_detection_map(gt, yolo_models)

    # 3. Full pipeline mAP
    # Free YOLO CUDA memory first, then reload for full pipeline on CPU
    del yolo_models
    torch.cuda.empty_cache()
    gc.collect()

    # Reload YOLO on CPU to leave GPU for DINOv2
    cpu_providers = ["CPUExecutionProvider"]
    yolo_models = []
    for mf in model_files:
        path = YOLO_DIR / mf
        if path.exists():
            yolo_models.append(load_onnx_session(path, cpu_providers))
            print(f"  Reloaded {mf} on CPU")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dino_model, transform = load_dino(device)
    linear_probe = load_linear_probe(device)
    centroids, centroid_mask, ref_matrix, ref_cat_ids = load_embedding_bank(device)

    full_map = eval_full_pipeline_map(
        gt, yolo_models, dino_model, transform, linear_probe,
        centroids, centroid_mask, ref_matrix, ref_cat_ids, device
    )

    elapsed = time.time() - t0

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  DINOv2 combined classification accuracy:  {cls_acc:.4f}")
    print(f"  YOLO class-agnostic detection mAP@0.5:    {det_map:.4f}")
    print(f"  Full pipeline per-class mAP@0.5:          {full_map:.4f}")
    print(f"  Time elapsed: {elapsed:.1f}s")
    print(f"\n  >>> Competition metric estimate: {full_map:.4f}")


if __name__ == "__main__":
    main()

"""
Dense shelf inference: 2-model ONNX detection ensemble + DINOv2 retrieval.

Usage:
    python run.py --input /data/images --output /predictions.json
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
MODEL_FILES = ("yolo11x_v3.onnx", "yolo26x_v3.onnx")
DINO_WEIGHTS = "dinov2_vits14.pth"
DATA_BANK = "ref_embeddings_data.pth"
MULTI_BANK = "ref_embeddings_multi.pth"
LINEAR_PROBE = "linear_probe.pth"

NUM_CLASSES = 356
CONF_THRESH = 0.001
NMS_IOU_THRESH = 0.45
WBF_IOU_THRESH = 0.60
SOURCE_MATCH_IOU = 0.55
MAX_DET = 300
CROP_BATCH_SIZE = 128
USE_FLIP_TTA = True


def load_onnx_session(model_path: Path, providers: list[str]):
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_info = session.get_inputs()[0]
    return session, input_info.name, tuple(input_info.shape)


def letterbox(image: np.ndarray, new_shape: tuple[int, int], color=(114, 114, 114)):
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
    bordered = cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )
    return bordered, ratio, (pad_w, pad_h)


def preprocess_image(image_bgr: np.ndarray, input_shape: tuple[int, ...]):
    target_h, target_w = int(input_shape[2]), int(input_shape[3])
    letterboxed, ratio, pad = letterbox(image_bgr, (target_h, target_w))
    image_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    tensor = image_rgb.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...]
    return tensor, ratio, pad


def compute_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    areas = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = box_area + areas - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def nms_by_class(boxes: np.ndarray, scores: np.ndarray, labels: np.ndarray, iou_thresh: float):
    if len(boxes) == 0:
        return boxes, scores, labels

    kept_boxes = []
    kept_scores = []
    kept_labels = []

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


def decode_yolo_output(output, ratio: float, pad: tuple[float, float], original_shape: tuple[int, int]):
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
    box_cxcywh = box_cxcywh[mask]
    labels = labels[mask]
    scores = scores[mask]
    if len(box_cxcywh) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

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
    boxes = boxes[valid]
    scores = scores[valid]
    labels = labels[valid]
    if len(boxes) == 0:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    boxes, scores, labels = nms_by_class(boxes, scores, labels, NMS_IOU_THRESH)
    if len(scores) > MAX_DET:
        top = np.argsort(scores)[::-1][:MAX_DET]
        boxes = boxes[top]
        scores = scores[top]
        labels = labels[top]
    return boxes.astype(np.float32), scores.astype(np.float32), labels.astype(np.int64)


def weighted_boxes_fusion(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    iou_thr: float,
    skip_box_thr: float,
):
    weighted = []
    for boxes, scores, labels in zip(boxes_list, scores_list, labels_list):
        for idx in range(len(boxes)):
            if scores[idx] < skip_box_thr:
                continue
            weighted.append(
                {
                    "box": boxes[idx].copy(),
                    "score": float(scores[idx]),
                    "label": int(labels[idx]),
                }
            )

    if not weighted:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int64),
        )

    weighted.sort(key=lambda entry: entry["score"], reverse=True)
    grouped: dict[int, list[list[dict]]] = {}
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
            if compute_iou(entry["box"], fused_box[None, :])[0] > iou_thr:
                cluster.append(entry)
                matched = True
                break
        if not matched:
            clusters.append([entry])

    fused_boxes = []
    fused_scores = []
    fused_labels = []
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


def run_detector(session, input_name: str, input_shape: tuple[int, ...], image_bgr: np.ndarray, flip: bool):
    source = cv2.flip(image_bgr, 1) if flip else image_bgr
    tensor, ratio, pad = preprocess_image(source, input_shape)
    outputs = session.run(None, {input_name: tensor})
    boxes, scores, labels = decode_yolo_output(outputs, ratio, pad, source.shape[:2])
    if flip and len(boxes) > 0:
        original_width = image_bgr.shape[1]
        flipped = boxes.copy()
        flipped[:, 0] = original_width - boxes[:, 2]
        flipped[:, 2] = original_width - boxes[:, 0]
        boxes = flipped
    return boxes, scores, labels


def build_detector_prior(fused_boxes: np.ndarray, source_boxes: np.ndarray, source_scores: np.ndarray, source_labels: np.ndarray):
    prior = np.zeros((len(fused_boxes), NUM_CLASSES), dtype=np.float32)
    if len(fused_boxes) == 0 or len(source_boxes) == 0:
        return prior

    for idx, fused_box in enumerate(fused_boxes):
        iou = compute_iou(fused_box, source_boxes)
        matches = np.where(iou >= SOURCE_MATCH_IOU)[0]
        if len(matches) == 0:
            continue
        for match in matches:
            prior[idx, source_labels[match]] += source_scores[match]
        row_sum = prior[idx].sum()
        if row_sum > 0:
            prior[idx] /= row_sum
    return prior


def load_dino(device: torch.device):
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
    state_dict = torch.load(SCRIPT_DIR / DINO_WEIGHTS, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config, is_training=False)
    return model, transform


def load_linear_probe(device: torch.device):
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


def load_embedding_bank(device: torch.device):
    centroids = torch.zeros((NUM_CLASSES, 384), dtype=torch.float32, device=device)
    centroid_mask = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=device)
    ref_embeddings = []
    ref_categories = []

    data_bank_path = SCRIPT_DIR / DATA_BANK
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

    multi_bank_path = SCRIPT_DIR / MULTI_BANK
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


def build_ref_similarity(embeddings: torch.Tensor, ref_matrix: torch.Tensor, ref_cat_ids: torch.Tensor):
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


def classify_boxes(
    image_bgr: np.ndarray,
    boxes: np.ndarray,
    detector_prior: np.ndarray,
    dino_model,
    transform,
    centroids: torch.Tensor,
    centroid_mask: torch.Tensor,
    ref_matrix: torch.Tensor,
    ref_cat_ids: torch.Tensor,
    linear_probe,
    device: torch.device,
    use_reference_bank: bool,
    crop_batch_size: int,
):
    if len(boxes) == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_rgb)
    category_ids = []
    confidences = []

    for start in range(0, len(boxes), crop_batch_size):
        batch_boxes = boxes[start:start + crop_batch_size]
        crops = []
        for box in batch_boxes:
            x1 = max(0, int(box[0]))
            y1 = max(0, int(box[1]))
            x2 = min(image_pil.width, int(box[2]))
            y2 = min(image_pil.height, int(box[3]))
            if x2 <= x1 or y2 <= y1:
                crops.append(Image.new("RGB", (32, 32), (114, 114, 114)))
            else:
                crops.append(image_pil.crop((x1, y1, x2, y2)))

        tensors = torch.stack([transform(crop) for crop in crops]).to(device)
        autocast_enabled = device.type == "cuda"
        with torch.inference_mode():
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=autocast_enabled):
                embeddings = dino_model(tensors)
            embeddings = F.normalize(embeddings.float(), dim=-1)

            centroid_logits = embeddings @ centroids.T
            centroid_logits[:, ~centroid_mask] = -20.0
            centroid_prob = F.softmax(centroid_logits / 0.12, dim=-1)

            if use_reference_bank:
                ref_logits = build_ref_similarity(embeddings, ref_matrix, ref_cat_ids)
                ref_prob = F.softmax(ref_logits / 0.10, dim=-1)
            else:
                ref_prob = torch.zeros_like(centroid_prob)

            ref_weight = 0.35 if use_reference_bank else 0.0
            combined = centroid_prob * 0.45 + ref_prob * ref_weight
            weight_total = torch.full((len(batch_boxes), 1), 0.45 + ref_weight, device=device)

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


def prepare_fused_detections_with_tta(models, image_bgr: np.ndarray, use_flip_tta: bool):
    all_boxes = []
    all_scores = []
    all_labels = []
    source_boxes = []
    source_scores = []
    source_labels = []

    for session, input_name, input_shape in models:
        for flip in (False, True) if use_flip_tta else (False,):
            boxes, scores, labels = run_detector(session, input_name, input_shape, image_bgr, flip)
            if len(boxes) == 0:
                continue
            normalized = boxes.copy()
            normalized[:, [0, 2]] /= image_bgr.shape[1]
            normalized[:, [1, 3]] /= image_bgr.shape[0]
            all_boxes.append(normalized)
            all_scores.append(scores)
            all_labels.append(np.zeros(len(labels), dtype=np.int64))
            source_boxes.append(boxes)
            source_scores.append(scores)
            source_labels.append(labels)

    if not all_boxes:
        empty_boxes = np.empty((0, 4), dtype=np.float32)
        empty_scores = np.empty(0, dtype=np.float32)
        empty_labels = np.empty(0, dtype=np.int64)
        return empty_boxes, empty_scores, empty_labels, empty_boxes, empty_scores, empty_labels

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes,
        all_scores,
        all_labels,
        iou_thr=WBF_IOU_THRESH,
        skip_box_thr=CONF_THRESH,
    )
    fused_boxes[:, [0, 2]] *= image_bgr.shape[1]
    fused_boxes[:, [1, 3]] *= image_bgr.shape[0]

    stacked_source_boxes = np.concatenate(source_boxes, axis=0) if source_boxes else np.empty((0, 4), dtype=np.float32)
    stacked_source_scores = np.concatenate(source_scores, axis=0) if source_scores else np.empty(0, dtype=np.float32)
    stacked_source_labels = np.concatenate(source_labels, axis=0) if source_labels else np.empty(0, dtype=np.int64)
    return fused_boxes, fused_scores, fused_labels, stacked_source_boxes, stacked_source_scores, stacked_source_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    available_providers = ort.get_available_providers()
    use_cuda = "CUDAExecutionProvider" in available_providers
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if use_cuda else ["CPUExecutionProvider"]
    model_files = MODEL_FILES if use_cuda else MODEL_FILES[:1]
    use_flip_tta = USE_FLIP_TTA and use_cuda
    use_reference_bank = use_cuda
    crop_batch_size = CROP_BATCH_SIZE if use_cuda else 16

    models = []
    for model_name in model_files:
        models.append(load_onnx_session(SCRIPT_DIR / model_name, providers))

    dino_model, transform = load_dino(device)
    linear_probe = load_linear_probe(device)
    centroids, centroid_mask, ref_matrix, ref_cat_ids = load_embedding_bank(device)

    input_dir = Path(args.input)
    output_path = Path(args.output)
    image_paths = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg")) + list(input_dir.glob("*.png")))

    results = []
    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            continue

        fused_boxes, fused_scores, _, source_boxes, source_scores, source_labels = prepare_fused_detections_with_tta(models, image_bgr, use_flip_tta)
        if len(fused_boxes) == 0:
            continue

        detector_prior = build_detector_prior(fused_boxes, source_boxes, source_scores, source_labels)
        category_ids, class_conf = classify_boxes(
            image_bgr,
            fused_boxes,
            detector_prior,
            dino_model,
            transform,
            centroids,
            centroid_mask,
            ref_matrix,
            ref_cat_ids,
            linear_probe,
            device,
            use_reference_bank,
            crop_batch_size,
        )

        final_scores = np.clip(fused_scores * (0.70 + 0.30 * class_conf), 0.0, 1.0)
        for idx, box in enumerate(fused_boxes):
            x1, y1, x2, y2 = box
            results.append(
                {
                    "image_id": image_path.name,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "category_id": int(category_ids[idx]),
                    "score": float(final_scores[idx]),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results))


if __name__ == "__main__":
    main()

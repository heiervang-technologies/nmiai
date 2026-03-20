"""
run.py -- Hybrid ONNX YOLO + DINOv2/v3 inference pipeline.

Usage: python run.py --input /data/images/ --output /predictions.json

No ultralytics dependency. Uses onnxruntime-gpu for detection
and timm for DINOv2/v3 classification.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import timm
from timm.data import resolve_data_config, create_transform
from PIL import Image


# === ONNX YOLO Detection ===

def load_onnx_model(model_path: Path):
    """Load ONNX model with CUDA provider."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape  # e.g. [1, 3, 1280, 1280]
    return session, input_name, input_shape


def letterbox(img, new_shape=(1280, 1280), color=(114, 114, 114)):
    """Resize and pad image to target size, preserving aspect ratio."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def preprocess_yolo(img_bgr, input_shape):
    """Preprocess image for YOLO ONNX inference."""
    h, w = input_shape[2], input_shape[3]
    img_lb, ratio, (dw, dh) = letterbox(img_bgr, (h, w))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_f = img_rgb.astype(np.float32) / 255.0
    img_t = np.transpose(img_f, (2, 0, 1))[np.newaxis, ...]  # [1, 3, H, W]
    return img_t, ratio, (dw, dh)


def postprocess_yolo(output, ratio, pad, conf_thresh=0.001, iou_thresh=0.45, max_det=300):
    """Post-process YOLOv8 ONNX output.

    Output shape: [1, 4+nc, num_boxes] -> transpose to [num_boxes, 4+nc].
    First 4 values: cx, cy, w, h (center format).
    """
    pred = output[0]  # [1, 4+nc, N]
    if pred.ndim == 3:
        pred = pred[0]  # [4+nc, N]
    pred = pred.T  # [N, 4+nc]

    # Split boxes and class scores
    boxes_cxcywh = pred[:, :4]
    class_scores = pred[:, 4:]
    nc = class_scores.shape[1]

    # Get best class per box
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.array([class_scores[i, class_ids[i]] for i in range(len(class_ids))])

    # Filter by confidence
    mask = confidences > conf_thresh
    boxes_cxcywh = boxes_cxcywh[mask]
    class_ids = class_ids[mask]
    confidences = confidences[mask]

    if len(boxes_cxcywh) == 0:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # Convert cx,cy,w,h to x1,y1,x2,y2
    boxes_xyxy = np.zeros_like(boxes_cxcywh)
    boxes_xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2  # x1
    boxes_xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2  # y1
    boxes_xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2  # x2
    boxes_xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2  # y2

    # Remove letterbox padding and scale to original image coords
    dw, dh = pad
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - dw) / ratio
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - dh) / ratio

    # Per-class NMS
    final_boxes = []
    final_scores = []
    final_classes = []

    for cls_id in range(nc):
        cls_mask = class_ids == cls_id
        if not np.any(cls_mask):
            continue
        cls_boxes = boxes_xyxy[cls_mask]
        cls_scores = confidences[cls_mask]
        keep = _nms(cls_boxes, cls_scores, iou_thresh)
        final_boxes.append(cls_boxes[keep])
        final_scores.append(cls_scores[keep])
        final_classes.append(np.full(len(keep), cls_id, dtype=int))

    if not final_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    final_boxes = np.concatenate(final_boxes)
    final_scores = np.concatenate(final_scores)
    final_classes = np.concatenate(final_classes)

    # Keep top max_det by confidence
    if len(final_scores) > max_det:
        top_idx = np.argsort(final_scores)[::-1][:max_det]
        final_boxes = final_boxes[top_idx]
        final_scores = final_scores[top_idx]
        final_classes = final_classes[top_idx]

    return final_boxes, final_scores, final_classes


def _nms(boxes, scores, iou_thresh):
    """Standard NMS on [N, 4] boxes (x1,y1,x2,y2) with [N] scores."""
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
        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]

    return keep


# === DINOv2/v3 Classification ===

def load_dino(weights_path: Path, model_name: str, device: torch.device):
    """Load DINOv2 or DINOv3 model via timm."""
    model = timm.create_model(model_name, pretrained=False, num_classes=0)
    if weights_path.exists():
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)
    data_config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**data_config, is_training=False)
    return model, transform


def load_ref_embeddings(path: Path, device: torch.device):
    """Load pre-computed reference embeddings (supports both formats)."""
    data = torch.load(path, map_location=device, weights_only=False)
    ref_embeddings = {}

    # Data agent format
    if "category_embeddings" in data:
        cat_embs = data["category_embeddings"].float().to(device)
        cat_ids = data.get("category_ids", list(range(cat_embs.shape[0])))
        for i, cat_id in enumerate(cat_ids):
            emb = F.normalize(cat_embs[i].unsqueeze(0), dim=-1)
            ref_embeddings[int(cat_id)] = emb
        if "reference_embeddings" in data and "barcode_to_category" in data:
            ref_emb = data["reference_embeddings"].float().to(device)
            barcodes = data.get("reference_barcodes", [])
            b2c = data["barcode_to_category"]
            for i, bc in enumerate(barcodes):
                if str(bc) in b2c:
                    cat_id = int(b2c[str(bc)])
                    emb = F.normalize(ref_emb[i].unsqueeze(0), dim=-1)
                    if cat_id in ref_embeddings:
                        ref_embeddings[cat_id] = torch.cat([ref_embeddings[cat_id], emb])
                    else:
                        ref_embeddings[cat_id] = emb

    # Our format
    elif "embeddings" in data:
        for cat_id, emb in data["embeddings"].items():
            if isinstance(emb, torch.Tensor):
                if emb.dim() == 1:
                    emb = emb.unsqueeze(0)
                ref_embeddings[int(cat_id)] = F.normalize(emb.float().to(device), dim=-1)

    return ref_embeddings


@torch.no_grad()
def classify_crops(dino_model, transform, crops, ref_embeddings, linear_probe,
                   yolo_classes, yolo_confs, device, batch_size=64,
                   yolo_trust=0.7, dino_trust=0.5):
    """Classify crops using DINOv2/v3 embeddings and fuse with YOLO predictions."""
    N = len(crops)
    if N == 0:
        return np.array([], dtype=int), np.array([])

    # Embed all crops
    all_embeddings = []
    for i in range(0, N, batch_size):
        batch = crops[i:i + batch_size]
        tensors = torch.stack([transform(c.convert("RGB")) for c in batch]).to(device)
        embs = F.normalize(dino_model(tensors), dim=-1)
        all_embeddings.append(embs)
    embeddings = torch.cat(all_embeddings)

    # Classify: linear probe if available, else nearest-neighbor
    if linear_probe is not None:
        logits = linear_probe(embeddings)
        probs = F.softmax(logits, dim=-1)
        dino_confs, dino_classes = probs.max(dim=-1)
    elif ref_embeddings:
        dino_classes = torch.zeros(N, dtype=torch.long, device=device)
        dino_confs = torch.full((N,), -1.0, device=device)
        for cat_id, ref_emb in ref_embeddings.items():
            sim = (embeddings @ ref_emb.T).max(dim=-1).values
            better = sim > dino_confs
            dino_confs[better] = sim[better]
            dino_classes[better] = cat_id
    else:
        return yolo_classes, yolo_confs

    # Fusion
    yolo_cls = torch.tensor(yolo_classes, dtype=torch.long, device=device)
    yolo_conf = torch.tensor(yolo_confs, dtype=torch.float, device=device)

    final_cls = yolo_cls.clone()
    final_conf = yolo_conf.clone()

    yolo_confident = yolo_conf > yolo_trust
    dino_confident = dino_confs > dino_trust
    agrees = dino_classes == yolo_cls

    # Both agree → boost
    agreed = yolo_confident & agrees
    final_conf[agreed] = torch.max(yolo_conf[agreed], dino_confs[agreed])

    # YOLO uncertain, DINOv confident → override
    override = ~yolo_confident & dino_confident
    final_cls[override] = dino_classes[override]
    final_conf[override] = dino_confs[override]

    # Both confident, disagree → trust higher
    disagree = yolo_confident & dino_confident & ~agrees
    dino_wins = disagree & (dino_confs > yolo_conf)
    final_cls[dino_wins] = dino_classes[dino_wins]
    final_conf[dino_wins] = dino_confs[dino_wins]

    return final_cls.cpu().numpy(), final_conf.cpu().numpy()


# === Main ===

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = Path(__file__).parent

    # --- Load ONNX YOLO ---
    onnx_path = script_dir / "best.onnx"
    if not onnx_path.exists():
        for name in ["model.onnx", "yolo.onnx"]:
            alt = script_dir / name
            if alt.exists():
                onnx_path = alt
                break
    session, input_name, input_shape = load_onnx_model(onnx_path)
    imgsz = (input_shape[2], input_shape[3])

    # --- Load DINOv2/v3 ---
    # Try DINOv3 first, fall back to DINOv2
    dino_model = None
    dino_transform = None
    for model_name, weights_name in [
        ("vit_small_patch16_dinov3", "dinov3_vits16.pth"),
        ("vit_small_patch14_dinov2.lvd142m", "dinov2_vits14.pth"),
    ]:
        weights_path = script_dir / weights_name
        if weights_path.exists():
            try:
                dino_model, dino_transform = load_dino(weights_path, model_name, device)
                break
            except Exception:
                continue

    # --- Load reference embeddings ---
    ref_embeddings = {}
    for ref_name in ["dinov3_ref_embeddings.pth", "ref_embeddings.pth"]:
        ref_path = script_dir / ref_name
        if ref_path.exists():
            ref_embeddings = load_ref_embeddings(ref_path, device)
            break

    # --- Load linear probe ---
    linear_probe = None
    for probe_name in ["dinov3_linear_probe.pth", "linear_probe.pth"]:
        probe_path = script_dir / probe_name
        if probe_path.exists():
            state = torch.load(probe_path, map_location=device, weights_only=False)
            if isinstance(state, dict) and "weight" in state:
                num_classes, embed_dim = state["weight"].shape
                probe = torch.nn.Linear(embed_dim, num_classes)
                probe.load_state_dict(state)
                linear_probe = probe.eval().to(device)
            break

    # --- Process images ---
    images_dir = Path(args.input)
    image_paths = sorted(images_dir.glob("*.jpg"))
    results = []

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            results.append({"image_id": img_path.name, "predictions": []})
            continue

        orig_h, orig_w = img_bgr.shape[:2]

        # YOLO detection
        img_input, ratio, pad = preprocess_yolo(img_bgr, input_shape)
        onnx_out = session.run(None, {input_name: img_input})
        boxes_xyxy, scores, class_ids = postprocess_yolo(
            onnx_out, ratio, pad, conf_thresh=0.001, iou_thresh=0.45, max_det=300
        )

        if len(boxes_xyxy) == 0:
            results.append({"image_id": img_path.name, "predictions": []})
            continue

        # Clamp boxes to image bounds
        boxes_xyxy[:, [0, 2]] = np.clip(boxes_xyxy[:, [0, 2]], 0, orig_w)
        boxes_xyxy[:, [1, 3]] = np.clip(boxes_xyxy[:, [1, 3]], 0, orig_h)

        # DINOv2/v3 classification
        if dino_model is not None and (ref_embeddings or linear_probe is not None):
            img_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            crops = []
            for x1, y1, x2, y2 in boxes_xyxy:
                x1i, y1i = max(0, int(x1)), max(0, int(y1))
                x2i, y2i = min(orig_w, int(x2)), min(orig_h, int(y2))
                if x2i > x1i and y2i > y1i:
                    crops.append(img_pil.crop((x1i, y1i, x2i, y2i)))
                else:
                    crops.append(Image.new("RGB", (32, 32), (128, 128, 128)))

            final_classes, final_confs = classify_crops(
                dino_model, dino_transform, crops, ref_embeddings, linear_probe,
                class_ids, scores, device
            )
        else:
            final_classes = class_ids
            final_confs = scores

        # Build predictions in COCO format
        predictions = []
        for i in range(len(boxes_xyxy)):
            x1, y1, x2, y2 = boxes_xyxy[i]
            predictions.append({
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "category_id": int(final_classes[i]),
                "score": float(final_confs[i]),
            })

        results.append({"image_id": img_path.name, "predictions": predictions})

    with open(args.output, "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()

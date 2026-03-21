"""Quick MarkusNet eval using transformers (local only).
Loads pruned model, classifies YOLO crops from val set, computes accuracy + mAP.
"""
import json
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from PIL import Image
import cv2
import onnxruntime as ort

# Paths
VAL_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data/yolo_augmented_v5/val/images")
LABEL_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data/yolo_augmented_v5/val/labels")
YOLO_PATH = Path("/home/me/ht/nmiai/tasks/object-detection/submission-markusnet/best.onnx")
PRUNED_DIR = Path("/home/me/ht/nmiai/tasks/object-detection/vlm-approach/pruned")
CLS_HEAD_PATH = Path("/home/me/ht/nmiai/tasks/object-detection/vlm-approach/exported/markusnet_351m_nf4.pt")

# Load category mapping
CATEGORY_FILE = Path("/home/me/ht/nmiai/tasks/object-detection/data-creation/data/yolo_augmented_v5/dataset.yaml")

def load_categories():
    """Load YOLO category names from dataset.yaml"""
    import yaml
    with open(CATEGORY_FILE) as f:
        data = yaml.safe_load(f)
    return data.get('names', {})

def load_yolo(path):
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), providers=providers)
    inp = session.get_inputs()[0]
    return session, inp.name, tuple(inp.shape)

def preprocess_yolo(image_bgr, input_shape):
    _, _, h_in, w_in = input_shape
    h, w = image_bgr.shape[:2]
    ratio = min(w_in / w, h_in / h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    resized = cv2.resize(image_bgr, (new_w, new_h))
    pad_w, pad_h = (w_in - new_w) // 2, (h_in - new_h) // 2
    canvas = np.full((h_in, w_in, 3), 114, dtype=np.uint8)
    canvas[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    tensor = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return tensor, ratio, (pad_w, pad_h)

def decode_yolo(outputs, ratio, pad, img_shape, conf_thresh=0.25, iou_thresh=0.5):
    pred = outputs[0]
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T

    scores = pred[:, 4] if pred.shape[1] == 5 else pred[:, 4:].max(axis=1)
    mask = scores > conf_thresh
    pred = pred[mask]
    scores = scores[mask]

    if len(pred) == 0:
        return np.array([]), np.array([])

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    x1 = (cx - w/2 - pad[0]) / ratio
    y1 = (cy - h/2 - pad[1]) / ratio
    x2 = (cx + w/2 - pad[0]) / ratio
    y2 = (cy + h/2 - pad[1]) / ratio

    boxes = np.stack([x1, y1, x2, y2], axis=1)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, img_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, img_shape[0])

    # NMS
    idxs = nms(boxes, scores, iou_thresh)
    return boxes[idxs], scores[idxs]

def nms(boxes, scores, iou_thresh):
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        areas_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        areas_j = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (areas_i + areas_j - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def load_gt_labels(label_dir, img_name, img_w, img_h):
    """Load YOLO format ground truth labels."""
    label_path = label_dir / (Path(img_name).stem + ".txt")
    if not label_path.exists():
        return [], []
    boxes, classes = [], []
    for line in label_path.read_text().strip().split('\n'):
        if not line.strip():
            continue
        parts = line.strip().split()
        cls_id = int(parts[0])
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w/2) * img_w
        y1 = (cy - h/2) * img_h
        x2 = (cx + w/2) * img_w
        y2 = (cy + h/2) * img_h
        boxes.append([x1, y1, x2, y2])
        classes.append(cls_id)
    return np.array(boxes) if boxes else np.array([]), classes

def match_boxes(pred_boxes, gt_boxes, iou_thresh=0.5):
    """Match predicted boxes to GT boxes by IoU. Returns list of (pred_idx, gt_idx) pairs."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return []

    matches = []
    used_gt = set()
    for pi in range(len(pred_boxes)):
        best_iou, best_gi = 0, -1
        for gi in range(len(gt_boxes)):
            if gi in used_gt:
                continue
            xx1 = max(pred_boxes[pi][0], gt_boxes[gi][0])
            yy1 = max(pred_boxes[pi][1], gt_boxes[gi][1])
            xx2 = min(pred_boxes[pi][2], gt_boxes[gi][2])
            yy2 = min(pred_boxes[pi][3], gt_boxes[gi][3])
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            inter = w * h
            a1 = (pred_boxes[pi][2] - pred_boxes[pi][0]) * (pred_boxes[pi][3] - pred_boxes[pi][1])
            a2 = (gt_boxes[gi][2] - gt_boxes[gi][0]) * (gt_boxes[gi][3] - gt_boxes[gi][1])
            iou = inter / (a1 + a2 - inter + 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= iou_thresh and best_gi >= 0:
            matches.append((pi, best_gi))
            used_gt.add(best_gi)
    return matches

def main():
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

    t0 = time.time()
    device = torch.device("cuda")

    # Load YOLO
    print("Loading YOLO...")
    yolo_session, yolo_input_name, yolo_input_shape = load_yolo(YOLO_PATH)

    # Load MarkusNet via transformers
    print("Loading MarkusNet (transformers)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(PRUNED_DIR), torch_dtype=torch.float16, device_map="cpu"
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(str(PRUNED_DIR))

    # Load classification head from NF4 checkpoint
    ckpt = torch.load(str(CLS_HEAD_PATH), map_location="cpu", weights_only=False)
    cls_head = torch.nn.Sequential(
        torch.nn.Linear(1024, 512),
        torch.nn.GELU(),
        torch.nn.Dropout(0.1),
        torch.nn.Linear(512, 356),
    )
    cls_head.load_state_dict(ckpt['cls_head_state'])
    cls_head = cls_head.half().to(device)
    cls_head.eval()

    # Run model on CPU (GPU is full from other training jobs)
    # Only YOLO runs on GPU via ONNX
    model = model.float().cpu()
    cls_head = cls_head.float().cpu()
    model_device = torch.device("cpu")

    print(f"Models loaded in {time.time()-t0:.1f}s")

    # Process val images
    image_paths = sorted(list(VAL_DIR.glob("*.jpg")) + list(VAL_DIR.glob("*.png")))
    print(f"Processing {len(image_paths)} validation images...")

    total_correct = 0
    total_matched = 0
    total_gt = 0
    total_pred = 0
    all_results = []  # For mAP computation

    for img_idx, img_path in enumerate(image_paths):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            continue
        h_img, w_img = image_bgr.shape[:2]

        # YOLO detection
        tensor, ratio, pad = preprocess_yolo(image_bgr, yolo_input_shape)
        outputs = yolo_session.run(None, {yolo_input_name: tensor})
        pred_boxes, det_scores = decode_yolo(outputs, ratio, pad, image_bgr.shape[:2])

        # Load GT
        gt_boxes, gt_classes = load_gt_labels(LABEL_DIR, img_path.name, w_img, h_img)
        total_gt += len(gt_classes)
        total_pred += len(pred_boxes)

        if len(pred_boxes) == 0:
            continue

        # Classify each crop with MarkusNet
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_pil = Image.fromarray(image_rgb)

        pred_cat_ids = []
        pred_confs = []

        for box in pred_boxes:
            x1, y1, x2, y2 = int(max(0, box[0])), int(max(0, box[1])), int(min(w_img, box[2])), int(min(h_img, box[3]))
            if x2 <= x1 or y2 <= y1:
                pred_cat_ids.append(0)
                pred_confs.append(0.0)
                continue

            crop = image_pil.crop((x1, y1, x2, y2))

            # Use transformers processor for preprocessing
            messages = [{"role": "user", "content": [
                {"type": "image", "image": crop, "min_pixels": 12544, "max_pixels": 65536},
                {"type": "text", "text": "classify"},
            ]}]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[crop], return_tensors="pt", padding=True)
            inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
                hidden = out.hidden_states[-1][:, -1, :]  # Last token
                logits = cls_head(hidden)
                probs = F.softmax(logits, dim=-1)
                conf, cat_id = probs.max(dim=-1)
                pred_cat_ids.append(cat_id.item())
                pred_confs.append(conf.item())

        # Match predictions to GT
        if len(gt_boxes) > 0:
            matches = match_boxes(pred_boxes, gt_boxes)
            for pi, gi in matches:
                total_matched += 1
                if pred_cat_ids[pi] == gt_classes[gi]:
                    total_correct += 1

        if (img_idx + 1) % 5 == 0:
            acc = total_correct / max(total_matched, 1) * 100
            print(f"  [{img_idx+1}/{len(image_paths)}] matched={total_matched}, correct={total_correct}, acc={acc:.1f}%")

        pass  # CPU mode, no cache to clear

    # Final stats
    acc = total_correct / max(total_matched, 1) * 100
    recall = total_matched / max(total_gt, 1) * 100
    print(f"\n=== RESULTS ===")
    print(f"GT objects: {total_gt}")
    print(f"Predicted boxes: {total_pred}")
    print(f"Matched (IoU>0.5): {total_matched}")
    print(f"Correct class: {total_correct}")
    print(f"Classification accuracy: {acc:.1f}%")
    print(f"Detection recall: {recall:.1f}%")
    print(f"Total time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()

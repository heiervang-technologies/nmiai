"""
Pseudo-label unlabeled store photos using our best YOLOv8x model + DINOv2 classifier.

Pipeline:
1. Run YOLOv8x ONNX detection on each store photo
2. For each detection, extract crop and classify with DINOv2 + linear probe
3. Filter by confidence thresholds
4. Output YOLO format labels

Requires GPU for efficient inference.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import torch

DATA_DIR = Path(__file__).parent / "data"
STORE_PHOTOS = DATA_DIR / "store_photos"
OUTPUT_DIR = DATA_DIR / "pseudo_labeled_stores"
OUT_IMAGES = OUTPUT_DIR / "images"
OUT_LABELS = OUTPUT_DIR / "labels"

# Model paths
YOLO_APPROACH = Path(__file__).parent.parent / "yolo-approach"
SUBMISSION_DIR = Path(__file__).parent.parent / "submission-single-model"

# Thresholds
DET_CONF_THRESHOLD = 0.3   # detection confidence (lower = more recall)
CLS_CONF_THRESHOLD = 0.4   # classification confidence
NMS_IOU_THRESHOLD = 0.45
MAX_DET = 300
INPUT_SIZE = 1280


def letterbox(img, new_shape=(INPUT_SIZE, INPUT_SIZE)):
    """Resize image with letterboxing."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = int(round(w * r)), int(round(h * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))

    return img, r, (dw, dh)


def nms_per_class(boxes, scores, class_ids, iou_threshold=0.45):
    """Per-class NMS."""
    keep = []
    unique_classes = np.unique(class_ids)
    for cls in unique_classes:
        mask = class_ids == cls
        cls_boxes = boxes[mask]
        cls_scores = scores[mask]
        cls_indices = np.where(mask)[0]

        # Sort by score
        order = cls_scores.argsort()[::-1]
        cls_boxes = cls_boxes[order]
        cls_scores = cls_scores[order]
        cls_indices = cls_indices[order]

        picked = []
        while len(cls_boxes) > 0:
            picked.append(cls_indices[0])
            if len(cls_boxes) == 1:
                break

            # IoU with rest
            x1 = np.maximum(cls_boxes[0, 0], cls_boxes[1:, 0])
            y1 = np.maximum(cls_boxes[0, 1], cls_boxes[1:, 1])
            x2 = np.minimum(cls_boxes[0, 2], cls_boxes[1:, 2])
            y2 = np.minimum(cls_boxes[0, 3], cls_boxes[1:, 3])

            intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
            area0 = (cls_boxes[0, 2] - cls_boxes[0, 0]) * (cls_boxes[0, 3] - cls_boxes[0, 1])
            areas = (cls_boxes[1:, 2] - cls_boxes[1:, 0]) * (cls_boxes[1:, 3] - cls_boxes[1:, 1])
            iou = intersection / (area0 + areas - intersection + 1e-6)

            mask_keep = iou < iou_threshold
            cls_boxes = cls_boxes[1:][mask_keep]
            cls_scores = cls_scores[1:][mask_keep]
            cls_indices = cls_indices[1:][mask_keep]

        keep.extend(picked)
    return keep


def run_yolo_onnx(session, img_bgr):
    """Run YOLOv8 ONNX inference."""
    orig_h, orig_w = img_bgr.shape[:2]

    # Letterbox
    img_lb, ratio, (dw, dh) = letterbox(img_bgr, (INPUT_SIZE, INPUT_SIZE))

    # Preprocess
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    blob = img_rgb.astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]

    # Inference
    outputs = session.run(None, {session.get_inputs()[0].name: blob})
    output = outputs[0]  # shape: [1, 4+nc, num_boxes]

    # Transpose to [num_boxes, 4+nc]
    pred = output[0].T

    # Extract boxes and class scores
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    class_scores = pred[:, 4:]

    # Get best class per box
    best_class = class_scores.argmax(axis=1)
    best_score = class_scores.max(axis=1)

    # Filter by confidence
    mask = best_score > DET_CONF_THRESHOLD
    cx, cy, w, h = cx[mask], cy[mask], w[mask], h[mask]
    best_class = best_class[mask]
    best_score = best_score[mask]

    # Convert to xyxy
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    # NMS
    keep = nms_per_class(boxes, best_score, best_class, NMS_IOU_THRESHOLD)
    boxes = boxes[keep]
    best_class = best_class[keep]
    best_score = best_score[keep]

    # Undo letterbox
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio

    # Clip to image
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    # Filter tiny boxes
    w_final = boxes[:, 2] - boxes[:, 0]
    h_final = boxes[:, 3] - boxes[:, 1]
    valid = (w_final > 5) & (h_final > 5)
    boxes = boxes[valid]
    best_class = best_class[valid]
    best_score = best_score[valid]

    # Keep top MAX_DET
    if len(boxes) > MAX_DET:
        top_idx = best_score.argsort()[::-1][:MAX_DET]
        boxes = boxes[top_idx]
        best_class = best_class[top_idx]
        best_score = best_score[top_idx]

    return boxes, best_class, best_score


def run_dino_classifier(model, transform, probe, img_bgr, boxes, device="cuda"):
    """Classify detected crops using DINOv2 + linear probe."""
    from PIL import Image as PILImage

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = PILImage.fromarray(img_rgb)
    h, w = img_rgb.shape[:2]

    class_ids = []
    class_confs = []

    # Batch process crops
    batch = []
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            batch.append(None)
            continue
        crop = img_pil.crop((x1, y1, x2, y2))
        try:
            batch.append(transform(crop))
        except Exception:
            batch.append(None)

    # Process in mini-batches (small batch for CPU)
    batch_size = 8 if device == "cpu" else 64
    all_preds = []
    all_confs = []

    valid_tensors = [t for t in batch if t is not None]
    if not valid_tensors:
        return np.array([]), np.array([])

    for i in range(0, len(valid_tensors), batch_size):
        mini = torch.stack(valid_tensors[i:i + batch_size]).to(device)
        with torch.no_grad():
            features = model(mini)
            logits = probe(features)
            probs = torch.softmax(logits, dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())

    # Map back including None entries
    idx = 0
    for t in batch:
        if t is None:
            class_ids.append(0)
            class_confs.append(0.0)
        else:
            class_ids.append(int(all_preds[idx]))
            class_confs.append(float(all_confs[idx]))
            idx += 1

    return np.array(class_ids), np.array(class_confs)


def main():
    import onnxruntime as ort

    print("="*50)
    print("PSEUDO-LABELING PIPELINE")
    print("="*50)

    # Find ONNX model
    onnx_candidates = [
        SUBMISSION_DIR / "best.onnx",
        YOLO_APPROACH / "best.onnx",
        Path(__file__).parent.parent / "best.onnx",
    ]
    onnx_path = None
    for p in onnx_candidates:
        if p.exists():
            onnx_path = p
            break

    if not onnx_path:
        print("ERROR: No ONNX model found!")
        print(f"Searched: {onnx_candidates}")
        return

    print(f"ONNX model: {onnx_path}")

    # Load ONNX session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    active_provider = session.get_providers()[0]
    print(f"Provider: {active_provider}")

    # Check for DINOv2 + probe
    probe_path = SUBMISSION_DIR / "linear_probe.pth"
    dino_path = SUBMISSION_DIR / "dinov2_vits14.pth"
    use_classifier = probe_path.exists()

    # Check GPU memory - use CPU if GPU is occupied
    device = "cpu"
    if torch.cuda.is_available():
        free_mem = torch.cuda.mem_get_info()[0] / 1024**3
        if free_mem > 2.0:
            device = "cuda"
        else:
            print(f"GPU has only {free_mem:.1f}GB free, using CPU")
    print(f"Device: {device}")

    dino_model = None
    dino_transform = None
    probe = None
    if use_classifier:
        import timm
        from timm.data import create_transform, resolve_data_config
        print(f"Loading DINOv2 classifier...")
        dino_model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False, num_classes=0)
        if dino_path.exists():
            dino_model.load_state_dict(torch.load(str(dino_path), map_location=device, weights_only=True), strict=False)
        dino_model = dino_model.to(device).eval()
        config = resolve_data_config(dino_model.pretrained_cfg)
        dino_transform = create_transform(**config, is_training=False)

        probe_data = torch.load(str(probe_path), map_location=device, weights_only=True)
        num_classes, embed_dim = probe_data["weight"].shape
        probe = torch.nn.Linear(embed_dim, num_classes)
        probe.load_state_dict(probe_data)
        probe = probe.to(device).eval()
        print(f"DINOv2 classifier loaded (embed_dim={embed_dim}, classes={num_classes})")
    else:
        print("No DINOv2 classifier found - using YOLO class predictions only")

    # Setup output
    OUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUT_LABELS.mkdir(parents=True, exist_ok=True)

    # Process store photos
    store_images = sorted(STORE_PHOTOS.glob("*.jpg")) + sorted(STORE_PHOTOS.glob("*.jpeg")) + sorted(STORE_PHOTOS.glob("*.png"))
    print(f"\nProcessing {len(store_images)} store photos...")

    total_images = 0
    total_annotations = 0
    total_filtered = 0

    for img_path in store_images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  SKIP: cannot read {img_path.name}")
            continue

        # Run detection
        boxes, yolo_classes, det_scores = run_yolo_onnx(session, img_bgr)

        if len(boxes) == 0:
            print(f"  {img_path.name}: 0 detections")
            continue

        # Run classification if available
        if dino_model is not None and probe is not None and dino_transform is not None:
            cls_ids, cls_confs = run_dino_classifier(dino_model, dino_transform, probe, img_bgr, boxes, device)

            # Use DINOv2 classification where confident enough, else YOLO class
            final_classes = []
            for i in range(len(boxes)):
                if cls_confs[i] > CLS_CONF_THRESHOLD:
                    final_classes.append(cls_ids[i])
                else:
                    final_classes.append(yolo_classes[i])
            final_classes = np.array(final_classes)
        else:
            final_classes = yolo_classes

        # Filter by detection confidence
        high_conf_mask = det_scores > DET_CONF_THRESHOLD
        boxes = boxes[high_conf_mask]
        final_classes = final_classes[high_conf_mask]
        det_scores = det_scores[high_conf_mask]

        filtered_out = (~high_conf_mask).sum()
        total_filtered += filtered_out

        if len(boxes) == 0:
            continue

        # Convert to YOLO format and save
        h_img, w_img = img_bgr.shape[:2]
        labels = []
        for box, cls_id in zip(boxes, final_classes):
            x1, y1, x2, y2 = box
            cx = ((x1 + x2) / 2) / w_img
            cy = ((y1 + y2) / 2) / h_img
            bw = (x2 - x1) / w_img
            bh = (y2 - y1) / h_img

            # Clip to [0, 1]
            cx = np.clip(cx, 0, 1)
            cy = np.clip(cy, 0, 1)
            bw = np.clip(bw, 0, 1)
            bh = np.clip(bh, 0, 1)

            labels.append(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

        # Copy image and save labels
        fname = f"pseudo_{img_path.stem}"
        import shutil
        shutil.copy2(str(img_path), str(OUT_IMAGES / f"{fname}.jpg"))
        with open(OUT_LABELS / f"{fname}.txt", "w") as f:
            f.write("\n".join(labels) + "\n")

        total_images += 1
        total_annotations += len(labels)
        print(f"  {img_path.name}: {len(labels)} labels (filtered {filtered_out})")

    print(f"\n{'='*50}")
    print(f"PSEUDO-LABELING COMPLETE")
    print(f"{'='*50}")
    print(f"Images labeled: {total_images}")
    print(f"Total annotations: {total_annotations}")
    print(f"Filtered out (low conf): {total_filtered}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

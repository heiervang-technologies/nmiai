#!/usr/bin/env python3
"""Honest evaluator: evaluate ANY model on ALL 248 competition images.

This is the single source of truth for model quality. It uses the full
competition dataset as validation and supports both ONNX and PyTorch models.

Metrics:
  - detection_map50: class-agnostic mAP@0.5 (70% of competition score)
  - classification_map50: class-aware mAP@0.5 (30% of competition score)
  - combined_map: 0.7 * det + 0.3 * cls

Usage:
  python eval_honest.py model.onnx
  python eval_honest.py model.pt
  python eval_honest.py model.pt --leakage-check /path/to/train/images/
  python eval_honest.py model.pt --experiment-id exp001 --description "Polish det-only" --unique-images 27246
"""
import argparse
import csv
import functools
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

print = functools.partial(print, flush=True)

SCRIPT_DIR = Path(__file__).resolve().parent
COCO_DIR = SCRIPT_DIR / "data-creation/data/coco_dataset/train"
COCO_IMAGES = COCO_DIR / "images"
COCO_ANNOTATIONS = COCO_DIR / "annotations.json"
RESULTS_TSV = SCRIPT_DIR / "data_experiment_results.tsv"

CONF_THRESH = 0.001
IOU_THRESH = 0.45
MAX_DET = 300
INPUT_SIZE = 1280
NUM_CLASSES = 356


# ── Geometry helpers ──────────────────────────────────────────────────

def iou(a, b):
    """IoU between two xyxy boxes."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def voc_ap(recalls, precisions):
    """VOC-style average precision (interpolated)."""
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    return sum(
        (mrec[i] - mrec[i - 1]) * mpre[i]
        for i in range(1, len(mrec))
        if mrec[i] != mrec[i - 1]
    )


# ── mAP computation ──────────────────────────────────────────────────

def evaluate_map(preds_by_img, gt_by_img, cat_ids, class_aware):
    """Compute mAP@0.5 (detection or classification)."""
    gt_index = {}
    positives = defaultdict(int)
    for img_id, anns in gt_by_img.items():
        gt_index[img_id] = [
            {"category_id": a["category_id"], "bbox": a["bbox"], "used": False}
            for a in anns
        ]
        for a in anns:
            positives[a["category_id"]] += 1

    ap_by_class = {}
    for cat in sorted(cat_ids):
        dets = []
        for img_id, preds in preds_by_img.items():
            for p in preds:
                if class_aware and p["category_id"] != cat:
                    continue
                dets.append((img_id, p))
        dets.sort(key=lambda x: x[1]["score"], reverse=True)

        tp_list, fp_list = [], []
        for img_id, pred in dets:
            cands = gt_index.get(img_id, [])
            best_iou, best_idx = 0, -1
            for idx, gt in enumerate(cands):
                if gt["used"]:
                    continue
                if class_aware and gt["category_id"] != cat:
                    continue
                o = iou(pred["bbox"], gt["bbox"])
                if o > best_iou:
                    best_iou, best_idx = o, idx
            if best_idx >= 0 and best_iou >= 0.5:
                cands[best_idx]["used"] = True
                tp_list.append(1)
                fp_list.append(0)
            else:
                tp_list.append(0)
                fp_list.append(1)

        total_pos = positives[cat]
        if total_pos == 0:
            continue
        if not dets:
            ap_by_class[cat] = 0.0
            continue

        cum_tp, cum_fp, rtp, rfp = [], [], 0, 0
        for t, f in zip(tp_list, fp_list):
            rtp += t
            rfp += f
            cum_tp.append(rtp)
            cum_fp.append(rfp)
        recalls = [t / total_pos for t in cum_tp]
        precisions = [t / max(t + f, 1) for t, f in zip(cum_tp, cum_fp)]
        ap_by_class[cat] = voc_ap(recalls, precisions)

    # Reset used flags
    for img_id in gt_index:
        for gt in gt_index[img_id]:
            gt["used"] = False

    return ap_by_class


# ── Ground truth loading ─────────────────────────────────────────────

def load_coco_gt():
    """Load ALL 248 competition images as ground truth (xyxy format)."""
    with open(COCO_ANNOTATIONS) as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img for img in coco["images"]}
    gt_by_img = defaultdict(list)
    cat_ids = set()
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}

    for ann in coco["annotations"]:
        img_info = id_to_file[ann["image_id"]]
        fname = img_info["file_name"]
        bx, by, bw, bh = ann["bbox"]
        gt_by_img[fname].append({
            "category_id": ann["category_id"],
            "bbox": [bx, by, bx + bw, by + bh],
        })
        cat_ids.add(ann["category_id"])

    return gt_by_img, cat_ids, id_to_file, cat_names


# ── Inference ────────────────────────────────────────────────────────

def letterbox(img, new_shape=INPUT_SIZE):
    h, w = img.shape[:2]
    ratio = new_shape / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (new_shape - new_w) / 2
    dh = (new_shape - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(
        img_resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    return img_padded, ratio, (dw, dh)


def postprocess_onnx(output, ratio, dw, dh, orig_h, orig_w):
    preds = output[0].squeeze(0).T
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    mask = confidences > CONF_THRESH
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    class_ids, confidences = class_ids[mask], confidences[mask]

    x1 = (cx - bw / 2 - dw) / ratio
    y1 = (cy - bh / 2 - dh) / ratio
    x2 = (cx + bw / 2 - dw) / ratio
    y2 = (cy + bh / 2 - dh) / ratio
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    boxes_xywh = [
        [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
        for b in boxes
    ]
    if len(boxes_xywh) == 0:
        return []
    indices = cv2.dnn.NMSBoxesBatched(
        boxes_xywh, confidences.tolist(), class_ids.tolist(),
        CONF_THRESH, IOU_THRESH
    )
    if len(indices) == 0:
        return []
    indices = indices.flatten()[:MAX_DET]
    return [
        {
            "bbox": [float(boxes[i][0]), float(boxes[i][1]),
                     float(boxes[i][2]), float(boxes[i][3])],  # xyxy
            "category_id": int(class_ids[i]),
            "score": float(confidences[i]),
        }
        for i in indices
    ]


def run_onnx(model_path, images):
    """Run ONNX model on images, return predictions by image."""
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    print(f"ONNX provider: {session.get_providers()[0]}")

    preds_by_img = defaultdict(list)
    total_dets = 0

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        lb, ratio, (dw, dh) = letterbox(img)
        blob = lb[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        outputs = session.run(None, {input_name: blob})
        dets = postprocess_onnx(outputs, ratio, dw, dh, h, w)
        preds_by_img[img_path.name] = dets
        total_dets += len(dets)

    return preds_by_img, total_dets


def run_ultralytics(model_path, images):
    """Run ultralytics YOLO .pt model on images."""
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    preds_by_img = defaultdict(list)
    total_dets = 0

    results = model.predict(
        source=[str(p) for p in images],
        imgsz=INPUT_SIZE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        max_det=MAX_DET,
        verbose=False,
        device=0,
    )

    for result in results:
        fname = Path(result.path).name
        boxes = result.boxes
        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
            preds_by_img[fname].append({
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "category_id": int(boxes.cls[i].item()),
                "score": float(boxes.conf[i].item()),
            })
            total_dets += 1

    return preds_by_img, total_dets


# ── Leakage check ───────────────────────────────────────────────────

def check_leakage(train_dir, val_filenames):
    """Verify zero overlap between training images and val set."""
    train_dir = Path(train_dir)
    if not train_dir.exists():
        print(f"WARNING: train dir {train_dir} does not exist")
        return True

    train_files = {p.name for p in train_dir.glob("*.jpg")}
    train_files |= {p.name for p in train_dir.glob("*.png")}
    overlap = train_files & val_filenames
    if overlap:
        print(f"\n*** LEAKAGE DETECTED: {len(overlap)} images in both train and val ***")
        for f in sorted(overlap)[:10]:
            print(f"  - {f}")
        return False
    print(f"Leakage check PASSED: 0 overlap ({len(train_files)} train, {len(val_filenames)} val)")
    return True


# ── Results logging ──────────────────────────────────────────────────

TSV_HEADER = [
    "timestamp", "experiment_id", "combined_map", "det_map50", "cls_map50",
    "unique_images", "aug_images", "data_sources", "description",
]


def append_result(result):
    """Append a result row to the TSV."""
    exists = RESULTS_TSV.exists()
    with open(RESULTS_TSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TSV_HEADER, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(result)
    print(f"Results appended to {RESULTS_TSV}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Honest evaluation on all 248 competition images")
    parser.add_argument("model", help="Path to model (.onnx or .pt)")
    parser.add_argument("--leakage-check", help="Path to training images dir (verify zero overlap)")
    parser.add_argument("--experiment-id", default="", help="Experiment ID for logging")
    parser.add_argument("--description", default="", help="Short description for logging")
    parser.add_argument("--unique-images", type=int, default=0, help="Number of unique training images")
    parser.add_argument("--aug-images", type=int, default=0, help="Number of augmented copies")
    parser.add_argument("--data-sources", default="", help="Comma-separated data sources")
    parser.add_argument("--no-log", action="store_true", help="Don't append to results TSV")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    # Load ground truth
    print("Loading COCO ground truth (all 248 competition images)...")
    gt_by_img, cat_ids, id_to_file, cat_names = load_coco_gt()
    total_gt = sum(len(v) for v in gt_by_img.values())
    print(f"GT: {total_gt} boxes, {len(gt_by_img)} images, {len(cat_ids)} categories")

    # Leakage check
    val_filenames = set(gt_by_img.keys())
    if args.leakage_check:
        if not check_leakage(args.leakage_check, val_filenames):
            print("ABORTING: leakage detected")
            return

    # Run inference
    images = sorted(COCO_IMAGES.glob("*.jpg")) + sorted(COCO_IMAGES.glob("*.jpeg"))
    if len(images) != len(gt_by_img):
        print(f"WARNING: Expected {len(gt_by_img)} images, found {len(images)} on disk")
    print(f"\nRunning inference on {len(images)} images...")

    t0 = time.time()
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        preds_by_img, total_dets = run_onnx(model_path, images)
    elif suffix == ".pt":
        preds_by_img, total_dets = run_ultralytics(model_path, images)
    else:
        print(f"Unsupported model format: {suffix}")
        return
    elapsed = time.time() - t0
    print(f"Inference: {elapsed:.1f}s, {total_dets} detections ({elapsed / len(images) * 1000:.0f}ms/img)")

    # Compute mAP
    print("\nComputing mAP@0.5...")
    ap_det = evaluate_map(preds_by_img, gt_by_img, cat_ids, class_aware=False)
    ap_cls = evaluate_map(preds_by_img, gt_by_img, cat_ids, class_aware=True)

    det_map = sum(ap_det.values()) / len(ap_det) if ap_det else 0
    cls_map = sum(ap_cls.values()) / len(ap_cls) if ap_cls else 0
    combined = 0.7 * det_map + 0.3 * cls_map

    # Print results
    print(f"\n{'=' * 60}")
    print(f"HONEST EVALUATION — {model_path.name}")
    print(f"{'=' * 60}")
    print(f"  detection_map50:      {det_map:.6f}")
    print(f"  classification_map50: {cls_map:.6f}")
    print(f"  combined_map:         {combined:.6f}")
    print(f"  inference_sec:        {elapsed:.1f}")
    print(f"{'=' * 60}")

    # Per-category breakdown (bottom 20 by classification AP)
    print(f"\n--- Bottom 20 categories (by classification AP) ---")
    sorted_cats = sorted(ap_cls.items(), key=lambda x: x[1])
    for cat_id, ap in sorted_cats[:20]:
        name = cat_names.get(cat_id, f"cat_{cat_id}")
        det_ap = ap_det.get(cat_id, 0)
        gt_count = sum(1 for anns in gt_by_img.values() for a in anns if a["category_id"] == cat_id)
        print(f"  cat {cat_id:>3d} ({gt_count:>3d} gt): det={det_ap:.3f} cls={ap:.3f}  {name[:50]}")

    # Log results
    if not args.no_log and args.experiment_id:
        row = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "experiment_id": args.experiment_id,
            "combined_map": f"{combined:.6f}",
            "det_map50": f"{det_map:.6f}",
            "cls_map50": f"{cls_map:.6f}",
            "unique_images": args.unique_images,
            "aug_images": args.aug_images,
            "data_sources": args.data_sources,
            "description": args.description,
        }
        append_result(row)

    return {"det_map": det_map, "cls_map": cls_map, "combined": combined}


if __name__ == "__main__":
    main()

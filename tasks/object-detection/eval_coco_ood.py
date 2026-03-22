"""Evaluate YOLO ONNX models on COCO competition dataset.
OOD generalization test: how well do models transfer to competition data?

Usage: python eval_coco_ood.py [model1.onnx model2.onnx ...]
"""
import json
import sys
import time
import functools
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

print = functools.partial(print, flush=True)

CONF_THRESH = 0.001
IOU_THRESH = 0.45
MAX_DET = 300
INPUT_SIZE = 1280
NUM_CLASSES = 356

COCO_DIR = Path(__file__).parent / "data-creation/data/coco_dataset/train"
COCO_IMAGES = COCO_DIR / "images"
COCO_ANNOTATIONS = COCO_DIR / "annotations.json"


def letterbox(img, new_shape=INPUT_SIZE):
    h, w = img.shape[:2]
    ratio = new_shape / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = (new_shape - new_w) / 2
    dh = (new_shape - new_h) / 2
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                     cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img_padded, ratio, (dw, dh)


def postprocess(output, ratio, dw, dh, orig_h, orig_w):
    preds = output[0].squeeze(0).T
    cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    class_scores = preds[:, 4:]
    class_ids = np.argmax(class_scores, axis=1)
    confidences = np.max(class_scores, axis=1)
    mask = confidences > CONF_THRESH
    cx, cy, bw, bh = cx[mask], cy[mask], bw[mask], bh[mask]
    class_ids, confidences = class_ids[mask], confidences[mask]

    x1 = (cx - bw/2 - dw) / ratio
    y1 = (cy - bh/2 - dh) / ratio
    x2 = (cx + bw/2 - dw) / ratio
    y2 = (cy + bh/2 - dh) / ratio
    x1 = np.clip(x1, 0, orig_w); y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w); y2 = np.clip(y2, 0, orig_h)
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    boxes_xywh = [[float(b[0]), float(b[1]), float(b[2]-b[0]), float(b[3]-b[1])] for b in boxes]
    if len(boxes_xywh) == 0:
        return []
    indices = cv2.dnn.NMSBoxesBatched(boxes_xywh, confidences.tolist(), class_ids.tolist(), CONF_THRESH, IOU_THRESH)
    if len(indices) == 0:
        return []
    indices = indices.flatten()[:MAX_DET]
    return [{"bbox": [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]-boxes[i][0]), float(boxes[i][3]-boxes[i][1])],
             "category_id": int(class_ids[i]), "score": float(confidences[i])} for i in indices]


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def voc_ap(recalls, precisions):
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    return sum((mrec[i]-mrec[i-1]) * mpre[i] for i in range(1, len(mrec)) if mrec[i] != mrec[i-1])


def evaluate_map(preds_by_img, gt_by_img, cat_ids, class_aware):
    gt_index = {}
    positives = defaultdict(int)
    for img_id, anns in gt_by_img.items():
        gt_index[img_id] = [{"category_id": a["category_id"], "bbox": a["bbox"], "used": False} for a in anns]
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
                if gt["used"]: continue
                if class_aware and gt["category_id"] != cat: continue
                o = iou(pred["bbox"], gt["bbox"])
                if o > best_iou: best_iou, best_idx = o, idx
            if best_idx >= 0 and best_iou >= 0.5:
                cands[best_idx]["used"] = True
                tp_list.append(1); fp_list.append(0)
            else:
                tp_list.append(0); fp_list.append(1)

        total_pos = positives[cat]
        if total_pos == 0: continue
        if not dets: ap_by_class[cat] = 0.0; continue

        cum_tp, cum_fp, rtp, rfp = [], [], 0, 0
        for t, f in zip(tp_list, fp_list):
            rtp += t; rfp += f; cum_tp.append(rtp); cum_fp.append(rfp)
        recalls = [t/total_pos for t in cum_tp]
        precisions = [t/max(t+f,1) for t,f in zip(cum_tp, cum_fp)]
        ap_by_class[cat] = voc_ap(recalls, precisions)

    # Reset used flags for next eval
    for img_id in gt_index:
        for gt in gt_index[img_id]:
            gt["used"] = False

    return sum(ap_by_class.values()) / len(ap_by_class) if ap_by_class else 0


def load_coco_gt():
    """Load COCO ground truth in xyxy format."""
    with open(COCO_ANNOTATIONS) as f:
        coco = json.load(f)

    id_to_file = {img["id"]: img for img in coco["images"]}
    gt_by_img = defaultdict(list)
    cat_ids = set()

    for ann in coco["annotations"]:
        img_info = id_to_file[ann["image_id"]]
        fname = img_info["file_name"]
        bx, by, bw, bh = ann["bbox"]  # COCO format: x,y,w,h
        gt_by_img[fname].append({
            "category_id": ann["category_id"],
            "bbox": [bx, by, bx + bw, by + bh],  # xyxy
        })
        cat_ids.add(ann["category_id"])

    return gt_by_img, cat_ids, id_to_file


def eval_model(model_path, gt_by_img, cat_ids):
    """Run inference + eval for one model."""
    name = str(model_path)
    print(f"\n{'='*60}")
    print(f"Evaluating: {name}")
    print(f"{'='*60}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name
    provider = session.get_providers()[0]
    print(f"Provider: {provider}")

    images = sorted(COCO_IMAGES.glob("*.jpg"))
    print(f"Images: {len(images)}")

    preds_by_img = defaultdict(list)
    total_dets = 0
    t0 = time.time()

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None: continue
        h, w = img.shape[:2]
        lb, ratio, (dw, dh) = letterbox(img)
        blob = lb[:,:,::-1].transpose(2,0,1).astype(np.float32) / 255.0
        blob = np.expand_dims(blob, 0)
        outputs = session.run(None, {input_name: blob})
        dets = postprocess(outputs, ratio, dw, dh, h, w)
        for d in dets:
            x, y, bw, bh = d["bbox"]
            preds_by_img[img_path.name].append({
                "category_id": d["category_id"],
                "score": d["score"],
                "bbox": [x, y, x+bw, y+bh],
            })
        total_dets += len(dets)

    elapsed = time.time() - t0
    print(f"Inference: {elapsed:.1f}s, {total_dets} detections")

    det_map = evaluate_map(preds_by_img, gt_by_img, cat_ids, class_aware=False)
    cls_map = evaluate_map(preds_by_img, gt_by_img, cat_ids, class_aware=True)
    combined = 0.7 * det_map + 0.3 * cls_map

    print(f"\nRESULTS for {Path(model_path).parent.parent.name}/{Path(model_path).parent.name}:")
    print(f"  Detection  mAP@0.5: {det_map*100:.1f}%")
    print(f"  Classification mAP@0.5: {cls_map*100:.1f}%")
    print(f"  Combined (0.7*det + 0.3*cls): {combined*100:.1f}%")
    print(f"  Inference time: {elapsed:.1f}s ({elapsed/len(images)*1000:.0f}ms/img)")

    return {"model": name, "det_map": det_map, "cls_map": cls_map, "combined": combined, "time": elapsed}


def main():
    # Default models to evaluate
    default_models = [
        Path("/home/me/ht/nmiai/runs/detect/yolov8x_v6_clean/weights/best.onnx"),
        Path("/home/me/ht/nmiai/runs/detect/yolov8x_v7_alldata/weights/best.onnx"),
    ]

    models = [Path(m) for m in sys.argv[1:]] if len(sys.argv) > 1 else default_models
    models = [m for m in models if m.exists()]

    if not models:
        print("No models found!")
        return

    print("Loading COCO ground truth...")
    gt_by_img, cat_ids, _ = load_coco_gt()
    print(f"GT: {sum(len(v) for v in gt_by_img.values())} boxes, {len(gt_by_img)} images, {len(cat_ids)} categories")

    results = []
    for model_path in models:
        r = eval_model(model_path, gt_by_img, cat_ids)
        results.append(r)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY - OOD COCO Evaluation")
    print(f"{'='*60}")
    print(f"{'Model':<50} {'Det%':>6} {'Cls%':>6} {'Comb%':>6}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["combined"], reverse=True):
        name = Path(r["model"]).parent.parent.name + "/" + Path(r["model"]).parent.name
        print(f"{name:<50} {r['det_map']*100:>5.1f}% {r['cls_map']*100:>5.1f}% {r['combined']*100:>5.1f}%")


if __name__ == "__main__":
    main()

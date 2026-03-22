"""NM i AI 2026 - Object Detection Submission (ONNX Runtime)
No ultralytics dependency. Uses only onnxruntime-gpu, numpy, cv2.
"""
import argparse
import json
import pathlib
import numpy as np
import cv2


def letterbox(img, new_shape=1280, color=(114, 114, 114)):
    """Resize and pad image to target shape maintaining aspect ratio."""
    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)


def nms(boxes, scores, iou_threshold=0.45):
    """Simple NMS implementation using numpy."""
    if len(boxes) == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 0] + boxes[:, 2]
    y2 = boxes[:, 1] + boxes[:, 3]
    areas = boxes[:, 2] * boxes[:, 3]
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep


def postprocess(output, orig_shape, ratio, pad, conf_thresh=0.001, iou_thresh=0.45, max_det=300):
    """Post-process YOLOv8 ONNX output."""
    preds = output[0]
    if preds.ndim == 3:
        preds = preds[0]

    # YOLOv8 ONNX output shape: [4+nc, num_boxes] -> transpose to [num_boxes, 4+nc]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T

    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4:]
    nc = class_scores.shape[1]

    # Get max class score and class id per box
    max_scores = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    # Filter by confidence
    mask = max_scores > conf_thresh
    boxes_xywh = boxes_xywh[mask]
    max_scores = max_scores[mask]
    class_ids = class_ids[mask]

    if len(boxes_xywh) == 0:
        return [], [], []

    # Convert from center x,y,w,h to x1,y1,w,h (COCO format)
    boxes_coco = np.zeros_like(boxes_xywh)
    boxes_coco[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2  # x1
    boxes_coco[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2  # y1
    boxes_coco[:, 2] = boxes_xywh[:, 2]  # w
    boxes_coco[:, 3] = boxes_xywh[:, 3]  # h

    # Scale boxes back to original image coordinates
    dw, dh = pad
    boxes_coco[:, 0] = (boxes_coco[:, 0] - dw) / ratio
    boxes_coco[:, 1] = (boxes_coco[:, 1] - dh) / ratio
    boxes_coco[:, 2] = boxes_coco[:, 2] / ratio
    boxes_coco[:, 3] = boxes_coco[:, 3] / ratio

    # Clip to image bounds
    h_orig, w_orig = orig_shape
    boxes_coco[:, 0] = np.clip(boxes_coco[:, 0], 0, w_orig)
    boxes_coco[:, 1] = np.clip(boxes_coco[:, 1], 0, h_orig)
    boxes_coco[:, 2] = np.clip(boxes_coco[:, 2], 0, w_orig - boxes_coco[:, 0])
    boxes_coco[:, 3] = np.clip(boxes_coco[:, 3], 0, h_orig - boxes_coco[:, 1])

    # Filter tiny boxes
    valid = (boxes_coco[:, 2] > 2) & (boxes_coco[:, 3] > 2)
    boxes_coco = boxes_coco[valid]
    max_scores = max_scores[valid]
    class_ids = class_ids[valid]

    # NMS per class
    keep_all = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_boxes = boxes_coco[cls_mask]
        cls_scores = max_scores[cls_mask]
        cls_indices = np.where(cls_mask)[0]
        keep = nms(cls_boxes, cls_scores, iou_thresh)
        keep_all.extend(cls_indices[keep].tolist())

    keep_all = sorted(keep_all, key=lambda i: -max_scores[i])[:max_det]

    return boxes_coco[keep_all].tolist(), max_scores[keep_all].tolist(), class_ids[keep_all].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    import onnxruntime as ort

    model_path = pathlib.Path(__file__).parent / 'best.onnx'
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_path), providers=providers)
    input_name = session.get_inputs()[0].name

    images_dir = pathlib.Path(args.input)
    image_files = sorted(
        list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.jpeg')) + list(images_dir.glob('*.png'))
    )

    results = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_shape = img.shape[:2]
        img_lb, ratio, pad = letterbox(img, 1280)
        img_input = img_lb[:, :, ::-1].transpose(2, 0, 1)
        img_input = np.ascontiguousarray(img_input, dtype=np.float32) / 255.0
        img_input = np.expand_dims(img_input, 0)

        outputs = session.run(None, {input_name: img_input})
        boxes, confs, class_ids = postprocess(outputs, orig_shape, ratio, pad)

        image_id = int(img_path.stem.replace("img_", ""))
        # Category aliases: merge umlaut spelling variants
        ALIASES = {59: 61, 170: 260, 36: 201}
        for box, conf, cls_id in zip(boxes, confs, class_ids):
            cat_id = int(cls_id)
            cat_id = ALIASES.get(cat_id, cat_id)
            results.append({
                'image_id': image_id,
                'bbox': [round(v, 2) for v in box],
                'category_id': cat_id,
                'score': round(float(conf), 4),
            })

    output_path = pathlib.Path(args.output)
    with open(str(output_path), 'w') as f:
        json.dump(results, f)


if __name__ == '__main__':
    main()

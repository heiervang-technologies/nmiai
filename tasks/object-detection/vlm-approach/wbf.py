"""
Minimal Weighted Boxes Fusion (WBF) implementation.

No external dependencies — uses only numpy.
Based on: https://arxiv.org/abs/1910.13302

Use this for TTA fusion (multi-scale + flip) or multi-model ensemble
without needing the ensemble_boxes package (not in sandbox).
"""

import numpy as np


def weighted_boxes_fusion(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    weights: list[float] | None = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    conf_type: str = "avg",  # "avg", "max", "box_and_model_avg"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weighted Boxes Fusion.

    Args:
        boxes_list: List of arrays, each [N, 4] with normalized coords [x1,y1,x2,y2] in [0,1].
        scores_list: List of arrays, each [N] confidence scores.
        labels_list: List of arrays, each [N] integer class labels.
        weights: Weight per model (len = len(boxes_list)). Default: equal weights.
        iou_thr: IoU threshold to consider boxes as same object.
        skip_box_thr: Skip boxes with score below this.
        conf_type: How to compute fused confidence.

    Returns:
        boxes: [M, 4] fused boxes.
        scores: [M] fused confidence scores.
        labels: [M] fused labels.
    """
    if weights is None:
        weights = [1.0] * len(boxes_list)

    # Collect all boxes with model index
    all_boxes = []
    for model_idx, (boxes, scores, labels) in enumerate(
        zip(boxes_list, scores_list, labels_list)
    ):
        for i in range(len(boxes)):
            if scores[i] < skip_box_thr:
                continue
            all_boxes.append({
                "box": boxes[i].copy(),
                "score": scores[i] * weights[model_idx],
                "label": int(labels[i]),
                "model_idx": model_idx,
            })

    if not all_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    # Sort by score descending
    all_boxes.sort(key=lambda x: -x["score"])

    # Group by label
    label_groups = {}
    for b in all_boxes:
        label_groups.setdefault(b["label"], []).append(b)

    fused_boxes = []
    fused_scores = []
    fused_labels = []

    for label, group_boxes in label_groups.items():
        clusters = []  # Each cluster is a list of boxes that overlap

        for b in group_boxes:
            matched = False
            for cluster in clusters:
                # Check IoU with the current fused box of this cluster
                fused = _fused_box(cluster)
                if _iou(b["box"], fused) > iou_thr:
                    cluster.append(b)
                    matched = True
                    break

            if not matched:
                clusters.append([b])

        # Fuse each cluster
        n_models = len(boxes_list)
        for cluster in clusters:
            # Weighted average of box coordinates
            total_weight = sum(b["score"] for b in cluster)
            if total_weight == 0:
                continue

            fused_box = np.zeros(4)
            for b in cluster:
                fused_box += b["box"] * b["score"]
            fused_box /= total_weight

            # Confidence
            if conf_type == "avg":
                score = total_weight / n_models
            elif conf_type == "max":
                score = max(b["score"] for b in cluster)
            elif conf_type == "box_and_model_avg":
                score = total_weight / max(1, len(cluster))
            else:
                score = total_weight / n_models

            fused_boxes.append(fused_box)
            fused_scores.append(score)
            fused_labels.append(label)

    if not fused_boxes:
        return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

    return (
        np.array(fused_boxes),
        np.array(fused_scores),
        np.array(fused_labels, dtype=int),
    )


def _fused_box(cluster: list[dict]) -> np.ndarray:
    """Compute weighted average box for a cluster."""
    total_weight = sum(b["score"] for b in cluster)
    if total_weight == 0:
        return cluster[0]["box"]
    fused = np.zeros(4)
    for b in cluster:
        fused += b["box"] * b["score"]
    return fused / total_weight


def _iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0

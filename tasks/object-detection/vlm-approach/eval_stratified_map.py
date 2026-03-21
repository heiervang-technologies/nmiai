"""Evaluate a submission against the stratified validation split.

Supports both zipped submissions and unpacked submission directories. The
output includes detection/classification mAP@0.5, runtime, and optional JSONL/
CSV append modes so the same evaluator can back leaderboards and checkpoint
watchers.
"""

import argparse
import csv
import json
import subprocess
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def discover_images(images_dir: Path) -> list[Path]:
    return sorted(
        [path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES],
        key=lambda path: path.name,
    )


def build_image_lookup(image_paths: list[Path]) -> dict[object, str]:
    lookup = {}
    for path in image_paths:
        lookup[path.name] = path.name
        lookup[path.stem] = path.name
        stem = path.stem
        if stem.startswith("img_"):
            suffix = stem.removeprefix("img_")
            if suffix.isdigit():
                lookup[int(suffix)] = path.name
                lookup[suffix] = path.name
    return lookup


def normalize_image_id(raw_image_id, image_lookup: dict[object, str]) -> str:
    if raw_image_id in image_lookup:
        return image_lookup[raw_image_id]
    if isinstance(raw_image_id, str):
        stripped = raw_image_id.strip()
        if stripped in image_lookup:
            return image_lookup[stripped]
        if stripped.isdigit():
            numeric = int(stripped)
            if numeric in image_lookup:
                return image_lookup[numeric]
            padded = f"img_{numeric:05d}"
            if padded in image_lookup:
                return image_lookup[padded]
        return stripped
    if isinstance(raw_image_id, (int, float)):
        numeric = int(raw_image_id)
        if numeric in image_lookup:
            return image_lookup[numeric]
        padded = f"img_{numeric:05d}"
        if padded in image_lookup:
            return image_lookup[padded]
    return str(raw_image_id)


def load_names(dataset_yaml: Path) -> dict[int, str]:
    data = yaml.safe_load(dataset_yaml.read_text())
    names = data["names"]
    return {int(key): value for key, value in names.items()}


def read_yolo_labels(label_path: Path, image_size: tuple[int, int]):
    width, height = image_size
    records = []
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, w, h = line.split()
        class_id = int(float(class_id))
        cx = float(cx) * width
        cy = float(cy) * height
        w = float(w) * width
        h = float(h) * height
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        records.append({"category_id": class_id, "bbox": [x1, y1, x2, y2]})
    return records


def iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def voc_ap(recalls, precisions):
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    ap = 0.0
    for idx in range(1, len(mrec)):
        if mrec[idx] != mrec[idx - 1]:
            ap += (mrec[idx] - mrec[idx - 1]) * mpre[idx]
    return ap


def evaluate(predictions_by_image, ground_truth_by_image, category_ids, class_aware: bool):
    gt_index = {}
    positives = defaultdict(int)
    for image_id, annotations in ground_truth_by_image.items():
        gt_index[image_id] = []
        for ann in annotations:
            gt_index[image_id].append({
                "category_id": ann["category_id"],
                "bbox": ann["bbox"],
                "used": False,
            })
            positives[ann["category_id"]] += 1

    ap_by_class = {}
    matched_tp = 0
    matched_fp = 0

    for category_id in sorted(category_ids):
        detections = []
        for image_id, preds in predictions_by_image.items():
            for pred in preds:
                if class_aware and pred["category_id"] != category_id:
                    continue
                detections.append((image_id, pred))

        detections.sort(key=lambda item: item[1]["score"], reverse=True)
        true_positive = []
        false_positive = []

        for image_id, pred in detections:
            candidates = gt_index.get(image_id, [])
            best_iou = 0.0
            best_idx = -1
            for idx, gt in enumerate(candidates):
                if gt["used"]:
                    continue
                if class_aware and gt["category_id"] != category_id:
                    continue
                overlap = iou(pred["bbox"], gt["bbox"])
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx

            if best_idx >= 0 and best_iou >= 0.5:
                candidates[best_idx]["used"] = True
                true_positive.append(1)
                false_positive.append(0)
                matched_tp += 1
            else:
                true_positive.append(0)
                false_positive.append(1)
                matched_fp += 1

        total_pos = positives[category_id]
        if total_pos == 0:
            continue
        if not detections:
            ap_by_class[category_id] = 0.0
            continue

        tp_cum = []
        fp_cum = []
        running_tp = 0
        running_fp = 0
        for tp, fp in zip(true_positive, false_positive):
            running_tp += tp
            running_fp += fp
            tp_cum.append(running_tp)
            fp_cum.append(running_fp)

        recalls = [value / total_pos for value in tp_cum]
        precisions = [tp / max(tp + fp, 1) for tp, fp in zip(tp_cum, fp_cum)]
        ap_by_class[category_id] = voc_ap(recalls, precisions)

    mean_ap = sum(ap_by_class.values()) / len(ap_by_class) if ap_by_class else 0.0
    return mean_ap, ap_by_class, matched_tp, matched_fp, positives


def run_submission(submission_dir: Path, images_dir: Path, output_json: Path, timeout: int | None):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        str(submission_dir / "run.py"),
        "--input",
        str(images_dir),
        "--output",
        str(output_json),
    ]
    started = time.perf_counter()
    subprocess.run(command, check=True, timeout=timeout)
    return time.perf_counter() - started


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_csv(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp_utc",
        "model_name",
        "submission_type",
        "submission_path",
        "image_dir",
        "label_dir",
        "dataset_yaml",
        "usable_images",
        "image_files",
        "label_files",
        "missing_images_for_labels",
        "detection_map50",
        "classification_map50",
        "combined_score",
        "detection_tp",
        "detection_fp",
        "classification_tp",
        "classification_fp",
        "gt_boxes",
        "num_categories",
        "inference_seconds",
        "output_json",
    ]
    row = {key: record.get(key, "") for key in fieldnames}
    row["missing_images_for_labels"] = json.dumps(record.get("missing_images_for_labels", []), ensure_ascii=False)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def evaluate_submission(
    submission_path: Path,
    submission_type: str,
    images_dir: Path,
    labels_dir: Path,
    dataset_yaml: Path,
    output_json: Path,
    model_name: str | None = None,
    timeout: int | None = None,
    metadata: dict | None = None,
):
    names = load_names(dataset_yaml)
    image_paths = discover_images(images_dir)
    image_lookup = build_image_lookup(image_paths)
    label_paths = {path.stem: path for path in labels_dir.glob("*.txt")}
    usable_images = [path for path in image_paths if path.stem in label_paths]

    if submission_type == "zip":
        with tempfile.TemporaryDirectory(prefix="dense_eval_") as tmpdir:
            extract_dir = Path(tmpdir) / "submission"
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(submission_path) as zf:
                zf.extractall(extract_dir)
            inference_seconds = run_submission(extract_dir, images_dir, output_json, timeout=timeout)
    elif submission_type == "dir":
        inference_seconds = run_submission(submission_path, images_dir, output_json, timeout=timeout)
    else:
        raise ValueError(f"Unsupported submission_type: {submission_type}")

    predictions_raw = json.loads(output_json.read_text())
    predictions_by_image = defaultdict(list)
    for pred in predictions_raw:
        x, y, w, h = pred["bbox"]
        image_id = normalize_image_id(pred["image_id"], image_lookup)
        predictions_by_image[image_id].append(
            {
                "category_id": int(pred["category_id"]),
                "score": float(pred["score"]),
                "bbox": [x, y, x + w, y + h],
            }
        )

    from PIL import Image

    ground_truth_by_image = {}
    category_ids = set()
    for image_path in usable_images:
        with Image.open(image_path) as image:
            gt = read_yolo_labels(label_paths[image_path.stem], image.size)
        ground_truth_by_image[image_path.name] = gt
        for ann in gt:
            category_ids.add(ann["category_id"])

    detection_map, _, det_tp, det_fp, positives = evaluate(
        predictions_by_image,
        ground_truth_by_image,
        category_ids,
        class_aware=False,
    )
    classification_map, _, cls_tp, cls_fp, _ = evaluate(
        predictions_by_image,
        ground_truth_by_image,
        category_ids,
        class_aware=True,
    )
    combined_score = 0.7 * detection_map + 0.3 * classification_map

    result = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_name": model_name or submission_path.stem,
        "submission_type": submission_type,
        "submission_path": str(submission_path.resolve()),
        "image_dir": str(images_dir.resolve()),
        "label_dir": str(labels_dir.resolve()),
        "dataset_yaml": str(dataset_yaml.resolve()),
        "output_json": str(output_json.resolve()),
        "usable_images": len(usable_images),
        "label_files": len(label_paths),
        "image_files": len(image_paths),
        "missing_images_for_labels": sorted(list(set(label_paths) - {path.stem for path in image_paths})),
        "detection_map50": detection_map,
        "classification_map50": classification_map,
        "combined_score": combined_score,
        "detection_tp": det_tp,
        "detection_fp": det_fp,
        "classification_tp": cls_tp,
        "classification_fp": cls_fp,
        "gt_boxes": int(sum(positives.values())),
        "num_categories": len(category_ids),
        "category_names": {int(key): names[int(key)] for key in sorted(category_ids)},
        "inference_seconds": inference_seconds,
    }
    if metadata:
        result["metadata"] = metadata
    return result


def main():
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--zip", type=Path)
    source_group.add_argument("--submission-dir", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--dataset-yaml", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=Path("/tmp/stratified_eval_predictions.json"))
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--append-jsonl", type=Path)
    parser.add_argument("--append-csv", type=Path)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--metadata-json", type=str, help="Optional JSON object to attach to the result")
    args = parser.parse_args()

    submission_path = args.zip or args.submission_dir
    submission_type = "zip" if args.zip else "dir"
    metadata = json.loads(args.metadata_json) if args.metadata_json else None
    result = evaluate_submission(
        submission_path=submission_path,
        submission_type=submission_type,
        images_dir=args.images,
        labels_dir=args.labels,
        dataset_yaml=args.dataset_yaml,
        output_json=args.output_json,
        model_name=args.model_name,
        timeout=args.timeout,
        metadata=metadata,
    )
    if args.append_jsonl:
        append_jsonl(args.append_jsonl, result)
    if args.append_csv:
        append_csv(args.append_csv, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

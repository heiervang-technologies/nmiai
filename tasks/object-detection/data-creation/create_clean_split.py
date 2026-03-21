"""Create a leakage-free validation split rooted in the original 80/20 COCO holdout.

Why this script exists:
- `stratified_split/val` is heavily leaked into V3/V4/V5 train data.
- The existing `yolo_dataset` 198/50 split is the only current holdout with zero
  overlap against every derived YOLO dataset already built in this repo.

This script promotes that holdout to `data/clean_split/`, writes fresh metadata,
and emits a machine-readable leakage audit for downstream model selection.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
COCO_DIR = DATA_DIR / "coco_dataset" / "train"
COCO_ANN = COCO_DIR / "annotations.json"
SOURCE_SPLIT = DATA_DIR / "yolo_dataset"
OUT_DIR = DATA_DIR / "clean_split"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
DERIVED_DATASETS = [
    "yolo_augmented",
    "yolo_augmented_v2",
    "yolo_augmented_v3",
    "yolo_augmented_v4",
    "yolo_augmented_v5",
    "yolo_all_data",
    "yolo_final_with_val",
    "yolo_final_no_val",
    "stratified_split",
]


def image_files(path: Path) -> list[Path]:
    return sorted([p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES], key=lambda p: p.name)


def reset_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def load_coco():
    with COCO_ANN.open(encoding="utf-8") as handle:
        coco = json.load(handle)
    img_by_name = {img["file_name"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    cat_names = {int(cat["id"]): cat["name"] for cat in coco["categories"]}
    return img_by_name, anns_by_image, cat_names


def real_paths(files: list[Path]) -> set[str]:
    return {str(path.resolve()) for path in files}


def write_split(split: str, files: list[Path]):
    img_out = OUT_DIR / split / "images"
    lbl_out = OUT_DIR / split / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for src in files:
        dst = img_out / src.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src.resolve())

        src_label = SOURCE_SPLIT / split / "labels" / f"{src.stem}.txt"
        dst_label = lbl_out / src_label.name
        shutil.copy2(src_label, dst_label)


def compute_category_stats(files: list[Path], img_by_name: dict, anns_by_image: dict[int, list[dict]]):
    categories = Counter()
    image_categories = set()
    density = []
    for path in files:
        img = img_by_name[path.name]
        anns = anns_by_image[img["id"]]
        density.append(len(anns))
        cats = {int(ann["category_id"]) for ann in anns}
        image_categories.update(cats)
        for ann in anns:
            categories[int(ann["category_id"])] += 1
    return {
        "annotation_counts": categories,
        "image_category_coverage": image_categories,
        "density": density,
    }


def overlap_report(val_files: list[Path]):
    val_real = real_paths(val_files)
    checks = []
    for dataset_name in DERIVED_DATASETS:
        dataset_root = DATA_DIR / dataset_name
        check = {"dataset": dataset_name}
        for split in ["train", "val"]:
            images_dir = dataset_root / split / "images"
            if not images_dir.exists():
                continue
            overlap = [
                path.name
                for path in image_files(images_dir)
                if str(path.resolve()) in val_real
            ]
            check[f"{split}_overlap_count"] = len(overlap)
            check[f"{split}_overlap_sample"] = overlap[:20]
        checks.append(check)
    return checks


def main():
    if not SOURCE_SPLIT.exists():
        raise FileNotFoundError(f"Missing source split: {SOURCE_SPLIT}")

    train_files = image_files(SOURCE_SPLIT / "train" / "images")
    val_files = image_files(SOURCE_SPLIT / "val" / "images")
    if len(train_files) != 198 or len(val_files) != 50:
        raise ValueError(f"Expected 198/50 split, got {len(train_files)}/{len(val_files)}")

    reset_dir(OUT_DIR)
    write_split("train", train_files)
    write_split("val", val_files)

    img_by_name, anns_by_image, cat_names = load_coco()
    train_stats = compute_category_stats(train_files, img_by_name, anns_by_image)
    val_stats = compute_category_stats(val_files, img_by_name, anns_by_image)

    train_cov = train_stats["image_category_coverage"]
    val_cov = val_stats["image_category_coverage"]
    all_cov = train_cov | val_cov
    overlaps = overlap_report(val_files)

    dataset_yaml = {
        "path": str(OUT_DIR.resolve()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(cat_names),
        "names": {int(idx): cat_names[idx] for idx in sorted(cat_names)},
    }
    (OUT_DIR / "dataset.yaml").write_text(
        yaml.dump(dataset_yaml, default_flow_style=False, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    (OUT_DIR / "train_images.txt").write_text("\n".join(path.name for path in train_files) + "\n", encoding="utf-8")
    (OUT_DIR / "val_images.txt").write_text("\n".join(path.name for path in val_files) + "\n", encoding="utf-8")

    analysis = {
        "split_name": "clean_split",
        "created_from": str(SOURCE_SPLIT.resolve()),
        "rationale": [
            "This split reuses the original 198/50 COCO holdout because it is the only existing validation split with zero overlap against all currently built augmented datasets.",
            "A different 50-image holdout would require rebuilding V2/V3/V4/V5-derived datasets before it could be used for fair evaluation.",
            "Treat this as the immediate leakage-free model-selection split; ask data creation to rebuild future datasets against these held-out filenames.",
        ],
        "split_sizes": {"train": len(train_files), "val": len(val_files)},
        "category_coverage": {
            "total_categories": len(all_cov),
            "train_categories": len(train_cov),
            "val_categories": len(val_cov),
            "val_only_category_ids": sorted(val_cov - train_cov),
            "val_only_category_names": [cat_names[cid] for cid in sorted(val_cov - train_cov)],
            "train_only_category_ids": sorted(train_cov - val_cov),
            "train_only_category_names_sample": [cat_names[cid] for cid in sorted(train_cov - val_cov)[:40]],
            "train_only_count": len(train_cov - val_cov),
        },
        "density_stats": {
            "train_mean_annotations_per_image": sum(train_stats["density"]) / len(train_stats["density"]),
            "val_mean_annotations_per_image": sum(val_stats["density"]) / len(val_stats["density"]),
            "train_total_annotations": int(sum(train_stats["annotation_counts"].values())),
            "val_total_annotations": int(sum(val_stats["annotation_counts"].values())),
        },
        "stratification_note": "Category-aware quality is limited by the original source split. Leakage-free evaluation takes priority over replacing it with a new holdout before downstream datasets are rebuilt.",
        "zero_overlap_checks": overlaps,
        "val_images": [path.name for path in val_files],
    }
    (OUT_DIR / "split_analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    leakage_summary = {
        "reference_split": str((OUT_DIR / "val" / "images").resolve()),
        "reference_count": len(val_files),
        "status": "CLEAN_FOR_CURRENT_REPO_STATE",
        "checks": overlaps,
    }
    (OUT_DIR / "leakage_audit.json").write_text(json.dumps(leakage_summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({
        "clean_split": str(OUT_DIR.resolve()),
        "train_images": len(train_files),
        "val_images": len(val_files),
        "val_categories": len(val_cov),
        "train_categories": len(train_cov),
        "val_only_categories": len(val_cov - train_cov),
        "all_overlap_counts_zero": all(
            check.get("train_overlap_count", 0) == 0 and check.get("val_overlap_count", 0) == 0
            for check in overlaps
        ),
    }, indent=2))


if __name__ == "__main__":
    main()

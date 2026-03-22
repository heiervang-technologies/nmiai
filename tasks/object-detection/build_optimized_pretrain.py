#!/usr/bin/env python3
"""Build optimized pre-training subsets based on distribution analysis.

Key findings:
- Polish shelves is best match (density + bbox size)
- Grocery Shelves is poor match (too dense, too small boxes)
- SKU-110K is moderate (filter for density 50-150/img)

Builds multiple variants for A/B testing on vast.ai.
"""
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "data-creation" / "data"
COCO_ANN = BASE / "coco_dataset" / "train" / "annotations.json"
OUTPUT_BASE = BASE / "optimized_subsets"


def load_coco_density(ann_path):
    """Load annotations and compute per-image density."""
    with open(ann_path) as f:
        coco = json.load(f)
    img_lookup = {img["id"]: img for img in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)
    densities = {img_id: len(anns) for img_id, anns in anns_by_img.items()}
    return coco, img_lookup, anns_by_img, densities


def coco_to_yolo_single_class(coco, img_lookup, anns_by_img, img_dir,
                                out_images, out_labels, image_ids, prefix=""):
    """Convert selected images to single-class YOLO format."""
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    n_img = 0
    n_ann = 0
    for img_id in image_ids:
        img_info = img_lookup[img_id]
        fname = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        # Find image file
        src = img_dir / fname
        if not src.exists():
            for ext in [".jpg", ".jpeg", ".png"]:
                alt = img_dir / (Path(fname).stem + ext)
                if alt.exists():
                    src = alt
                    break
        if not src.exists():
            continue

        anns = anns_by_img.get(img_id, [])
        if not anns:
            continue

        yolo_lines = []
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            cx = max(0, min(1, (bx + bw/2) / w))
            cy = max(0, min(1, (by + bh/2) / h))
            nw = max(0.001, min(1, bw / w))
            nh = max(0.001, min(1, bh / h))
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_ann += 1

        dst_name = f"{prefix}{src.name}" if prefix else src.name
        shutil.copy2(src, out_images / dst_name)
        label_name = f"{prefix}{src.stem}.txt" if prefix else f"{src.stem}.txt"
        with open(out_labels / label_name, "w") as f:
            f.write("\n".join(yolo_lines) + "\n")
        n_img += 1

    return n_img, n_ann


def build_val(out_dir):
    """Build val set from all 248 competition images (single class for YOLO)."""
    with open(COCO_ANN) as f:
        coco = json.load(f)
    img_lookup = {img["id"]: img for img in coco["images"]}
    anns_by_img = defaultdict(list)
    for a in coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    val_imgs = out_dir / "val" / "images"
    val_lbls = out_dir / "val" / "labels"
    val_imgs.mkdir(parents=True, exist_ok=True)
    val_lbls.mkdir(parents=True, exist_ok=True)

    img_dir = BASE / "coco_dataset" / "train" / "images"
    n = 0
    for img_id, img_info in img_lookup.items():
        fname = img_info["file_name"]
        w, h = img_info["width"], img_info["height"]

        src = img_dir / fname
        if not src.exists():
            continue

        anns = anns_by_img.get(img_id, [])
        yolo_lines = []
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            cx = max(0, min(1, (bx + bw/2) / w))
            cy = max(0, min(1, (by + bh/2) / h))
            nw = max(0.001, min(1, bw / w))
            nh = max(0.001, min(1, bh / h))
            yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        if yolo_lines:
            shutil.copy2(src, val_imgs / fname)
            with open(val_lbls / (Path(fname).stem + ".txt"), "w") as f:
                f.write("\n".join(yolo_lines) + "\n")
            n += 1

    return n


def write_yaml(out_dir, name):
    yaml = f"path: {out_dir}\ntrain: train/images\nval: val/images\n\nnc: 1\nnames:\n  0: product\n"
    with open(out_dir / "dataset.yaml", "w") as f:
        f.write(yaml)


def main():
    random.seed(42)

    # Load all sources
    print("Loading Polish shelves...")
    pl_coco, pl_imgs, pl_anns, pl_density = load_coco_density(
        BASE / "external/skus_on_shelves_pl/extracted/annotations.json")
    pl_img_dir = BASE / "external/skus_on_shelves_pl/extracted"

    print("Loading SKU-110K...")
    sku_coco, sku_imgs, sku_anns, sku_density = load_coco_density(
        BASE / "external/sku110k_extracted/annotations.json")
    sku_img_dir = BASE / "external/sku110k_extracted/images"

    # === VARIANT A: Polish 5K only (best match) ===
    name = "v1_polish_5k"
    out = OUTPUT_BASE / name
    if out.exists(): shutil.rmtree(out)
    print(f"\n=== Building {name} ===")

    # Sample 5K images, preferring density 40-150 (competition-like range)
    good_density = [img_id for img_id, d in pl_density.items() if 40 <= d <= 150]
    random.shuffle(good_density)
    selected = good_density[:5000]
    if len(selected) < 5000:
        # Fill remaining from other images
        other = [img_id for img_id in pl_density if img_id not in set(selected)]
        random.shuffle(other)
        selected.extend(other[:5000 - len(selected)])

    n_img, n_ann = coco_to_yolo_single_class(
        pl_coco, pl_imgs, pl_anns, pl_img_dir,
        out / "train/images", out / "train/labels", selected, prefix="pl_")
    n_val = build_val(out)
    write_yaml(out, name)
    print(f"  Train: {n_img} images, {n_ann} annotations")
    print(f"  Val: {n_val} images")

    # === VARIANT B: Polish 5K + density-filtered SKU-110K ===
    name = "v2_polish5k_sku_filtered"
    out = OUTPUT_BASE / name
    if out.exists(): shutil.rmtree(out)
    print(f"\n=== Building {name} ===")

    n_img, n_ann = coco_to_yolo_single_class(
        pl_coco, pl_imgs, pl_anns, pl_img_dir,
        out / "train/images", out / "train/labels", selected, prefix="pl_")

    # Filter SKU-110K for density 50-150
    sku_filtered = [img_id for img_id, d in sku_density.items() if 50 <= d <= 150]
    random.shuffle(sku_filtered)
    sku_selected = sku_filtered[:1000]
    n_img2, n_ann2 = coco_to_yolo_single_class(
        sku_coco, sku_imgs, sku_anns, sku_img_dir,
        out / "train/images", out / "train/labels", sku_selected, prefix="sku_")

    n_val = build_val(out)
    write_yaml(out, name)
    print(f"  Train: {n_img + n_img2} images ({n_img} Polish + {n_img2} SKU filtered)")
    print(f"  Val: {n_val} images")

    # === VARIANT C: Polish 10K (max data, best source) ===
    name = "v3_polish_10k"
    out = OUTPUT_BASE / name
    if out.exists(): shutil.rmtree(out)
    print(f"\n=== Building {name} ===")

    # 10K with density filtering
    good_density_10k = [img_id for img_id, d in pl_density.items() if 30 <= d <= 200]
    random.shuffle(good_density_10k)
    selected_10k = good_density_10k[:10000]

    n_img, n_ann = coco_to_yolo_single_class(
        pl_coco, pl_imgs, pl_anns, pl_img_dir,
        out / "train/images", out / "train/labels", selected_10k, prefix="pl_")
    n_val = build_val(out)
    write_yaml(out, name)
    print(f"  Train: {n_img} images, {n_ann} annotations")
    print(f"  Val: {n_val} images")

    # === VARIANT D: SKU-110K only (control - how much does it help alone?) ===
    name = "v4_sku_only_filtered"
    out = OUTPUT_BASE / name
    if out.exists(): shutil.rmtree(out)
    print(f"\n=== Building {name} ===")

    sku_all_filtered = [img_id for img_id, d in sku_density.items() if 50 <= d <= 200]
    random.shuffle(sku_all_filtered)

    n_img, n_ann = coco_to_yolo_single_class(
        sku_coco, sku_imgs, sku_anns, sku_img_dir,
        out / "train/images", out / "train/labels", sku_all_filtered, prefix="sku_")
    n_val = build_val(out)
    write_yaml(out, name)
    print(f"  Train: {n_img} images, {n_ann} annotations")
    print(f"  Val: {n_val} images")

    print(f"\n{'='*60}")
    print(f"All variants built at: {OUTPUT_BASE}")
    print(f"Upload to HuggingFace with: python upload_pretrain_hf.py")


if __name__ == "__main__":
    main()

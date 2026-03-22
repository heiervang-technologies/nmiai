#!/usr/bin/env python3
"""Qualitative distribution analysis across shelf detection datasets.

Compares image properties, annotation density, and bbox characteristics
to understand how well external data matches competition distribution.

Usage: python analyze_distributions.py
"""
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
BASE = SCRIPT_DIR / "data-creation" / "data"

DATASETS = [
    {
        "name": "Competition (NorgesGruppen)",
        "images": BASE / "coco_dataset" / "train" / "images",
        "annotations": BASE / "coco_dataset" / "train" / "annotations.json",
        "color": "#2196F3",
    },
    {
        "name": "Polish Shelves (27K)",
        "images": BASE / "external" / "skus_on_shelves_pl" / "extracted",
        "annotations": BASE / "external" / "skus_on_shelves_pl" / "extracted" / "annotations.json",
        "color": "#FF9800",
    },
    {
        "name": "Grocery Shelves (45)",
        "images": BASE / "external" / "grocery_shelves" / "Supermarket shelves" / "Supermarket shelves" / "images",
        "annotations": BASE / "external" / "grocery_shelves" / "coco_annotations.json",
        "color": "#9C27B0",
    },
]


def analyze_coco(config):
    ann_path = Path(config["annotations"])
    if not ann_path.exists():
        return None

    with open(ann_path) as f:
        coco = json.load(f)

    img_lookup = {img["id"]: img for img in coco["images"]}
    anns_per_image = Counter()
    bbox_areas = []
    bbox_aspects = []
    bbox_rel_areas = []  # relative to image size

    for ann in coco["annotations"]:
        anns_per_image[ann["image_id"]] += 1
        bx, by, bw, bh = ann["bbox"]
        if bw > 0 and bh > 0:
            bbox_areas.append(bw * bh)
            bbox_aspects.append(bw / bh)
            img = img_lookup.get(ann["image_id"])
            if img:
                bbox_rel_areas.append((bw * bh) / (img["width"] * img["height"]))

    img_widths = [img["width"] for img in coco["images"]]
    img_heights = [img["height"] for img in coco["images"]]
    ann_counts = [anns_per_image.get(img["id"], 0) for img in coco["images"]]
    densities = []
    for img in coco["images"]:
        mpx = (img["width"] * img["height"]) / 1e6
        n = anns_per_image.get(img["id"], 0)
        if mpx > 0:
            densities.append(n / mpx)

    return {
        "name": config["name"],
        "color": config["color"],
        "n_images": len(coco["images"]),
        "n_annotations": len(coco["annotations"]),
        "n_categories": len(coco.get("categories", [])),
        "ann_counts": ann_counts,
        "bbox_areas": random.sample(bbox_areas, min(20000, len(bbox_areas))),
        "bbox_aspects": random.sample(bbox_aspects, min(20000, len(bbox_aspects))),
        "bbox_rel_areas": random.sample(bbox_rel_areas, min(20000, len(bbox_rel_areas))),
        "img_widths": img_widths,
        "img_heights": img_heights,
        "densities": densities,
    }


def create_sample_grid(config, output_path, n=12):
    img_dir = Path(config["images"])
    if not img_dir.exists():
        return
    imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.jpeg")) + list(img_dir.glob("*.png"))
    if not imgs:
        return
    random.seed(42)
    sample = random.sample(imgs, min(n, len(imgs)))

    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    fig.suptitle(f"Samples: {config['name']}", fontsize=14, fontweight="bold")

    for idx, ax in enumerate(axes.flat):
        if idx < len(sample):
            img = cv2.imread(str(sample[idx]))
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                scale = min(600 / w, 450 / h)
                img = cv2.resize(img, (int(w * scale), int(h * scale)))
                ax.imshow(img)
                ax.set_title(f"{w}x{h}", fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grid: {output_path}")


def main():
    random.seed(42)
    all_stats = []

    for ds in DATASETS:
        print(f"Analyzing {ds['name']}...")
        stats = analyze_coco(ds)
        if stats:
            all_stats.append(stats)
            print(f"  {stats['n_images']} imgs, {stats['n_annotations']} anns, "
                  f"{stats['n_categories']} cats, avg {np.mean(stats['ann_counts']):.0f}/img")

        grid_name = ds["name"].split("(")[0].strip().lower().replace(" ", "_")
        create_sample_grid(ds, SCRIPT_DIR / f"sample_grid_{grid_name}.png")

    if not all_stats:
        print("No datasets found!")
        return

    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Shelf Dataset Distribution Comparison", fontsize=16, fontweight="bold")

    # 1. Annotations per image
    ax = axes[0, 0]
    for s in all_stats:
        ax.hist(s["ann_counts"], bins=40, alpha=0.5, label=f"{s['name']} (n={s['n_images']})",
                color=s["color"], density=True)
    ax.set_xlabel("Annotations per image")
    ax.set_ylabel("Density")
    ax.set_title("Annotation Count per Image")
    ax.legend(fontsize=7)

    # 2. Bbox area (log)
    ax = axes[0, 1]
    for s in all_stats:
        if s["bbox_areas"]:
            ax.hist(np.log10(np.array(s["bbox_areas"]) + 1), bins=50, alpha=0.5,
                    label=s["name"], color=s["color"], density=True)
    ax.set_xlabel("log10(bbox area px²)")
    ax.set_ylabel("Density")
    ax.set_title("Bounding Box Size")
    ax.legend(fontsize=7)

    # 3. Relative bbox area
    ax = axes[0, 2]
    for s in all_stats:
        if s["bbox_rel_areas"]:
            vals = np.array(s["bbox_rel_areas"]) * 100  # percent
            ax.hist(vals, bins=50, alpha=0.5, range=(0, 5),
                    label=s["name"], color=s["color"], density=True)
    ax.set_xlabel("Bbox area as % of image")
    ax.set_ylabel("Density")
    ax.set_title("Relative Box Size")
    ax.legend(fontsize=7)

    # 4. Bbox aspect ratio
    ax = axes[1, 0]
    for s in all_stats:
        if s["bbox_aspects"]:
            ax.hist(np.clip(s["bbox_aspects"], 0.1, 5), bins=50, alpha=0.5,
                    label=s["name"], color=s["color"], density=True)
    ax.set_xlabel("width / height")
    ax.set_ylabel("Density")
    ax.set_title("Bbox Aspect Ratio")
    ax.legend(fontsize=7)

    # 5. Image resolution
    ax = axes[1, 1]
    for s in all_stats:
        megapix = [w * h / 1e6 for w, h in zip(s["img_widths"], s["img_heights"])]
        ax.hist(megapix, bins=30, alpha=0.5, label=s["name"], color=s["color"], density=True)
    ax.set_xlabel("Megapixels")
    ax.set_ylabel("Density")
    ax.set_title("Image Resolution")
    ax.legend(fontsize=7)

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis("off")
    headers = ["Dataset", "Imgs", "Anns", "Cats", "Avg/img", "Med bbox%"]
    rows = []
    for s in all_stats:
        med_rel = np.median(s["bbox_rel_areas"]) * 100 if s["bbox_rel_areas"] else 0
        rows.append([
            s["name"][:20],
            f"{s['n_images']:,}",
            f"{s['n_annotations']:,}",
            str(s["n_categories"]),
            f"{np.mean(s['ann_counts']):.0f}",
            f"{med_rel:.2f}%",
        ])
    table = ax.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)

    plt.tight_layout()
    out = SCRIPT_DIR / "distribution_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nComparison plot: {out}")


if __name__ == "__main__":
    main()

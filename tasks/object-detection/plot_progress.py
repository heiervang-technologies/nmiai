#!/usr/bin/env python3
"""Karpathy-style progress plot for data mixture experiments."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
TSV_PATH = SCRIPT_DIR / "data_experiment_results.tsv"
PROGRESS_PNG = SCRIPT_DIR / "data_experiment_progress.png"
EFFICIENCY_PNG = SCRIPT_DIR / "data_experiment_efficiency.png"

REFERENCE_LINES = [
    (0.92, "SOTA", "#888888", 1.0),
    (0.9079, "Our best server", "#e67e22", 0.7),
]


def load_data():
    if not TSV_PATH.exists():
        return []
    rows = []
    with open(TSV_PATH, "r") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            try:
                rows.append({
                    "experiment_id": r["experiment_id"].strip(),
                    "combined_map": float(r["combined_map"]),
                    "det_map50": float(r["det_map50"]),
                    "cls_map50": float(r["cls_map50"]),
                    "unique_images": int(r["unique_images"]),
                    "description": r.get("description", "").strip(),
                })
            except (ValueError, KeyError):
                continue
    return rows


def style_ax(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(True, which="major", color="#e0e0e0", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, width=0.8)


def add_reference_lines(ax):
    for yval, label, color, alpha in REFERENCE_LINES:
        ax.axhline(y=yval, color=color, linestyle="--", linewidth=1.0, alpha=alpha)
        ax.text(ax.get_xlim()[1], yval, f"  {label} ({yval})",
                va="center", ha="left", fontsize=7.5, color=color, alpha=alpha,
                clip_on=False)


def plot_progress(rows):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    style_ax(ax)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("mAP@0.5", fontsize=11)
    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_title("Data Mixture Optimization \u2014 mAP@0.5", fontsize=13, fontweight="bold", pad=12)

    if not rows:
        ax.set_xlim(0, 10)
        add_reference_lines(ax)
        ax.text(5, 0.5, "No experiments yet", ha="center", va="center",
                fontsize=14, color="#aaaaaa")
        fig.tight_layout()
        fig.savefig(PROGRESS_PNG, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    xs = list(range(1, len(rows) + 1))
    combined = [r["combined_map"] for r in rows]
    det = [r["det_map50"] for r in rows]
    cls = [r["cls_map50"] for r in rows]

    ax.plot(xs, combined, color="#2563eb", linewidth=2.2, marker="o", markersize=6,
            label="Combined mAP", zorder=5)
    ax.plot(xs, det, color="#16a34a", linewidth=1.4, linestyle="--", marker="^",
            markersize=5, label="Detection mAP@0.5", zorder=4)
    ax.plot(xs, cls, color="#dc2626", linewidth=1.4, linestyle="--", marker="s",
            markersize=5, label="Classification mAP@0.5", zorder=4)

    for i, r in enumerate(rows):
        if r["description"]:
            ax.annotate(r["description"], (xs[i], combined[i]),
                        textcoords="offset points", xytext=(4, 8),
                        fontsize=6.5, rotation=45, color="#555555",
                        ha="left", va="bottom")

    ax.set_xlim(0.5, len(rows) + 0.5)
    ymin = max(0, min(min(combined), min(det), min(cls)) - 0.05)
    ax.set_ylim(ymin, min(1.0, max(max(combined), max(det), max(cls)) + 0.06))

    add_reference_lines(ax)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(PROGRESS_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_efficiency(rows):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor="white")
    style_ax(ax)
    ax.set_ylabel("Combined mAP@0.5", fontsize=11)
    ax.set_xlabel("Unique Images", fontsize=11)
    ax.set_title("Sample Efficiency \u2014 mAP vs Dataset Size", fontsize=13, fontweight="bold", pad=12)

    if not rows:
        ax.text(0.5, 0.5, "No experiments yet", ha="center", va="center",
                fontsize=14, color="#aaaaaa", transform=ax.transAxes)
        fig.tight_layout()
        fig.savefig(EFFICIENCY_PNG, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    imgs = [r["unique_images"] for r in rows]
    combined = [r["combined_map"] for r in rows]

    ax.scatter(imgs, combined, c="#2563eb", s=60, zorder=5, edgecolors="white", linewidth=0.5)

    for r in rows:
        if r["description"]:
            ax.annotate(r["description"], (r["unique_images"], r["combined_map"]),
                        textcoords="offset points", xytext=(6, 6),
                        fontsize=6.5, rotation=45, color="#555555",
                        ha="left", va="bottom")

    for yval, label, color, alpha in REFERENCE_LINES:
        ax.axhline(y=yval, color=color, linestyle="--", linewidth=1.0, alpha=alpha,
                   label=f"{label} ({yval})")

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    fig.tight_layout()
    fig.savefig(EFFICIENCY_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    rows = load_data()
    plot_progress(rows)
    plot_efficiency(rows)
    print(f"Saved: {PROGRESS_PNG}")
    print(f"Saved: {EFFICIENCY_PNG}")


if __name__ == "__main__":
    main()

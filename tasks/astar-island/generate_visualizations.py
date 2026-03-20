#!/usr/bin/env python3
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np

BASE = Path("tasks/astar-island")
GT_DIR = BASE / "ground_truth"
OUT_DIR = BASE / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_NAMES = {
    0: "Empty",
    1: "Settlement",
    2: "Port",
    3: "Ruin",
    4: "Forest",
    5: "Mountain",
}
INIT_NAMES = {
    0: "Empty",
    1: "Settlement",
    2: "Port",
    4: "Forest",
    5: "Mountain",
    10: "Ocean",
    11: "Plains",
}
PRED_COLORS = ["#c8b88a", "#d4760a", "#0e7490", "#7f1d1d", "#2d5a27", "#6b7280"]
INIT_CODE_ORDER = [10, 11, 1, 2, 4, 5]
INIT_COLORS = {
    10: "#5aa6d1",
    11: "#d9c98b",
    1: "#d4760a",
    2: "#0e7490",
    4: "#2d5a27",
    5: "#6b7280",
}
INIT_CODE_TO_INDEX = {code: idx for idx, code in enumerate(INIT_CODE_ORDER)}
INIT_CMAP = ListedColormap([INIT_COLORS[c] for c in INIT_CODE_ORDER])
PRED_CMAP = ListedColormap(PRED_COLORS)
TRANSITION_CMAP = ListedColormap(["#efefef"] + PRED_COLORS)
ENTROPY_CMAP = "viridis"
PATTERN_CMAP = "magma"
DPI = 100


def load_seed(seed_idx: int):
    path = GT_DIR / f"round1_seed{seed_idx}.json"
    data = json.loads(path.read_text())
    init = np.array(data["initial_grid"], dtype=int)
    gt = np.array(data["ground_truth"], dtype=float)
    return init, gt


def shannon_entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1e-12, 1.0)
    return -(clipped * np.log2(clipped)).sum(axis=2)


def map_initial_grid(init: np.ndarray) -> np.ndarray:
    mapped = np.zeros_like(init)
    for code, idx in INIT_CODE_TO_INDEX.items():
        mapped[init == code] = idx
    return mapped


def coast_mask(init: np.ndarray) -> np.ndarray:
    ocean = init == 10
    h, w = init.shape
    out = np.zeros_like(ocean, dtype=bool)
    for r in range(h):
        for c in range(w):
            if ocean[r, c]:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and ocean[rr, cc]:
                    out[r, c] = True
                    break
    return out


def neighbor_count(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=int)
    for r in range(h):
        for c in range(w):
            total = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        total += int(mask[rr, cc])
            out[r, c] = total
    return out


def near_target(target: np.ndarray) -> np.ndarray:
    h, w = target.shape
    out = np.zeros_like(target, dtype=bool)
    for r in range(h):
        for c in range(w):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and target[rr, cc]:
                        out[r, c] = True
                        break
                if out[r, c]:
                    break
    return out


def style_grid(ax, title: str):
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def add_pred_legend(ax):
    handles = [Patch(facecolor=PRED_COLORS[i], edgecolor="none", label=PRED_NAMES[i]) for i in range(6)]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)


def save_overview(seed_idx: int, init: np.ndarray, gt: np.ndarray):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=DPI)

    axes[0].imshow(
        map_initial_grid(init),
        cmap=INIT_CMAP,
        vmin=0,
        vmax=len(INIT_CODE_ORDER) - 1,
        interpolation="nearest",
    )
    style_grid(axes[0], f"Seed {seed_idx}: Initial State")
    init_handles = [Patch(facecolor=INIT_COLORS[c], edgecolor="none", label=INIT_NAMES[c]) for c in INIT_CODE_ORDER]
    axes[0].legend(handles=init_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=8)

    argmax = gt.argmax(axis=2)
    axes[1].imshow(argmax, cmap=PRED_CMAP, vmin=0, vmax=5, interpolation="nearest")
    style_grid(axes[1], "Ground Truth Argmax")
    add_pred_legend(axes[1])

    entropy = shannon_entropy(gt)
    im = axes[2].imshow(entropy, cmap=ENTROPY_CMAP, interpolation="nearest")
    style_grid(axes[2], "Ground Truth Entropy")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(OUT_DIR / f"seed{seed_idx}_overview.png", bbox_inches="tight")
    plt.close(fig)


def save_transition_maps(seed_idx: int, init: np.ndarray, gt: np.ndarray):
    argmax = gt.argmax(axis=2)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=DPI)
    axes = axes.ravel()

    for ax, code in zip(axes, INIT_CODE_ORDER):
        panel = np.full_like(argmax, -1)
        panel[init == code] = argmax[init == code] + 1
        ax.imshow(panel, cmap=TRANSITION_CMAP, vmin=0, vmax=6, interpolation="nearest")
        style_grid(ax, f"Initial {INIT_NAMES[code]}")
        counts = np.bincount(argmax[init == code], minlength=6)
        summary = ", ".join(f"{PRED_NAMES[i][0]}:{counts[i]}" for i in range(6) if counts[i])
        ax.text(0.02, -0.09, summary, transform=ax.transAxes, fontsize=8)

    handles = [Patch(facecolor="#efefef", edgecolor="none", label="Other cells")]
    handles += [Patch(facecolor=PRED_COLORS[i], edgecolor="none", label=PRED_NAMES[i]) for i in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(f"Seed {seed_idx}: Transition Maps", fontsize=12)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(OUT_DIR / f"seed{seed_idx}_transitions.png", bbox_inches="tight")
    plt.close(fig)


def save_pattern_maps(seed_idx: int, init: np.ndarray, gt: np.ndarray):
    settlement_prob = gt[:, :, 1]
    port_prob = gt[:, :, 2]
    ruin_prob = gt[:, :, 3]
    coast = coast_mask(init)
    near_init_settlement = near_target(init == 1)
    settlement_neighbors = neighbor_count(init == 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=DPI)

    im0 = axes[0, 0].imshow(settlement_prob, cmap=PATTERN_CMAP, interpolation="nearest")
    axes[0, 0].contour((init == 1).astype(float), levels=[0.5], colors="white", linewidths=0.6)
    style_grid(axes[0, 0], "Settlement Probability + Initial Settlements")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    coastal_port = np.ma.masked_where(~coast, port_prob)
    axes[0, 1].imshow(
        map_initial_grid(init),
        cmap=INIT_CMAP,
        vmin=0,
        vmax=len(INIT_CODE_ORDER) - 1,
        interpolation="nearest",
        alpha=0.4,
    )
    im1 = axes[0, 1].imshow(coastal_port, cmap=PATTERN_CMAP, vmin=0, vmax=max(0.12, float(port_prob.max())), interpolation="nearest")
    style_grid(axes[0, 1], "Coastal Port Probability")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[1, 0].imshow(ruin_prob, cmap=PATTERN_CMAP, interpolation="nearest")
    axes[1, 0].contour(near_init_settlement.astype(float), levels=[0.5], colors="cyan", linewidths=0.6)
    style_grid(axes[1, 0], "Ruin Probability + Near Initial Settlements")
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(settlement_neighbors, cmap="cividis", interpolation="nearest")
    axes[1, 1].contour(settlement_prob, levels=[0.2, 0.4], colors=["white", "red"], linewidths=0.6)
    style_grid(axes[1, 1], "Initial Settlement Neighbors + Growth Contours")
    fig.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(f"Seed {seed_idx}: Spatial Patterns", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / f"seed{seed_idx}_patterns.png", bbox_inches="tight")
    plt.close(fig)


def summarize_across_seeds():
    seeds = []
    for seed_idx in range(5):
        init, gt = load_seed(seed_idx)
        seeds.append((seed_idx, init, gt))

    agg = defaultdict(lambda: np.zeros(6, dtype=float))
    counts = defaultdict(int)
    context_rows = []
    metric_rows = []
    mean_settlement = np.zeros((40, 40), dtype=float)
    mean_port = np.zeros((40, 40), dtype=float)
    mean_ruin = np.zeros((40, 40), dtype=float)
    mean_entropy = np.zeros((40, 40), dtype=float)

    for seed_idx, init, gt in seeds:
        argmax = gt.argmax(axis=2)
        entropy = shannon_entropy(gt)
        mean_settlement += gt[:, :, 1]
        mean_port += gt[:, :, 2]
        mean_ruin += gt[:, :, 3]
        mean_entropy += entropy

        coast = coast_mask(init)
        near_init_settlement = near_target(init == 1)
        settlement_neighbors = neighbor_count(init == 1)

        for code in np.unique(init):
            mask = init == code
            agg[int(code)] += gt[mask].sum(axis=0)
            counts[int(code)] += int(mask.sum())

        contexts = {
            "plains": init == 11,
            "plains_near_initial_settlement": (init == 11) & near_init_settlement,
            "plains_coastal": (init == 11) & coast,
            "forest": init == 4,
            "forest_near_initial_settlement": (init == 4) & near_init_settlement,
            "initial_settlement": init == 1,
            "initial_settlement_coastal": (init == 1) & coast,
            "cells_with_2plus_initial_settlement_neighbors": settlement_neighbors >= 2,
        }
        for label, mask in contexts.items():
            if not mask.any():
                continue
            context_rows.append((label, int(mask.sum()), gt[mask].mean(axis=0)))

        metric_rows.append(
            {
                "seed": seed_idx,
                "mean_entropy": float(entropy.mean()),
                "max_entropy": float(entropy.max()),
                "nonzero_entropy_cells": int((entropy > 0.01).sum()),
                "argmax_settlements": int((argmax == 1).sum()),
                "argmax_ports": int((argmax == 2).sum()),
                "argmax_ruins": int((argmax == 3).sum()),
            }
        )

    mean_settlement /= len(seeds)
    mean_port /= len(seeds)
    mean_ruin /= len(seeds)
    mean_entropy /= len(seeds)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9), dpi=DPI)
    panels = [
        (mean_settlement, "Mean Settlement Probability", PATTERN_CMAP),
        (mean_port, "Mean Port Probability", PATTERN_CMAP),
        (mean_ruin, "Mean Ruin Probability", PATTERN_CMAP),
        (mean_entropy, "Mean Entropy", ENTROPY_CMAP),
    ]
    for ax, (data, title, cmap) in zip(axes.ravel(), panels):
        im = ax.imshow(data, cmap=cmap, interpolation="nearest")
        style_grid(ax, title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Round 1: Cross-Seed Mean Probability Fields", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_DIR / "round1_cross_seed_means.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=DPI)
    x = np.arange(len(metric_rows))
    width = 0.24
    ax.bar(x - width, [m["argmax_settlements"] for m in metric_rows], width, label="Argmax settlements", color=PRED_COLORS[1])
    ax.bar(x, [m["argmax_ports"] for m in metric_rows], width, label="Argmax ports", color=PRED_COLORS[2])
    ax.bar(x + width, [m["argmax_ruins"] for m in metric_rows], width, label="Argmax ruins", color=PRED_COLORS[3])
    ax2 = ax.twinx()
    ax2.plot(x, [m["mean_entropy"] for m in metric_rows], color="black", marker="o", linewidth=1.5, label="Mean entropy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Seed {m['seed']}" for m in metric_rows])
    ax.set_ylabel("Cells")
    ax2.set_ylabel("Bits")
    ax.set_title("Round 1: Outcome Scale by Seed")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "round1_seed_metrics.png", bbox_inches="tight")
    plt.close(fig)

    by_label = defaultdict(list)
    for label, count, probs in context_rows:
        by_label[label].append((count, probs))

    with (OUT_DIR / "round1_summary_stats.txt").open("w") as f:
        f.write("Transition probabilities by initial type\n")
        for code in INIT_CODE_ORDER:
            probs = agg[code] / counts[code]
            summary = ", ".join(f"{PRED_NAMES[i]}={probs[i]:.4f}" for i in range(6))
            f.write(f"{INIT_NAMES[code]} ({counts[code]} cells): {summary}\n")

        f.write("\nContextual probabilities\n")
        for label in sorted(by_label):
            total = sum(count for count, _ in by_label[label])
            probs = sum(count * p for count, p in by_label[label]) / total
            summary = ", ".join(f"{PRED_NAMES[i]}={probs[i]:.4f}" for i in range(6))
            f.write(f"{label} ({total} cells): {summary}\n")

        f.write("\nPer-seed metrics\n")
        for metric in metric_rows:
            f.write(
                f"Seed {metric['seed']}: mean_entropy={metric['mean_entropy']:.4f}, "
                f"max_entropy={metric['max_entropy']:.4f}, nonzero_entropy_cells={metric['nonzero_entropy_cells']}, "
                f"argmax_settlements={metric['argmax_settlements']}, argmax_ports={metric['argmax_ports']}, "
                f"argmax_ruins={metric['argmax_ruins']}\n"
            )


def main():
    for seed_idx in range(5):
        init, gt = load_seed(seed_idx)
        save_overview(seed_idx, init, gt)
        save_transition_maps(seed_idx, init, gt)
        save_pattern_maps(seed_idx, init, gt)
    summarize_across_seeds()


if __name__ == "__main__":
    main()

"""
Autoresearch progress plot — Karpathy style.

Renders TWO plots per task:
  1. Competition/live metric progress (the real score)
  2. Validation metric progress (the local eval signal)

Usage:
    python tools/plot_autoresearch.py              # all tasks
    python tools/plot_autoresearch.py astar-island  # one task
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

TASKS = {
    "astar-island": {
        "tsv": REPO / "tasks/astar-island/autoresearch_results.tsv",
        "out_dir": REPO / "tasks/astar-island",
        "label_col": "pipeline_change",
        "id_col": "round",
        "plots": [
            {
                "name": "competition",
                "metric_col": "raw_score",
                "direction": "max",
                "title": "Astar Island — Competition Score",
                "ylabel": "Raw Score (higher = better)",
                "filename": "autoresearch_progress_competition.png",
            },
            {
                "name": "validation",
                "metric_col": "cv_wkl",
                "direction": "min",
                "title": "Astar Island — Validation (CV wKL)",
                "ylabel": "CV Weighted KL (lower = better)",
                "filename": "autoresearch_progress_validation.png",
            },
        ],
    },
    "object-detection": {
        "tsv": REPO / "tasks/object-detection/autoresearch_results.tsv",
        "out_dir": REPO / "tasks/object-detection",
        "label_col": "notes",
        "id_col": "model",
        "plots": [
            {
                "name": "competition",
                "metric_col": "value",
                "direction": "max",
                "title": "Object Detection — Server/Combined Score",
                "ylabel": "Score (higher = better)",
                "filename": "autoresearch_progress_competition.png",
                "filter": lambda df: df[df["metric"].isin(["combined", "server_mAP"])],
            },
            {
                "name": "validation",
                "metric_col": "value",
                "direction": "max",
                "title": "Object Detection — Validation (mAP@0.5)",
                "ylabel": "mAP@0.5 (higher = better)",
                "filename": "autoresearch_progress_validation.png",
                "filter": lambda df: df[df["metric"] == "mAP50"],
            },
        ],
    },
    "accounting": {
        "tsv": REPO / "tasks/accounting/autoresearch_results.tsv",
        "out_dir": REPO / "tasks/accounting",
        "label_col": "description",
        "id_col": "batch_id",
        "plots": [
            {
                "name": "competition",
                "metric_col": "dashboard_score",
                "direction": "max",
                "title": "Accounting — Dashboard Score",
                "ylabel": "Dashboard Score (higher = better)",
                "filename": "autoresearch_progress_competition.png",
            },
            {
                "name": "validation",
                "metric_col": "proxy_clean_rate",
                "direction": "max",
                "title": "Accounting — Validation (Proxy Clean Rate)",
                "ylabel": "Clean Rate (higher = better)",
                "filename": "autoresearch_progress_validation.png",
            },
        ],
    },
}


def load_and_filter(tsv_path, metric_col, plot_filter=None):
    """Load TSV, apply filter, parse metric, return df."""
    if not tsv_path.exists():
        return None

    df = pd.read_csv(tsv_path, sep="\t")

    if plot_filter is not None:
        df = plot_filter(df)

    if metric_col not in df.columns:
        print(f"  Warning: '{metric_col}' not in columns {list(df.columns)}")
        return None

    df[metric_col] = pd.to_numeric(df[metric_col], errors="coerce")
    df = df.dropna(subset=[metric_col]).reset_index(drop=True)
    if len(df) == 0:
        return None

    df["experiment_num"] = range(1, len(df) + 1)
    return df


def render_plot(df, metric_col, direction, title, ylabel, label_col, out_path):
    """Render a single Karpathy-style progress plot."""
    if direction == "min":
        df["running_best"] = df[metric_col].cummin()
    else:
        df["running_best"] = df[metric_col].cummax()

    # Mark new-best experiments as "kept"
    kept_mask = df["running_best"] != df["running_best"].shift(1)
    kept_mask.iloc[0] = True

    fig, ax = plt.subplots(figsize=(16, 8))

    # Discarded: faint gray
    disc = df[~kept_mask]
    ax.scatter(disc["experiment_num"], disc[metric_col],
               color="#cccccc", s=40, zorder=2, label="Discarded / no improvement")

    # Kept: green
    kept = df[kept_mask]
    ax.scatter(kept["experiment_num"], kept[metric_col],
               color="#22cc55", s=100, edgecolors="black", linewidths=1,
               zorder=4, label="New best")

    # Running best frontier
    ax.step(df["experiment_num"], df["running_best"],
            where="post", color="#22cc55", linewidth=2, alpha=0.7,
            zorder=3, label="Running best")

    # Annotate kept experiments
    for _, row in kept.iterrows():
        label = str(row.get(label_col, ""))[:50]
        ax.annotate(label,
                    (row["experiment_num"], row[metric_col]),
                    textcoords="offset points", xytext=(8, 8),
                    fontsize=7, alpha=0.8, rotation=15,
                    arrowprops=dict(arrowstyle="-", alpha=0.3))

    ax.set_xlabel("Experiment #", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    vals = df[metric_col]
    margin = (vals.max() - vals.min()) * 0.1 if vals.max() != vals.min() else 1
    if direction == "min":
        ax.set_ylim(vals.min() - margin, vals.max() + margin)
        ax.invert_yaxis()
    else:
        ax.set_ylim(vals.min() - margin, vals.max() + margin)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    best_val = kept[metric_col].min() if direction == "min" else kept[metric_col].max()
    baseline = df[metric_col].iloc[0]
    print(f"  [{title}]")
    print(f"    {len(df)} experiments, {len(kept)} kept")
    print(f"    Baseline: {baseline:.4f} -> Best: {best_val:.4f}")
    print(f"    Saved {out_path}")


def plot_task(name, cfg):
    """Render both competition and validation plots for a task."""
    for plot_cfg in cfg["plots"]:
        df = load_and_filter(
            cfg["tsv"],
            plot_cfg["metric_col"],
            plot_cfg.get("filter"),
        )
        if df is None:
            print(f"  Skipping {name}/{plot_cfg['name']}: no valid data")
            continue

        out_path = cfg["out_dir"] / plot_cfg["filename"]
        render_plot(
            df,
            plot_cfg["metric_col"],
            plot_cfg["direction"],
            plot_cfg["title"],
            plot_cfg["ylabel"],
            cfg.get("label_col", "description"),
            out_path,
        )


def main():
    tasks_to_plot = sys.argv[1:] if len(sys.argv) > 1 else list(TASKS.keys())

    for name in tasks_to_plot:
        if name not in TASKS:
            print(f"Unknown task: {name}. Available: {list(TASKS.keys())}")
            continue
        print(f"\n=== {name} ===")
        plot_task(name, TASKS[name])


if __name__ == "__main__":
    main()

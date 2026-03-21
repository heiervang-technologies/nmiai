#!/usr/bin/env python3
"""Generate progress-style plots and frontier summaries for Astar Island."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / 'tasks' / 'astar-island'
DEFAULT_RESULTS = TASK_DIR / 'autoresearch_results.tsv'
DEFAULT_OUT = TASK_DIR / 'analysis' / 'progress.png'
DEFAULT_FRONTIER = TASK_DIR / 'analysis' / 'cv_vs_live_frontier.json'
REGIME_COLORS = {
    'harsh': 'tab:red',
    'moderate': 'tab:blue',
    'prosperous': 'tab:green',
    'pending': 'tab:gray',
    '-': 'tab:gray',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--frontier-json', type=Path, default=DEFAULT_FRONTIER)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text or text in {'-', 'pending'}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def rank_percentile(rank: str | None, total: str | None) -> float | None:
    rank_value = to_float(rank)
    total_value = to_float(total)
    if rank_value is None or total_value in (None, 0):
        return None
    return 1.0 - ((rank_value - 1.0) / total_value)


def round_order(value: str) -> int:
    if value.startswith('R'):
        try:
            return int(value[1:])
        except ValueError:
            return 0
    return 0


def build_frontier(rows: list[dict[str, str]]) -> dict[str, object]:
    points = []
    for row in rows:
        cv_wkl = to_float(row.get('cv_wkl'))
        weighted_score = to_float(row.get('weighted_score'))
        if cv_wkl is None or weighted_score is None:
            continue
        points.append(
            {
                'label': row.get('round', 'unknown'),
                'cv_wkl': cv_wkl,
                'weighted_score': weighted_score,
                'regime': row.get('regime', ''),
                'pipeline_change': row.get('pipeline_change', ''),
            }
        )
    frontier = []
    for point in points:
        dominated = False
        for other in points:
            if other is point:
                continue
            if (
                other['cv_wkl'] <= point['cv_wkl']
                and other['weighted_score'] >= point['weighted_score']
                and (other['cv_wkl'] < point['cv_wkl'] or other['weighted_score'] > point['weighted_score'])
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(point)
    frontier.sort(key=lambda item: (item['cv_wkl'], -item['weighted_score']))
    return {'points': points, 'frontier': frontier}


def metric_series(rows: list[dict[str, str]], key: str) -> list[tuple[int, str, float, str]]:
    series = []
    for row in rows:
        value = to_float(row.get(key))
        if value is None:
            continue
        round_id = row.get('round', 'R0')
        series.append((round_order(round_id), round_id, value, row.get('regime', '-')))
    series.sort(key=lambda item: item[0])
    return series


def percentile_series(rows: list[dict[str, str]]) -> list[tuple[int, str, float, str]]:
    series = []
    for row in rows:
        value = rank_percentile(row.get('rank'), row.get('total_teams'))
        if value is None:
            continue
        round_id = row.get('round', 'R0')
        series.append((round_order(round_id), round_id, value, row.get('regime', '-')))
    series.sort(key=lambda item: item[0])
    return series


def plot_metric(ax, series: list[tuple[int, str, float, str]], title: str, ylabel: str, best: str) -> None:
    if not series:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    xs = [item[0] for item in series]
    labels = [item[1] for item in series]
    ys = [item[2] for item in series]
    colors = [REGIME_COLORS.get(item[3], 'tab:gray') for item in series]
    ax.plot(xs, ys, color='black', linewidth=1.0, alpha=0.5)
    ax.scatter(xs, ys, c=colors, s=45)
    best_trace = []
    current = None
    for value in ys:
        if current is None:
            current = value
        elif best == 'max':
            current = max(current, value)
        else:
            current = min(current, value)
        best_trace.append(current)
    ax.plot(xs, best_trace, linestyle='--', linewidth=1.4, color='tab:purple', label='best so far')
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=45)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.frontier_json:
        args.frontier_json.parent.mkdir(parents=True, exist_ok=True)

    weighted = metric_series(rows, 'weighted_score')
    raw = metric_series(rows, 'raw_score')
    cv_wkl = metric_series(rows, 'cv_wkl')
    percentile = percentile_series(rows)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    plot_metric(axes[0, 0], weighted, 'Weighted Score Over Time', 'Weighted Score', 'max')
    plot_metric(axes[0, 1], cv_wkl, 'Cross-Validation wKL', 'wKL', 'min')
    plot_metric(axes[1, 0], raw, 'Raw Score Over Time', 'Raw Score', 'max')
    plot_metric(axes[1, 1], percentile, 'Placement Percentile', 'Percentile', 'max')

    legend_handles = []
    legend_labels = []
    for regime, color in REGIME_COLORS.items():
        if regime == '-':
            continue
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8))
        legend_labels.append(regime)
    fig.legend(legend_handles, legend_labels, loc='upper center', ncol=4, fontsize=9)
    fig.suptitle('Astar Island Progress Metrics', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.output, dpi=160)

    if args.frontier_json:
        args.frontier_json.write_text(json.dumps(build_frontier(rows), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f'Wrote {args.output}')
    if args.frontier_json:
        print(f'Wrote {args.frontier_json}')


if __name__ == '__main__':
    main()

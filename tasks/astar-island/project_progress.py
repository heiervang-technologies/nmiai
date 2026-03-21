#!/usr/bin/env python3
"""Generate explicit projection artifacts for Astar Island."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / 'tasks' / 'astar-island'
DEFAULT_RESULTS = TASK_DIR / 'autoresearch_results.tsv'
DEFAULT_OUT = TASK_DIR / 'analysis' / 'projection.png'
DEFAULT_JSON = TASK_DIR / 'analysis' / 'projection_report.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_JSON)
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


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov_xy / var_x if var_x else 0.0
    intercept = mean_y - slope * mean_x
    rmse = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / n)
    return slope, intercept, rmse


def predict(intercept: float, slope: float, x: float) -> float:
    return intercept + slope * x


def main() -> None:
    args = parse_args()
    rows = read_rows(args.results)

    completed = []
    pending = []
    for row in rows:
        cv_wkl = to_float(row.get('cv_wkl'))
        weighted = to_float(row.get('weighted_score'))
        round_id = row.get('round', 'unknown')
        if cv_wkl is None:
            continue
        if weighted is None:
            pending.append({'round': round_id, 'cv_wkl': cv_wkl, 'regime': row.get('regime', ''), 'pipeline_change': row.get('pipeline_change', '')})
        else:
            completed.append({'round': round_id, 'cv_wkl': cv_wkl, 'weighted_score': weighted, 'regime': row.get('regime', ''), 'pipeline_change': row.get('pipeline_change', '')})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        'completed_points': completed,
        'pending_points': pending,
        'projection_status': 'unavailable',
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    if len(completed) >= 2:
        xs = [item['cv_wkl'] for item in completed]
        ys = [item['weighted_score'] for item in completed]
        slope, intercept, rmse = fit_line(xs, ys)
        report['projection_status'] = 'ok'
        report['calibration'] = {'slope': slope, 'intercept': intercept, 'rmse_weighted_score': rmse, 'points': len(completed)}
        ax.scatter(xs, ys, color='tab:blue', label='completed rounds')
        for item in completed:
            ax.annotate(item['round'], (item['cv_wkl'], item['weighted_score']), textcoords='offset points', xytext=(4, 4), fontsize=8)
        x_min = min(xs + [item['cv_wkl'] for item in pending] if pending else xs)
        x_max = max(xs + [item['cv_wkl'] for item in pending] if pending else xs)
        span = max(x_max - x_min, 0.01)
        line_x = [x_min - 0.05 * span, x_max + 0.05 * span]
        line_y = [predict(intercept, slope, value) for value in line_x]
        ax.plot(line_x, line_y, color='black', linestyle='--', linewidth=1.4, label='cv -> weighted fit')

        projections = []
        for item in pending:
            projected = predict(intercept, slope, item['cv_wkl'])
            projections.append({**item, 'projected_weighted_score': projected})
            ax.scatter([item['cv_wkl']], [projected], color='tab:orange', marker='*', s=180, label='pending projection' if len(projections) == 1 else None)
            ax.annotate(f"{item['round']}~{projected:.1f}", (item['cv_wkl'], projected), textcoords='offset points', xytext=(6, -10), fontsize=8)
        report['projections'] = projections
        title = f"Astar Projection: CV wKL -> Weighted Score (RMSE {rmse:.1f})"
    else:
        title = 'Astar Projection: insufficient completed rounds for calibration'
        ax.text(0.5, 0.5, 'Need at least 2 completed rounds with both cv_wkl and weighted_score.', ha='center', va='center', transform=ax.transAxes)

    ax.set_title(title)
    ax.set_xlabel('Cross-Validation wKL')
    ax.set_ylabel('Weighted Score')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)

    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')
    print(f'Wrote {args.output_json}')


if __name__ == '__main__':
    main()

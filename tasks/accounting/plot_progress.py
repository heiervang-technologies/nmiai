#!/usr/bin/env python3
"""Generate progress.png-style plots for accounting metrics."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / 'tasks' / 'accounting'
DEFAULT_RESULTS = TASK_DIR / 'autoresearch_results.tsv'
DEFAULT_DASHBOARD = TASK_DIR / 'dashboard_scores.tsv'
DEFAULT_OUT = TASK_DIR / 'analysis' / 'progress.png'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--dashboard', type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as handle:
        return [row for row in csv.DictReader(handle, delimiter='\t')]


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def grouped_family_series(rows: list[dict], value_key: str) -> dict[str, list[tuple[int, float]]]:
    series = defaultdict(list)
    for idx, row in enumerate(rows):
        value = to_float(row.get(value_key))
        if value is None:
            continue
        family = row.get('family', 'unknown')
        series[family].append((idx, value))
    return dict(series)


def cumulative_family_coverage(rows: list[dict]) -> list[int]:
    seen = set()
    values = []
    for row in rows:
        family = row.get('family')
        if family:
            seen.add(family)
        values.append(len(seen))
    return values


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result_rows = read_tsv(args.results)
    dashboard_rows = read_tsv(args.dashboard)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax_clean, ax_errors, ax_coverage, ax_dashboard = axes.flatten()

    clean_series = grouped_family_series(result_rows, 'proxy_clean_rate')
    for family, points in clean_series.items():
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax_clean.plot(xs, ys, marker='o', label=family)
    ax_clean.set_title('Proxy Clean Rate by Family')
    ax_clean.set_xlabel('Analyzer Row')
    ax_clean.set_ylabel('Proxy Clean Rate')
    ax_clean.set_ylim(0, 1.05)
    if clean_series:
        ax_clean.legend(fontsize=8)
    ax_clean.grid(alpha=0.3)

    error_series = grouped_family_series(result_rows, 'api_errors')
    for family, points in error_series.items():
        xs = [x for x, _ in points]
        ys = [y for _, y in points]
        ax_errors.plot(xs, ys, marker='o', label=family)
    ax_errors.set_title('API Errors by Family')
    ax_errors.set_xlabel('Analyzer Row')
    ax_errors.set_ylabel('4xx Errors')
    if error_series:
        ax_errors.legend(fontsize=8)
    ax_errors.grid(alpha=0.3)

    coverage = cumulative_family_coverage(result_rows)
    ax_coverage.plot(list(range(len(coverage))), coverage, marker='o', color='black')
    ax_coverage.set_title('Cumulative Family Coverage')
    ax_coverage.set_xlabel('Analyzer Row')
    ax_coverage.set_ylabel('Unique Families Seen')
    ax_coverage.grid(alpha=0.3)

    dash_scores = [to_float(row.get('dashboard_score')) for row in dashboard_rows]
    dash_scores = [score for score in dash_scores if score is not None]
    if dash_scores:
        ax_dashboard.plot(list(range(len(dash_scores))), dash_scores, marker='o', color='tab:green')
    ax_dashboard.set_title('Dashboard Scores Over Time')
    ax_dashboard.set_xlabel('Manual Score Check')
    ax_dashboard.set_ylabel('Dashboard Score')
    ax_dashboard.grid(alpha=0.3)

    fig.suptitle('Accounting Progress Metrics', fontsize=16)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()

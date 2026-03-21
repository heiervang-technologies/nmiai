#!/usr/bin/env python3
"""Generate progress-style plots and frontier summaries for object detection."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / 'tasks' / 'object-detection'
DEFAULT_RESULTS = TASK_DIR / 'autoresearch_results.tsv'
DEFAULT_OUT = TASK_DIR / 'analysis' / 'progress.png'
DEFAULT_FRONTIER = TASK_DIR / 'analysis' / 'detection_frontier.json'

PRIMARY_METRICS = {'combined', 'server_mAP'}
DETECTION_METRICS = {'mAP50', 'det_mAP'}
CLASSIFICATION_METRICS = {'cls_mAP', 'top1_acc', 'val_acc'}
SECONDARY_METRICS = {'mAP50-95', 'val_loss'}
NUMBER_RE = re.compile(r'-?\d+(?:\.\d+)?')


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
    match = NUMBER_RE.search(value.strip())
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_time(value: str | None, fallback_index: int) -> datetime:
    if value:
        text = value.strip()
        if text.endswith('Z'):
            text = text[:-1] + '+00:00'
        try:
            return datetime.fromisoformat(text)
        except ValueError:
            pass
    return datetime.fromtimestamp(fallback_index, tz=timezone.utc)


def trust_tier(row: dict[str, str]) -> str:
    leakage = (row.get('leakage') or '').lower()
    eval_set = (row.get('eval_set') or '').lower()
    notes = (row.get('notes') or '').lower()
    if 'server_private' in eval_set or 'clean_split' in eval_set or leakage == 'none':
        return 'trusted'
    if 'leaked' in eval_set or 'high' in leakage or 'overlap' in leakage:
        return 'leaked'
    if 'verified zero overlap' in notes:
        return 'trusted'
    return 'mixed'


def iter_metrics(row: dict[str, str]) -> list[tuple[str, float]]:
    metrics: list[tuple[str, float]] = []
    for metric_key, value_key in (('metric', 'value'), ('metric2', 'value2')):
        metric = (row.get(metric_key) or '').strip()
        value = to_float(row.get(value_key))
        if metric and value is not None:
            metrics.append((metric, value))
    return metrics


def build_points(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        when = parse_time(row.get('timestamp'), index)
        trust = trust_tier(row)
        for metric, value in iter_metrics(row):
            points.append(
                {
                    'time': when,
                    'metric': metric,
                    'value': value,
                    'model': row.get('model', 'unknown'),
                    'dataset': row.get('dataset', ''),
                    'eval_set': row.get('eval_set', ''),
                    'trust': trust,
                    'notes': row.get('notes', ''),
                }
            )
    return points


def style_key(metric: str, trust: str) -> str:
    return f'{metric} [{trust}]'


def group_series(points: list[dict[str, object]], metrics: set[str]) -> dict[str, list[tuple[datetime, float]]]:
    grouped: dict[str, list[tuple[datetime, float]]] = defaultdict(list)
    for point in points:
        metric = point['metric']
        if metric not in metrics:
            continue
        grouped[style_key(str(metric), str(point['trust']))].append((point['time'], float(point['value'])))
    for series in grouped.values():
        series.sort(key=lambda item: item[0])
    return dict(grouped)


def trusted_primary_series(points: list[dict[str, object]]) -> list[tuple[datetime, float, str]]:
    series = []
    for point in points:
        metric = str(point['metric'])
        trust = str(point['trust'])
        eval_set = str(point['eval_set']).lower()
        if trust != 'trusted':
            continue
        if metric in PRIMARY_METRICS:
            series.append((point['time'], float(point['value']), metric))
        elif metric in DETECTION_METRICS and 'clean_split' in eval_set:
            series.append((point['time'], float(point['value']), f'{metric} clean'))
    series.sort(key=lambda item: item[0])
    return series


def frontier(rows: list[dict[str, str]], points: list[dict[str, object]]) -> dict[str, object]:
    candidates = []
    for point in points:
        if str(point['metric']) != 'mAP50':
            continue
        model = str(point['model'])
        time = point['time']
        primary = float(point['value'])
        note = str(point['notes'])
        eval_set = str(point['eval_set'])
        trust = str(point['trust'])
        candidates.append(
            {
                'label': model,
                'time': time.isoformat(),
                'mAP50': primary,
                'eval_set': eval_set,
                'trust': trust,
                'notes': note,
                'mAP50_95': None,
            }
        )
    for row in rows:
        metrics = dict(iter_metrics(row))
        if 'mAP50' not in metrics or 'mAP50-95' not in metrics:
            continue
        label = row.get('model', 'unknown')
        time = parse_time(row.get('timestamp'), 0).isoformat()
        for candidate in candidates:
            if candidate['label'] == label and candidate['time'] == time:
                candidate['mAP50_95'] = metrics['mAP50-95']
                break

    paired = [row for row in candidates if row['mAP50_95'] is not None]
    frontier_rows = []
    for row in paired:
        dominated = False
        for other in paired:
            if other is row:
                continue
            if (
                other['mAP50'] >= row['mAP50']
                and other['mAP50_95'] >= row['mAP50_95']
                and (other['mAP50'] > row['mAP50'] or other['mAP50_95'] > row['mAP50_95'])
            ):
                dominated = True
                break
        if not dominated:
            frontier_rows.append(row)
    frontier_rows.sort(key=lambda item: (item['mAP50_95'], item['mAP50']))
    return {'points': paired, 'frontier': frontier_rows}


def plot_series(ax, grouped: dict[str, list[tuple[datetime, float]]], title: str, ylabel: str) -> None:
    if not grouped:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return
    for label, series in sorted(grouped.items()):
        xs = [item[0] for item in series]
        ys = [item[1] for item in series]
        linestyle = '--' if '[leaked]' in label else '-'
        alpha = 0.7 if '[trusted]' in label else 0.5
        ax.plot(xs, ys, marker='o', ms=3, linewidth=1.4, linestyle=linestyle, alpha=alpha, label=label)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.results)
    points = build_points(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.frontier_json:
        args.frontier_json.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    ax_primary, ax_detection, ax_classification, ax_secondary = axes.flatten()

    primary = trusted_primary_series(points)
    if primary:
        xs = [item[0] for item in primary]
        ys = [item[1] for item in primary]
        ax_primary.plot(xs, ys, marker='o', color='tab:green', linewidth=1.6, label='trusted primary')
        running_best = []
        best = float('-inf')
        for _, value, _ in primary:
            best = max(best, value)
            running_best.append(best)
        ax_primary.plot(xs, running_best, color='black', linewidth=1.2, linestyle='--', label='best so far')
        ax_primary.legend(fontsize=8)
    else:
        ax_primary.text(0.5, 0.5, 'No trusted primary scores yet', ha='center', va='center', transform=ax_primary.transAxes)
    ax_primary.set_title('Trusted Score Progress')
    ax_primary.set_ylabel('Score')
    ax_primary.grid(alpha=0.3)

    plot_series(ax_detection, group_series(points, DETECTION_METRICS), 'Detection Metrics', 'mAP')
    plot_series(ax_classification, group_series(points, CLASSIFICATION_METRICS), 'Classification Metrics', 'Score')

    secondary_groups = group_series(points, SECONDARY_METRICS)
    if secondary_groups:
        plot_series(ax_secondary, secondary_groups, 'Secondary Metrics', 'Score / Loss')
    else:
        trust_counts = defaultdict(int)
        for point in points:
            trust_counts[str(point['trust'])] += 1
        labels = list(sorted(trust_counts))
        values = [trust_counts[label] for label in labels]
        ax_secondary.bar(labels, values, color=['tab:green', 'tab:orange', 'tab:red'][: len(labels)])
        ax_secondary.set_title('Evidence By Trust Tier')
        ax_secondary.set_ylabel('Metric Count')
        ax_secondary.grid(alpha=0.3, axis='y')

    for ax in axes[-1]:
        ax.set_xlabel('Time')
    for ax in axes.flatten():
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))

    fig.suptitle('Object Detection Progress Metrics', fontsize=16)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)

    if args.frontier_json:
        args.frontier_json.write_text(json.dumps(frontier(rows, points), indent=2, ensure_ascii=False) + '\n', encoding='utf-8')

    print(f'Wrote {args.output}')
    if args.frontier_json:
        print(f'Wrote {args.frontier_json}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate projection-readiness artifacts for object detection."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / 'tasks' / 'object-detection'
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
    text = value.strip().replace('~', '')
    if not text or text in {'TIMEOUT', 'N/A'}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def extract_metrics(row: dict[str, str]) -> dict[str, float]:
    metrics = {}
    for metric_key, value_key in (('metric', 'value'), ('metric2', 'value2')):
        metric = (row.get(metric_key) or '').strip()
        value = to_float(row.get(value_key))
        if metric and value is not None:
            metrics[metric] = value
    return metrics


def main() -> None:
    args = parse_args()
    rows = read_rows(args.results)

    trusted_server = []
    trusted_clean = []
    for row in rows:
        eval_set = (row.get('eval_set') or '').lower()
        metrics = extract_metrics(row)
        model = row.get('model', 'unknown')
        if 'server_private' in eval_set and 'combined' in metrics:
            trusted_server.append({'model': model, 'score': metrics['combined'], 'notes': row.get('notes', '')})
        if 'clean_split' in eval_set and 'mAP50' in metrics:
            trusted_clean.append({'model': model, 'mAP50': metrics['mAP50'], 'mAP50_95': metrics.get('mAP50-95'), 'notes': row.get('notes', '')})

    report = {
        'projection_status': 'insufficient_calibration',
        'reason': 'No paired clean-val/server observations exist yet for a defensible OD server projection model.',
        'trusted_server_scores': trusted_server,
        'trusted_clean_candidates': trusted_clean,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_server, ax_text = axes

    if trusted_server:
        ax_server.bar([item['model'] for item in trusted_server], [item['score'] for item in trusted_server], color='tab:green', label='server observed')
    if trusted_clean:
        ax_server.scatter([item['model'] for item in trusted_clean], [item['mAP50'] for item in trusted_clean], color='tab:blue', s=90, marker='o', label='clean mAP50')
    ax_server.set_title('Trusted OD Evidence')
    ax_server.set_ylabel('Score / mAP')
    ax_server.tick_params(axis='x', rotation=45)
    ax_server.grid(alpha=0.3)
    ax_server.legend(fontsize=8)

    ax_text.axis('off')
    ax_text.text(
        0.0,
        1.0,
        '\n'.join(
            [
                'Projection Status: unavailable',
                '',
                'Why:',
                '- We have trusted server results.',
                '- We have a trusted clean-split candidate.',
                '- We do not yet have paired clean-val/server observations',
                '  for the same candidate family, so any numeric server',
                '  projection would be made up.',
                '',
                'Current trusted shortlist:',
            ]
            + [f"- server: {item['model']} = {item['score']:.4f}" for item in trusted_server[:3]]
            + [f"- clean: {item['model']} mAP50={item['mAP50']:.3f}" for item in trusted_clean[:3]]
        ),
        va='top',
        fontsize=10,
        family='monospace',
    )
    fig.suptitle('Object Detection Projection Readiness', fontsize=16)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)

    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')
    print(f'Wrote {args.output_json}')


if __name__ == '__main__':
    main()

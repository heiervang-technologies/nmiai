#!/usr/bin/env python3
"""Append manually observed dashboard scores and enrich them with local log metadata."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / 'tasks' / 'accounting'
DEFAULT_LOG_DIR = Path('/tmp/accounting-logs')
DEFAULT_TRACKER = TASK_DIR / 'dashboard_scores.tsv'

HEADER = [
    'recorded_at',
    'log_ts',
    'family',
    'dashboard_score',
    'notes',
    'api_calls',
    'api_errors',
    'successful_writes',
    'attachment_present',
    'prompt_preview',
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-ts', required=True, help='Timestamp stem from /tmp/accounting-logs/<ts>.json')
    parser.add_argument('--dashboard-score', required=True, type=float)
    parser.add_argument('--notes', default='')
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument('--tracker', type=Path, default=DEFAULT_TRACKER)
    return parser.parse_args()


def ensure_header(path: Path) -> None:
    if path.exists():
        return
    path.write_text('\t'.join(HEADER) + '\n', encoding='utf-8')


def successful_writes(calls: list[dict]) -> int:
    return sum(1 for call in calls if call.get('method') in {'POST', 'PUT', 'DELETE'} and 200 <= int(call.get('status', 0)) < 300)


def main() -> None:
    args = parse_args()
    log_path = args.log_dir / f'{args.log_ts}.json'
    if not log_path.exists():
        raise FileNotFoundError(f'Missing log file: {log_path}')
    payload = json.loads(log_path.read_text(encoding='utf-8'))
    calls = ((payload.get('api_stats') or {}).get('calls') or [])
    row = {
        'recorded_at': datetime.now(timezone.utc).isoformat(),
        'log_ts': args.log_ts,
        'family': ((payload.get('plan') or {}).get('family') or 'unknown'),
        'dashboard_score': args.dashboard_score,
        'notes': args.notes,
        'api_calls': (payload.get('api_stats') or {}).get('total_calls', 0),
        'api_errors': (payload.get('api_stats') or {}).get('errors_4xx', 0),
        'successful_writes': successful_writes(calls),
        'attachment_present': bool(payload.get('files')),
        'prompt_preview': (payload.get('prompt') or '')[:180],
    }

    ensure_header(args.tracker)
    existing = args.tracker.read_text(encoding='utf-8').splitlines()[1:] if args.tracker.exists() else []
    if any(line.split('\t', 2)[1] == args.log_ts for line in existing if line.strip()):
        print(f'{args.log_ts} already recorded in {args.tracker}')
        return

    with args.tracker.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter='\t')
        writer.writerow(row)
    print(json.dumps(row, ensure_ascii=False))


if __name__ == '__main__':
    main()

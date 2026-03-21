#!/usr/bin/env python3
"""Sync visible Tripletex dashboard results into dashboard_scores.tsv.

This bridges the human-visible dashboard with the local `/tmp/accounting-logs`
so the accounting tracker can correlate real task outcomes with log-derived
signals.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

CDP_URL = 'http://localhost:9222'
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
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument('--tracker', type=Path, default=DEFAULT_TRACKER)
    parser.add_argument('--limit', type=int, default=20, help='Max visible dashboard task results to ingest')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def ensure_header(path: Path) -> None:
    if path.exists():
        return
    path.write_text('\t'.join(HEADER) + '\n', encoding='utf-8')


def successful_writes(calls: list[dict]) -> int:
    return sum(1 for call in calls if call.get('method') in {'POST', 'PUT', 'DELETE'} and 200 <= int(call.get('status', 0)) < 300)


def load_existing_log_ts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding='utf-8', newline='') as handle:
        rows = csv.DictReader(handle, delimiter='\t')
        return {row.get('log_ts', '').strip() for row in rows if row.get('log_ts')}


def find_matching_log(log_dir: Path, utc_time: str) -> dict | None:
    logs = sorted(log_dir.glob('*.json'))
    logs = [path for path in logs if path.name != 'summary.jsonl']
    try:
        hour, minute = (int(part) for part in utc_time.split(':', 1))
    except ValueError:
        return None
    for path in reversed(logs):
        payload = json.loads(path.read_text(encoding='utf-8'))
        ts = payload.get('timestamp', '')
        if len(ts) < 15 or not ts.startswith(f'20260321_{hour:02d}'):
            continue
        ts_minute = int(ts[13:15])
        if abs(ts_minute - minute) > 3:
            continue
        calls = ((payload.get('api_stats') or {}).get('calls') or [])
        return {
            'log_ts': ts,
            'family': ((payload.get('plan') or {}).get('family') or 'unknown'),
            'prompt': (payload.get('prompt') or '')[:180],
            'api_calls': (payload.get('api_stats') or {}).get('total_calls', 0),
            'api_errors': (payload.get('api_stats') or {}).get('errors_4xx', 0),
            'successful_writes': successful_writes(calls),
            'attachment_present': bool(payload.get('files')),
        }
    return None


def cet_to_utc_time(cet_text: str) -> str | None:
    try:
        dt = datetime.strptime(cet_text, '%I:%M %p')
    except ValueError:
        return None
    hour = dt.hour - 1
    if hour < 0:
        hour += 24
    return f'{hour:02d}:{dt.minute:02d}'


def parse_blocks(text: str, limit: int) -> list[dict]:
    idx = text.find('Recent Results')
    if idx < 0:
        return []
    results_text = text[idx:]
    blocks = re.split(r'(?=Task \(\d)', results_text)
    blocks = [block for block in blocks if block.startswith('Task (')]
    parsed = []
    for block in blocks[:limit]:
        match = re.match(
            r'Task \((\d+\.?\d*)/(\d+)\)\s*\n(\d{2}:\d{2} [AP]M) · ([\d.]+s)\s*\n[\d.]+/\d+ \((\d+)%\)',
            block,
        )
        if not match:
            continue
        points, max_points, cet_time, duration, pct = match.groups()
        checks = re.findall(r'Check (\d+): (passed|failed)', block)
        parsed.append(
            {
                'points': points,
                'max_points': max_points,
                'cet_time': cet_time,
                'duration': duration,
                'pct': float(pct),
                'checks': checks,
            }
        )
    return parsed


async def collect_dashboard_rows(args: argparse.Namespace) -> list[dict]:
    from playwright.async_api import async_playwright

    playwright = await async_playwright().start()
    try:
        browser = await playwright.chromium.connect_over_cdp(CDP_URL)
        page = None
        for context in browser.contexts:
            for candidate in context.pages:
                if 'submit/tripletex' in candidate.url:
                    page = candidate
                    break
            if page:
                break
        if page is None:
            raise RuntimeError('No submit/tripletex tab found in Brave')

        await page.evaluate(
            r'''() => {
                const buttons = Array.from(document.querySelectorAll('button'));
                const taskButtons = buttons.filter(button => /Task \(/.test(button.textContent));
                taskButtons.forEach(button => {
                    const next = button.nextElementSibling;
                    if (!next || !next.textContent.includes('Check')) {
                        button.click();
                    }
                });
            }'''
        )
        await asyncio.sleep(1.0)
        text = await page.evaluate('() => document.body.innerText')
    finally:
        await playwright.stop()

    existing = load_existing_log_ts(args.tracker)
    rows = []
    for block in parse_blocks(text, args.limit):
        utc_time = cet_to_utc_time(block['cet_time'])
        if utc_time is None:
            continue
        log_match = find_matching_log(args.log_dir, utc_time)
        if not log_match:
            continue
        log_ts = log_match['log_ts']
        if log_ts in existing:
            continue
        passed = sum(1 for _, state in block['checks'] if state == 'passed')
        failed = sum(1 for _, state in block['checks'] if state == 'failed')
        rows.append(
            {
                'recorded_at': datetime.now(timezone.utc).isoformat(),
                'log_ts': log_ts,
                'family': log_match['family'],
                'dashboard_score': block['pct'],
                'notes': f"task={block['points']}/{block['max_points']}; cet={block['cet_time']}; duration={block['duration']}; checks={passed}P/{failed}F",
                'api_calls': log_match['api_calls'],
                'api_errors': log_match['api_errors'],
                'successful_writes': log_match['successful_writes'],
                'attachment_present': str(log_match['attachment_present']).lower(),
                'prompt_preview': log_match['prompt'],
            }
        )
        existing.add(log_ts)
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    ensure_header(path)
    with path.open('a', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, delimiter='\t')
        for row in rows:
            writer.writerow(row)


async def async_main() -> None:
    args = parse_args()
    rows = await collect_dashboard_rows(args)
    if rows and not args.dry_run:
        write_rows(args.tracker, rows)
    print(json.dumps({'rows_added': len(rows), 'tracker': str(args.tracker), 'dry_run': args.dry_run, 'families': [row['family'] for row in rows]}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    asyncio.run(async_main())

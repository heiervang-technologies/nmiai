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
from datetime import datetime, timedelta, timezone
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


def parse_log_datetime(ts: str) -> datetime | None:
    if len(ts) < 15:
        return None
    try:
        return datetime.strptime(ts, '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def load_existing_log_ts(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding='utf-8', newline='') as handle:
        rows = csv.DictReader(handle, delimiter='\t')
        return {row.get('log_ts', '').strip() for row in rows if row.get('log_ts')}


def load_logs(log_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(log_dir.glob('*.json')):
        if path.name == 'summary.jsonl':
            continue
        payload = json.loads(path.read_text(encoding='utf-8'))
        ts = payload.get('timestamp', '')
        dt = parse_log_datetime(ts)
        if dt is None:
            continue
        calls = ((payload.get('api_stats') or {}).get('calls') or [])
        rows.append(
            {
                'log_ts': ts,
                'dt': dt,
                'family': ((payload.get('plan') or {}).get('family') or 'unknown'),
                'prompt': (payload.get('prompt') or '')[:180],
                'api_calls': (payload.get('api_stats') or {}).get('total_calls', 0),
                'api_errors': (payload.get('api_stats') or {}).get('errors_4xx', 0),
                'successful_writes': successful_writes(calls),
                'attachment_present': bool(payload.get('files')),
            }
        )
    return rows


def parse_duration_seconds(duration_text: str) -> float | None:
    text = duration_text.strip().rstrip('s')
    try:
        return float(text)
    except ValueError:
        return None


def cet_to_target_utc_datetime(cet_text: str, duration_text: str) -> datetime | None:
    try:
        local_time = datetime.strptime(cet_text, '%I:%M %p')
    except ValueError:
        return None
    duration_seconds = parse_duration_seconds(duration_text)
    if duration_seconds is None:
        return None
    now = datetime.now(timezone.utc)
    completed = datetime(now.year, now.month, now.day, local_time.hour - 1, local_time.minute, tzinfo=timezone.utc)
    if completed.hour < 0:
        completed += timedelta(days=-1)
    return completed - timedelta(seconds=duration_seconds)


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


def match_logs(blocks: list[dict], logs: list[dict], max_gap_seconds: float = 120.0) -> list[dict | None]:
    tasks = []
    for idx, block in enumerate(blocks):
        target_dt = cet_to_target_utc_datetime(block['cet_time'], block['duration'])
        tasks.append({'index': idx, 'target_dt': target_dt})

    matched: list[dict | None] = [None] * len(blocks)
    used: set[str] = set()
    sortable_tasks = [task for task in tasks if task['target_dt'] is not None]
    sortable_tasks.sort(key=lambda item: item['target_dt'], reverse=True)

    for task in sortable_tasks:
        candidates = []
        for log in logs:
            if log['log_ts'] in used:
                continue
            gap = abs((log['dt'] - task['target_dt']).total_seconds())
            if gap > max_gap_seconds:
                continue
            candidates.append((gap, log['dt'], log))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], -item[1].timestamp()))
        gap, _, log = candidates[0]
        used.add(log['log_ts'])
        confidence = 'high' if gap <= 30 else 'medium'
        matched[task['index']] = {**log, 'match_gap_seconds': round(gap, 1), 'match_confidence': confidence}
    return matched


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
    blocks = parse_blocks(text, args.limit)
    matched_logs = match_logs(blocks, load_logs(args.log_dir))
    rows = []
    for block, log_match in zip(blocks, matched_logs):
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

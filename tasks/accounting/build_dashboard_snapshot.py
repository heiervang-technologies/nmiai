#!/usr/bin/env python3
"""Build a versioned dashboard snapshot artifact with matched accounting logs."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

CDP_URL = 'http://localhost:9222'
ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / 'tasks' / 'accounting'
DEFAULT_LOG_DIR = Path('/tmp/accounting-logs')
DEFAULT_OUTPUT_MD = TASK_DIR / 'analysis' / 'dashboard_snapshot.md'
DEFAULT_OUTPUT_JSON = TASK_DIR / 'analysis' / 'dashboard_snapshot.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument('--output-md', type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument('--limit', type=int, default=20)
    return parser.parse_args()


def parse_log_datetime(ts: str) -> datetime | None:
    if len(ts) < 15:
        return None
    try:
        return datetime.strptime(ts, '%Y%m%d_%H%M%S').replace(tzinfo=timezone.utc)
    except ValueError:
        return None


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
        error_details = []
        for call in calls:
            status = int(call.get('status', 0) or 0)
            if status < 400:
                continue
            error_details.append(
                {
                    'method': call.get('method', ''),
                    'path': call.get('path', ''),
                    'status': status,
                    'error': str(call.get('error', ''))[:200],
                }
            )
        rows.append(
            {
                'timestamp': ts,
                'dt': dt,
                'family': ((payload.get('plan') or {}).get('family') or 'unknown'),
                'prompt_preview': (payload.get('prompt') or '')[:220],
                'result_preview': (((payload.get('result') or {}).get('final_message')) or '')[:220],
                'api_calls': len(calls),
                'api_errors': sum(1 for call in calls if int(call.get('status', 0) or 0) >= 400),
                'successful_writes': sum(
                    1
                    for call in calls
                    if call.get('method') in {'POST', 'PUT', 'DELETE'} and 200 <= int(call.get('status', 0) or 0) < 300
                ),
                'error_details': error_details[:3],
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
            if log['timestamp'] in used:
                continue
            gap = abs((log['dt'] - task['target_dt']).total_seconds())
            if gap > max_gap_seconds:
                continue
            candidates.append((gap, log['dt'], log))
        if not candidates:
            continue
        candidates.sort(key=lambda item: (item[0], -item[1].timestamp()))
        gap, _, log = candidates[0]
        used.add(log['timestamp'])
        confidence = 'high' if gap <= 30 else 'medium'
        matched[task['index']] = {**log, 'match_gap_seconds': round(gap, 1), 'match_confidence': confidence}
    return matched


async def collect_dashboard_snapshot(args: argparse.Namespace) -> dict:
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

    score_match = re.search(r'Total Score\s*\n\s*([\d.]+)', text)
    rank_match = re.search(r'Rank\s*\n\s*#?(\d+)', text)
    submissions_match = re.search(r'(\d+)\s*/\s*300\s*daily', text)

    blocks = parse_blocks(text, args.limit)
    matched_logs = match_logs(blocks, load_logs(args.log_dir))

    tasks = []
    for block, log_match in zip(blocks, matched_logs):
        passed = sum(1 for _, state in block['checks'] if state == 'passed')
        failed = sum(1 for _, state in block['checks'] if state == 'failed')
        tasks.append(
            {
                'points': float(block['points']),
                'max_points': float(block['max_points']),
                'dashboard_score': block['pct'],
                'cet_time': block['cet_time'],
                'utc_time': None if log_match is None else log_match['dt'].strftime('%H:%M:%S'),
                'duration': block['duration'],
                'checks_passed': passed,
                'checks_failed': failed,
                'family': (log_match or {}).get('family', '?'),
                'log_timestamp': (log_match or {}).get('timestamp', ''),
                'match_confidence': (log_match or {}).get('match_confidence', 'none'),
                'match_gap_seconds': (log_match or {}).get('match_gap_seconds'),
                'api_calls': (log_match or {}).get('api_calls'),
                'api_errors': (log_match or {}).get('api_errors'),
                'successful_writes': (log_match or {}).get('successful_writes'),
                'prompt_preview': (log_match or {}).get('prompt_preview', ''),
                'result_preview': (log_match or {}).get('result_preview', ''),
                'error_details': (log_match or {}).get('error_details', []),
            }
        )

    return {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'total_score': float(score_match.group(1)) if score_match else None,
        'rank': int(rank_match.group(1)) if rank_match else None,
        'submissions_used': int(submissions_match.group(1)) if submissions_match else None,
        'tasks': tasks,
    }


def render_markdown(snapshot: dict) -> str:
    lines = [
        '# Accounting Dashboard Snapshot',
        '',
        f"Generated: `{snapshot.get('generated_at', '')}`",
        f"Total score: `{snapshot.get('total_score', 'n/a')}`",
        f"Rank: `#{snapshot.get('rank', 'n/a')}`",
        f"Daily submissions used: `{snapshot.get('submissions_used', 'n/a')}`",
        '',
        '## Recent Results',
        '',
    ]
    for index, task in enumerate(snapshot.get('tasks', []), start=1):
        lines.append(
            '- '
            + f"[{index}] {task['points']}/{task['max_points']} ({task['dashboard_score']}%), family={task['family']}, "
            + f"time={task['cet_time']}, duration={task['duration']}, checks={task['checks_passed']}P/{task['checks_failed']}F, "
            + f"api={task.get('api_calls', '?')}c/{task.get('api_errors', '?')}e/{task.get('successful_writes', '?')}w, "
            + f"match={task.get('match_confidence', 'none')}"
        )
        if task.get('dashboard_score', 0.0) < 50:
            lines.append(f"  prompt: {task.get('prompt_preview', '')}")
            lines.append(f"  result: {task.get('result_preview', '')}")
            details = task.get('error_details', [])
            if details:
                rendered = '; '.join(f"{item['method']} {item['path']} [{item['status']}] {item['error']}" for item in details[:2])
                lines.append(f"  errors: {rendered}")
    if not snapshot.get('tasks'):
        lines.append('- none')
    return '\n'.join(lines) + '\n'


async def async_main() -> None:
    args = parse_args()
    snapshot = await collect_dashboard_snapshot(args)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    args.output_md.write_text(render_markdown(snapshot), encoding='utf-8')
    print(f'Wrote {args.output_json}')
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    asyncio.run(async_main())

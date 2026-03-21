#!/usr/bin/env python3
"""Build a versioned dashboard snapshot artifact with matched accounting logs."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from datetime import datetime, timezone
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
        return {
            'timestamp': ts,
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

    tasks = []
    for block in parse_blocks(text, args.limit):
        utc_time = cet_to_utc_time(block['cet_time'])
        log_match = find_matching_log(args.log_dir, utc_time) if utc_time else None
        passed = sum(1 for _, state in block['checks'] if state == 'passed')
        failed = sum(1 for _, state in block['checks'] if state == 'failed')
        tasks.append(
            {
                'points': float(block['points']),
                'max_points': float(block['max_points']),
                'dashboard_score': block['pct'],
                'cet_time': block['cet_time'],
                'utc_time': utc_time,
                'duration': block['duration'],
                'checks_passed': passed,
                'checks_failed': failed,
                'family': (log_match or {}).get('family', '?'),
                'log_timestamp': (log_match or {}).get('timestamp', ''),
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
            + f"api={task.get('api_calls', '?')}c/{task.get('api_errors', '?')}e/{task.get('successful_writes', '?')}w"
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

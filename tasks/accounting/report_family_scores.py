#!/usr/bin/env python3
"""Build a compact family-level scoreboard from dashboard, projection, and priority data."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / 'tasks' / 'accounting'
DEFAULT_DASHBOARD = TASK_DIR / 'dashboard_scores.tsv'
DEFAULT_PRIORITY = TASK_DIR / 'analysis' / 'priority_queue.json'
DEFAULT_PROJECTION = TASK_DIR / 'analysis' / 'projection_report.json'
DEFAULT_JSON = TASK_DIR / 'analysis' / 'family_scoreboard.json'
DEFAULT_MD = TASK_DIR / 'analysis' / 'family_scoreboard.md'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dashboard', type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument('--priority', type=Path, default=DEFAULT_PRIORITY)
    parser.add_argument('--projection', type=Path, default=DEFAULT_PROJECTION)
    parser.add_argument('--output-json', type=Path, default=DEFAULT_JSON)
    parser.add_argument('--output-md', type=Path, default=DEFAULT_MD)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle, delimiter='\t'))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding='utf-8'))


def to_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def best_dashboard_by_family(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    best: dict[str, dict[str, object]] = {}
    for row in rows:
        family = row.get('family', 'unknown')
        score = to_float(row.get('dashboard_score'))
        if score is None:
            continue
        current = best.get(family)
        if current is None or score >= float(current['score']):
            best[family] = {
                'score': score,
                'log_ts': row.get('log_ts', ''),
                'notes': row.get('notes', ''),
            }
    return best


def dashboard_stats_by_family(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for row in rows:
        family = row.get('family', 'unknown')
        score = to_float(row.get('dashboard_score'))
        if score is None:
            continue
        bucket = stats.setdefault(family, {'count': 0.0, 'sum': 0.0, 'best': 0.0})
        bucket['count'] += 1.0
        bucket['sum'] += score
        bucket['best'] = max(bucket['best'], score)
    for family, bucket in stats.items():
        bucket['avg'] = bucket['sum'] / bucket['count'] if bucket['count'] else 0.0
    return stats


def projection_by_family(payload: dict) -> dict[str, dict]:
    family_rows = payload.get('family_projections', []) or []
    return {row.get('family', 'unknown'): row for row in family_rows if row.get('family')}


def priority_by_family(payload: dict) -> dict[str, dict]:
    items = payload.get('priority_targets', []) or []
    return {row.get('family', 'unknown'): row for row in items if row.get('family')}


def score_estimate(best_observed: float | None, projected: float | None) -> tuple[float | None, str]:
    if best_observed is not None:
        return best_observed, 'observed'
    if projected is not None:
        return projected, 'projected'
    return None, 'none'


def stabilized_estimate(observed_scores: list[float], projected: float | None, prior_weight: float = 2.0) -> float | None:
    if not observed_scores:
        return projected
    observed_sum = sum(observed_scores)
    observed_count = len(observed_scores)
    if projected is None:
        return observed_sum / observed_count
    return (observed_sum + (prior_weight * projected)) / (observed_count + prior_weight)


def opportunity_score(priority_score: float | None, gap_to_100: float | None) -> float | None:
    if priority_score is None:
        return None
    if gap_to_100 is None:
        return priority_score
    return priority_score * (gap_to_100 / 100.0)


def render_markdown(rows: list[dict[str, object]], generated_at: str, calibration_status: str) -> str:
    lines = [
        '# Accounting Family Scoreboard',
        '',
        f'Generated: `{generated_at}`',
        f'Projection status: `{calibration_status}`',
        '',
        '## Best Observed / Estimated By Family',
        '',
    ]
    for row in rows:
        observed = row.get('best_observed_dashboard_score')
        projected = row.get('projected_dashboard_score')
        estimated = row.get('current_estimated_score')
        blockers = ', '.join(row.get('blockers', [])[:3]) or 'none'
        missing = ', '.join(row.get('missing_fields', [])[:3]) or 'none'
        lines.append(
            '- '
            + f"{row['family']}: est={estimated if estimated is not None else 'n/a'} "
            + f"({row['estimate_source']}), observed_best={observed if observed is not None else 'n/a'}, "
            + f"projected={projected if projected is not None else 'n/a'}, gap_to_100={row['gap_to_100'] if row['gap_to_100'] is not None else 'n/a'}, "
            + f"clean={row['proxy_clean_rate'] if row['proxy_clean_rate'] is not None else 'n/a'}, priority={row['priority_score'] if row['priority_score'] is not None else 'n/a'}, opportunity={row['opportunity_score'] if row['opportunity_score'] is not None else 'n/a'}, "
            + f"blockers={blockers}, missing={missing}"
        )

    lines.extend(['', '## Fix First', ''])
    urgent = sorted(
        [row for row in rows if row.get('priority_score') is not None],
        key=lambda row: (
            -(float(row['opportunity_score']) if row['opportunity_score'] is not None else 0.0),
            -(float(row['priority_score']) if row['priority_score'] is not None else 0.0),
            float(row.get('current_estimated_score') or 0.0),
        ),
    )
    for row in urgent[:8]:
        lines.append(
            '- '
            + f"{row['family']}: opportunity={row['opportunity_score'] if row['opportunity_score'] is not None else 'n/a'}, priority={row['priority_score']:.2f}, est={row['current_estimated_score'] if row['current_estimated_score'] is not None else 'n/a'}, "
            + f"gap={row['gap_to_100'] if row['gap_to_100'] is not None else 'n/a'}, blockers={', '.join(row.get('blockers', [])[:4]) or 'none'}"
        )
    if not urgent:
        lines.append('- none')
    return '\n'.join(lines) + '\n'


def main() -> None:
    args = parse_args()
    dashboard = read_tsv(args.dashboard)
    priority = read_json(args.priority)
    projection = read_json(args.projection)

    best_dashboard = best_dashboard_by_family(dashboard)
    dashboard_stats = dashboard_stats_by_family(dashboard)
    projections = projection_by_family(projection)
    priorities = priority_by_family(priority)
    families = sorted(set(best_dashboard) | set(projections) | set(priorities))

    rows = []
    for family in families:
        observed = to_float((best_dashboard.get(family) or {}).get('score'))
        projected = to_float((projections.get(family) or {}).get('projected_dashboard_score'))
        proxy_clean_rate = to_float((projections.get(family) or {}).get('proxy_clean_rate'))
        stats = dashboard_stats.get(family, {})
        observed_count = int(stats.get('count', 0.0))
        observed_avg = to_float(stats.get('avg'))
        observed_scores = []
        if observed_avg is not None and observed_count > 0:
            observed_scores = [observed_avg] * observed_count
        estimate = stabilized_estimate(observed_scores, projected)
        source = 'stabilized' if observed_count > 0 and projected is not None else ('observed_avg' if observed_count > 0 else 'projected')
        if estimate is None:
            estimate, source = score_estimate(observed, projected)
        gap_to_100 = None if estimate is None else round(max(0.0, 100.0 - estimate), 1)
        priority_item = priorities.get(family) or {}
        rows.append(
            {
                'family': family,
                'best_observed_dashboard_score': observed,
                'observed_dashboard_avg': None if observed_avg is None else round(observed_avg, 1),
                'observed_dashboard_count': observed_count,
                'projected_dashboard_score': None if projected is None else round(projected, 1),
                'current_estimated_score': None if estimate is None else round(estimate, 1),
                'estimate_source': source,
                'gap_to_100': gap_to_100,
                'proxy_clean_rate': None if proxy_clean_rate is None else round(proxy_clean_rate, 3),
                'priority_score': to_float(priority_item.get('priority_score')),
                'opportunity_score': None,
                'blockers': priority_item.get('blockers', []) or [],
                'missing_fields': priority_item.get('missing_fields', []) or [],
                'status': (projections.get(family) or {}).get('status') or priority_item.get('status') or '',
                'best_observed_log_ts': (best_dashboard.get(family) or {}).get('log_ts', ''),
            }
        )

    for row in rows:
        row['opportunity_score'] = None if row['priority_score'] is None else round(opportunity_score(float(row['priority_score']), row['gap_to_100']), 3)

    rows.sort(
        key=lambda row: (
            float(row['current_estimated_score']) if row['current_estimated_score'] is not None else -1.0,
            -(float(row['priority_score']) if row['priority_score'] is not None else 0.0),
            row['family'],
        ),
        reverse=True,
    )

    payload = {
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'projection_status': projection.get('projection_status', 'unavailable'),
        'family_scores': rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    args.output_md.write_text(
        render_markdown(rows, payload['generated_at'], str(payload['projection_status'])),
        encoding='utf-8',
    )
    print(f'Wrote {args.output_json}')
    print(f'Wrote {args.output_md}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate explicit projection artifacts for accounting."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
TASK_DIR = ROOT / 'tasks' / 'accounting'
DEFAULT_RESULTS = TASK_DIR / 'autoresearch_results.tsv'
DEFAULT_DASHBOARD = TASK_DIR / 'dashboard_scores.tsv'
DEFAULT_OUT = TASK_DIR / 'analysis' / 'projection.png'
DEFAULT_JSON = TASK_DIR / 'analysis' / 'projection_report.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=DEFAULT_RESULTS)
    parser.add_argument('--dashboard', type=Path, default=DEFAULT_DASHBOARD)
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
    if not text:
        return None
    try:
        return float(text)
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


def fit_line(xs: list[float], ys: list[float]) -> tuple[float, float, float]:
    count = len(xs)
    mean_x = sum(xs) / count
    mean_y = sum(ys) / count
    var_x = sum((x - mean_x) ** 2 for x in xs)
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov_xy / var_x if var_x else 0.0
    intercept = mean_y - slope * mean_x
    rmse = math.sqrt(sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys)) / count)
    return slope, intercept, rmse


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def build_paired_points(result_rows: list[dict[str, str]], dashboard_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    results_by_key = {(row.get('batch_id', ''), row.get('family', 'unknown')): row for row in result_rows}
    paired = []
    for row in dashboard_rows:
        key = (row.get('log_ts', ''), row.get('family', 'unknown'))
        result = results_by_key.get(key)
        if result is None:
            continue
        proxy_clean_rate = to_float(result.get('proxy_clean_rate'))
        dashboard_score = to_float(row.get('dashboard_score'))
        if proxy_clean_rate is None or dashboard_score is None:
            continue
        paired.append(
            {
                'family': key[1],
                'batch_id': key[0],
                'proxy_clean_rate': proxy_clean_rate,
                'dashboard_score': dashboard_score,
                'api_errors': to_float(result.get('api_errors')),
                'successful_writes': to_float(result.get('successful_writes')),
                'attachment_present': (result.get('attachment_present') or '').strip().lower() == 'true',
                'description': result.get('description', ''),
            }
        )
    return paired


def latest_rows_by_family(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    latest: dict[str, tuple[datetime, dict[str, str]]] = {}
    for index, row in enumerate(rows):
        family = row.get('family', 'unknown')
        when = parse_time(row.get('timestamp'), index)
        current = latest.get(family)
        if current is None or when >= current[0]:
            latest[family] = (when, row)
    return {family: row for family, (_, row) in latest.items()}


def best_dashboard_by_family(rows: list[dict[str, str]]) -> dict[str, float]:
    best: dict[str, float] = {}
    for row in rows:
        family = row.get('family', 'unknown')
        score = to_float(row.get('dashboard_score'))
        if score is None:
            continue
        best[family] = max(score, best.get(family, score))
    return best


def main() -> None:
    args = parse_args()
    result_rows = read_rows(args.results)
    dashboard_rows = read_rows(args.dashboard)
    paired = build_paired_points(result_rows, dashboard_rows)
    latest = latest_rows_by_family(result_rows)
    best_dash = best_dashboard_by_family(dashboard_rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        'projection_status': 'insufficient_data',
        'paired_points': paired,
        'family_projections': [],
        'best_observed_dashboard_by_family': best_dash,
    }

    calibration = None
    if len(paired) >= 2:
        xs = [float(point['proxy_clean_rate']) for point in paired]
        ys = [float(point['dashboard_score']) for point in paired]
        slope, intercept, rmse = fit_line(xs, ys)
        calibration = {'slope': slope, 'intercept': intercept, 'rmse_dashboard_score': rmse, 'points': len(paired)}
        report['calibration'] = calibration
        report['projection_status'] = 'ok' if rmse <= 20.0 else 'noisy_calibration'
        report['reason'] = (
            'Dashboard-backed calibration exists but remains noisy.'
            if rmse > 20.0
            else 'Dashboard-backed calibration exists and is reasonably stable.'
        )
    else:
        report['reason'] = 'Need at least two paired proxy/dashboard observations before projecting score.'

    family_projections = []
    if calibration is not None:
        slope = float(calibration['slope'])
        intercept = float(calibration['intercept'])
        for family, row in sorted(latest.items()):
            proxy_clean_rate = to_float(row.get('proxy_clean_rate'))
            if proxy_clean_rate is None:
                continue
            projected = clamp(intercept + slope * proxy_clean_rate, 0.0, 100.0)
            family_projections.append(
                {
                    'family': family,
                    'batch_id': row.get('batch_id', ''),
                    'proxy_clean_rate': proxy_clean_rate,
                    'projected_dashboard_score': projected,
                    'best_observed_dashboard_score': best_dash.get(family),
                    'api_errors': to_float(row.get('api_errors')),
                    'successful_writes': to_float(row.get('successful_writes')),
                    'status': row.get('status', ''),
                }
            )
        family_projections.sort(key=lambda item: (-float(item['projected_dashboard_score']), item['family']))
        report['family_projections'] = family_projections

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_fit, ax_family = axes

    if paired:
        family_names = sorted({str(point['family']) for point in paired})
        cmap = plt.get_cmap('tab20')
        color_by_family = {family: cmap(index % 20) for index, family in enumerate(family_names)}
        for point in paired:
            family = str(point['family'])
            ax_fit.scatter(
                [float(point['proxy_clean_rate'])],
                [float(point['dashboard_score'])],
                color=color_by_family[family],
                s=70,
                label=family,
            )
        if calibration is not None:
            line_x = [0.0, 1.0]
            line_y = [clamp(float(calibration['intercept']) + float(calibration['slope']) * value, 0.0, 100.0) for value in line_x]
            ax_fit.plot(line_x, line_y, color='black', linestyle='--', linewidth=1.4)
        handles, labels = ax_fit.get_legend_handles_labels()
        dedup = dict(zip(labels, handles))
        ax_fit.legend(dedup.values(), dedup.keys(), fontsize=7, loc='lower right')
    else:
        ax_fit.text(0.5, 0.5, 'No paired proxy/dashboard rows yet.', ha='center', va='center', transform=ax_fit.transAxes)
    ax_fit.set_title('Proxy Clean Rate vs Dashboard Score')
    ax_fit.set_xlabel('Proxy Clean Rate')
    ax_fit.set_ylabel('Dashboard Score')
    ax_fit.set_xlim(-0.02, 1.02)
    ax_fit.set_ylim(-2, 102)
    ax_fit.grid(alpha=0.3)

    if family_projections:
        labels = [str(item['family']) for item in family_projections]
        projected = [float(item['projected_dashboard_score']) for item in family_projections]
        observed = [item.get('best_observed_dashboard_score') for item in family_projections]
        positions = list(range(len(labels)))
        ax_family.barh(positions, projected, color='tab:blue', alpha=0.75, label='projected from latest proxy')
        obs_x = []
        obs_y = []
        for idx, value in enumerate(observed):
            if value is None:
                continue
            obs_x.append(float(value))
            obs_y.append(idx)
        if obs_x:
            ax_family.scatter(obs_x, obs_y, color='tab:green', s=70, marker='D', label='best observed dashboard')
        ax_family.set_yticks(positions)
        ax_family.set_yticklabels(labels)
        ax_family.invert_yaxis()
        ax_family.legend(fontsize=8, loc='lower right')
    else:
        ax_family.text(0.5, 0.5, 'Need paired calibration before projecting families.', ha='center', va='center', transform=ax_family.transAxes)
    ax_family.set_title('Family Score Projection')
    ax_family.set_xlabel('Dashboard Score')
    ax_family.set_xlim(0, 100)
    ax_family.grid(alpha=0.3, axis='x')

    status = str(report['projection_status'])
    fig.suptitle(f'Accounting Projection Readiness ({status})', fontsize=16)
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)

    args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'Wrote {args.output}')
    print(f'Wrote {args.output_json}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Refresh progress plots and Pareto/frontier artifacts for all competition tasks."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print('+', ' '.join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    run('python', 'tasks/accounting/plot_progress.py')
    run('python', 'tasks/object-detection/plot_progress.py')
    run('python', 'tasks/object-detection/project_progress.py')
    run('python', 'tasks/astar-island/plot_progress.py')
    run('python', 'tasks/astar-island/project_progress.py')

    run(
        'python',
        'tools/pareto_frontier.py',
        '--input',
        'tasks/accounting/autoresearch_results.tsv',
        '--x',
        'api_errors',
        '--x-direction',
        'min',
        '--y',
        'proxy_clean_rate',
        '--y-direction',
        'max',
        '--label',
        'batch_id',
        '--time',
        'timestamp',
        '--output-json',
        'tasks/accounting/analysis/pareto_clean_vs_errors.json',
    )
    run(
        'python',
        'tools/pareto_frontier.py',
        '--input',
        'tasks/accounting/autoresearch_results.tsv',
        '--x',
        'submissions_used',
        '--x-direction',
        'min',
        '--y',
        'proxy_clean_rate',
        '--y-direction',
        'max',
        '--label',
        'batch_id',
        '--time',
        'timestamp',
        '--output-json',
        'tasks/accounting/analysis/pareto_clean_vs_submissions.json',
    )
    run(
        'python',
        'tools/pareto_frontier.py',
        '--input',
        'tasks/accounting/dashboard_scores.tsv',
        '--x',
        'api_errors',
        '--x-direction',
        'min',
        '--y',
        'dashboard_score',
        '--y-direction',
        'max',
        '--label',
        'family',
        '--time',
        'recorded_at',
        '--output-json',
        'tasks/accounting/analysis/pareto_dashboard_vs_errors.json',
    )


if __name__ == '__main__':
    main()

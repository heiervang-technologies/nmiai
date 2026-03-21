#!/usr/bin/env python3
"""Refresh the full live accounting tracking stack in one command."""

from __future__ import annotations

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def run(*args: str) -> None:
    print('+', ' '.join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    run('python', 'tasks/accounting/analyze_logs.py')
    run('python', 'tasks/accounting/sync_dashboard_scores.py')
    run('python', 'tasks/accounting/build_dashboard_snapshot.py')
    run('python', 'tools/refresh_autoresearch_artifacts.py')


if __name__ == '__main__':
    main()

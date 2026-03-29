"""Tests for tasks/accounting/sync_dashboard_scores.py — pure helper functions.

Covers: parse_log_datetime, parse_duration_seconds, parse_blocks.
All pure parsing functions — no file system or network access required.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting"))

from sync_dashboard_scores import parse_blocks, parse_duration_seconds, parse_log_datetime


# ---------------------------------------------------------------------------
# parse_log_datetime
# ---------------------------------------------------------------------------

class TestParseLogDatetime:
    def test_valid_timestamp_returns_datetime(self):
        result = parse_log_datetime("20260328_143000")
        assert isinstance(result, datetime)

    def test_year_month_day_correct(self):
        result = parse_log_datetime("20260315_090000")
        assert result.year == 2026
        assert result.month == 3
        assert result.day == 15

    def test_hour_minute_second_correct(self):
        result = parse_log_datetime("20260101_142530")
        assert result.hour == 14
        assert result.minute == 25
        assert result.second == 30

    def test_has_utc_timezone(self):
        result = parse_log_datetime("20260101_000000")
        assert result.tzinfo == timezone.utc

    def test_short_string_returns_none(self):
        assert parse_log_datetime("20260101") is None

    def test_invalid_format_returns_none(self):
        assert parse_log_datetime("not-a-timestamp") is None

    def test_empty_string_returns_none(self):
        assert parse_log_datetime("") is None

    def test_too_short_returns_none(self):
        assert parse_log_datetime("20260101_1") is None


# ---------------------------------------------------------------------------
# parse_duration_seconds
# ---------------------------------------------------------------------------

class TestParseDurationSeconds:
    def test_basic_float_with_s(self):
        result = parse_duration_seconds("12.5s")
        assert result == pytest.approx(12.5)

    def test_integer_with_s(self):
        result = parse_duration_seconds("30s")
        assert result == pytest.approx(30.0)

    def test_float_without_s(self):
        result = parse_duration_seconds("5.25")
        assert result == pytest.approx(5.25)

    def test_invalid_returns_none(self):
        result = parse_duration_seconds("not-a-number")
        assert result is None

    def test_whitespace_stripped(self):
        result = parse_duration_seconds("  10.0s  ")
        assert result == pytest.approx(10.0)

    def test_returns_float(self):
        result = parse_duration_seconds("7s")
        assert isinstance(result, float)

    def test_zero(self):
        result = parse_duration_seconds("0s")
        assert result == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# parse_blocks
# ---------------------------------------------------------------------------

class TestParseBlocks:
    def _make_text(self, tasks: list[tuple]) -> str:
        """Helper to build a fake dashboard text with Recent Results section."""
        lines = ["Some header\n\nRecent Results\n"]
        for points, max_pts, cet_time, duration, pct, checks in tasks:
            lines.append(f"Task ({points}/{max_pts})\n")
            lines.append(f"{cet_time} · {duration}\n")
            lines.append(f"{pct}/{max_pts} ({pct}%)\n")
            for check_id, outcome in checks:
                lines.append(f"Check {check_id}: {outcome}\n")
        return "".join(lines)

    def test_empty_text_returns_empty(self):
        assert parse_blocks("", limit=10) == []

    def test_no_recent_results_section_returns_empty(self):
        assert parse_blocks("Some other text\nwithout results", limit=10) == []

    def test_valid_block_parsed(self):
        text = (
            "Recent Results\n"
            "Task (3/5)\n"
            "02:30 PM · 15.2s\n"
            "3/5 (60%)\n"
            "Check 1: passed\n"
            "Check 2: failed\n"
        )
        result = parse_blocks(text, limit=10)
        assert len(result) == 1
        assert result[0]["pct"] == 60.0

    def test_limit_enforced(self):
        block = (
            "Task (3/5)\n"
            "02:30 PM · 10.0s\n"
            "3/5 (60%)\n"
        )
        # Create text with 5 identical blocks
        text = "Recent Results\n" + block * 5
        result = parse_blocks(text, limit=2)
        assert len(result) <= 2

    def test_checks_extracted(self):
        text = (
            "Recent Results\n"
            "Task (4/5)\n"
            "03:00 PM · 20.0s\n"
            "4/5 (80%)\n"
            "Check 1: passed\n"
            "Check 2: passed\n"
            "Check 3: failed\n"
        )
        result = parse_blocks(text, limit=10)
        assert len(result) == 1
        checks = result[0]["checks"]
        assert ("1", "passed") in checks
        assert ("3", "failed") in checks

    def test_points_and_max_points_captured(self):
        text = (
            "Recent Results\n"
            "Task (7/10)\n"
            "01:15 PM · 8.5s\n"
            "7/10 (70%)\n"
        )
        result = parse_blocks(text, limit=10)
        assert len(result) == 1
        assert result[0]["points"] == "7"
        assert result[0]["max_points"] == "10"

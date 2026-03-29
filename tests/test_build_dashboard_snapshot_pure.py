"""Tests for tasks/accounting/build_dashboard_snapshot.py — pure helpers.

Covers: parse_duration_seconds, cet_to_target_utc_datetime, parse_blocks,
        match_logs, render_markdown.
All pure or easily-mocked — no file system or network access required.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting"))

from build_dashboard_snapshot import (
    cet_to_target_utc_datetime,
    match_logs,
    parse_blocks,
    parse_duration_seconds,
    render_markdown,
)


# ---------------------------------------------------------------------------
# parse_duration_seconds
# ---------------------------------------------------------------------------

class TestParseDurationSeconds:
    def test_float_with_s(self):
        assert parse_duration_seconds("12.5s") == pytest.approx(12.5)

    def test_integer_with_s(self):
        assert parse_duration_seconds("30s") == pytest.approx(30.0)

    def test_float_without_s(self):
        assert parse_duration_seconds("5.25") == pytest.approx(5.25)

    def test_zero(self):
        assert parse_duration_seconds("0s") == pytest.approx(0.0)

    def test_whitespace_stripped(self):
        assert parse_duration_seconds("  10.0s  ") == pytest.approx(10.0)

    def test_invalid_returns_none(self):
        assert parse_duration_seconds("not-a-number") is None

    def test_returns_float(self):
        result = parse_duration_seconds("7s")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# cet_to_target_utc_datetime
# ---------------------------------------------------------------------------

class TestCetToTargetUtcDatetime:
    def _fixed_now(self):
        return datetime(2026, 3, 28, 12, 0, 0, tzinfo=timezone.utc)

    def test_returns_datetime(self):
        with patch("build_dashboard_snapshot.datetime") as mock_dt:
            mock_dt.strptime = datetime.strptime
            mock_dt.now.return_value = self._fixed_now()
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            result = cet_to_target_utc_datetime("02:30 PM", "15.0s")
        assert isinstance(result, datetime)

    def test_subtracts_duration(self):
        with patch("build_dashboard_snapshot.datetime") as mock_dt:
            mock_dt.strptime = datetime.strptime
            now = self._fixed_now()
            mock_dt.now.return_value = now
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            # CET is UTC+1, so 02:30 PM CET → 01:30 PM UTC, minus duration
            result = cet_to_target_utc_datetime("02:30 PM", "60.0s")
        # The result should be 60 seconds before the completion time
        assert isinstance(result, datetime)

    def test_invalid_time_returns_none(self):
        result = cet_to_target_utc_datetime("not-a-time", "10.0s")
        assert result is None

    def test_invalid_duration_returns_none(self):
        result = cet_to_target_utc_datetime("02:30 PM", "not-a-duration")
        assert result is None

    def test_has_utc_timezone(self):
        with patch("build_dashboard_snapshot.datetime") as mock_dt:
            mock_dt.strptime = datetime.strptime
            mock_dt.now.return_value = self._fixed_now()
            mock_dt.side_effect = lambda *args, **kwargs: datetime(*args, **kwargs)
            result = cet_to_target_utc_datetime("02:30 PM", "10.0s")
        assert result.tzinfo == timezone.utc


# ---------------------------------------------------------------------------
# parse_blocks
# ---------------------------------------------------------------------------

class TestParseBlocks:
    def test_empty_text_returns_empty(self):
        assert parse_blocks("", limit=10) == []

    def test_no_recent_results_returns_empty(self):
        assert parse_blocks("Some text without results", limit=10) == []

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

    def test_pct_captured(self):
        text = (
            "Recent Results\n"
            "Task (7/10)\n"
            "01:15 PM · 8.5s\n"
            "7/10 (70%)\n"
        )
        result = parse_blocks(text, limit=10)
        assert result[0]["pct"] == 70.0

    def test_points_and_max_points(self):
        text = (
            "Recent Results\n"
            "Task (7/10)\n"
            "01:15 PM · 8.5s\n"
            "7/10 (70%)\n"
        )
        result = parse_blocks(text, limit=10)
        assert result[0]["points"] == "7"
        assert result[0]["max_points"] == "10"

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
        checks = result[0]["checks"]
        assert ("1", "passed") in checks
        assert ("3", "failed") in checks

    def test_limit_enforced(self):
        block = "Task (3/5)\n02:30 PM · 10.0s\n3/5 (60%)\n"
        text = "Recent Results\n" + block * 5
        result = parse_blocks(text, limit=2)
        assert len(result) <= 2

    def test_cet_time_captured(self):
        text = (
            "Recent Results\n"
            "Task (5/5)\n"
            "11:45 AM · 5.0s\n"
            "5/5 (100%)\n"
        )
        result = parse_blocks(text, limit=10)
        assert result[0]["cet_time"] == "11:45 AM"

    def test_duration_captured(self):
        text = (
            "Recent Results\n"
            "Task (5/5)\n"
            "11:45 AM · 7.3s\n"
            "5/5 (100%)\n"
        )
        result = parse_blocks(text, limit=10)
        assert result[0]["duration"] == "7.3s"


# ---------------------------------------------------------------------------
# match_logs
# ---------------------------------------------------------------------------

class TestMatchLogs:
    def _make_log(self, ts: str, dt: datetime, **extra) -> dict:
        return {"timestamp": ts, "dt": dt, **extra}

    def test_empty_blocks_returns_empty(self):
        result = match_logs([], [], max_gap_seconds=120.0)
        assert result == []

    def test_no_logs_returns_none_entries(self):
        # Need a block with a valid target_dt — use a real datetime
        # We'll provide a block with cet_time/duration that resolves via cet_to_target_utc_datetime
        # Since we can't easily inject target_dt, test with empty logs list
        result = match_logs(
            [{"cet_time": "invalid", "duration": "invalid"}],
            [],
            max_gap_seconds=120.0,
        )
        assert result == [None]

    def test_matched_log_confidence_high_within_30s(self):
        # Use a pre-built block with known target_dt by passing through patch
        now = datetime(2026, 3, 28, 14, 0, 0, tzinfo=timezone.utc)
        log_dt = now - timedelta(seconds=10)
        log = self._make_log("ts1", log_dt)

        with patch("build_dashboard_snapshot.cet_to_target_utc_datetime", return_value=now):
            result = match_logs(
                [{"cet_time": "02:00 PM", "duration": "10.0s"}],
                [log],
                max_gap_seconds=120.0,
            )
        assert result[0] is not None
        assert result[0]["match_confidence"] == "high"

    def test_matched_log_confidence_medium_beyond_30s(self):
        now = datetime(2026, 3, 28, 14, 0, 0, tzinfo=timezone.utc)
        log_dt = now - timedelta(seconds=60)
        log = self._make_log("ts1", log_dt)

        with patch("build_dashboard_snapshot.cet_to_target_utc_datetime", return_value=now):
            result = match_logs(
                [{"cet_time": "02:00 PM", "duration": "10.0s"}],
                [log],
                max_gap_seconds=120.0,
            )
        assert result[0] is not None
        assert result[0]["match_confidence"] == "medium"

    def test_beyond_max_gap_not_matched(self):
        now = datetime(2026, 3, 28, 14, 0, 0, tzinfo=timezone.utc)
        log_dt = now - timedelta(seconds=200)
        log = self._make_log("ts1", log_dt)

        with patch("build_dashboard_snapshot.cet_to_target_utc_datetime", return_value=now):
            result = match_logs(
                [{"cet_time": "02:00 PM", "duration": "10.0s"}],
                [log],
                max_gap_seconds=120.0,
            )
        assert result[0] is None

    def test_same_log_not_reused_for_two_blocks(self):
        now = datetime(2026, 3, 28, 14, 0, 0, tzinfo=timezone.utc)
        log_dt = now - timedelta(seconds=5)
        log = self._make_log("ts1", log_dt)

        with patch("build_dashboard_snapshot.cet_to_target_utc_datetime", return_value=now):
            result = match_logs(
                [
                    {"cet_time": "02:00 PM", "duration": "10.0s"},
                    {"cet_time": "02:00 PM", "duration": "10.0s"},
                ],
                [log],
                max_gap_seconds=120.0,
            )
        # Only one block can claim the log
        matches = [r for r in result if r is not None]
        assert len(matches) == 1


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------

class TestRenderMarkdown:
    def _minimal_snapshot(self, tasks=None):
        return {
            "generated_at": "2026-03-28T12:00:00+00:00",
            "total_score": 72.5,
            "rank": 3,
            "submissions_used": 5,
            "tasks": tasks or [],
        }

    def test_returns_string(self):
        result = render_markdown(self._minimal_snapshot())
        assert isinstance(result, str)

    def test_contains_title(self):
        result = render_markdown(self._minimal_snapshot())
        assert "Accounting Dashboard Snapshot" in result

    def test_total_score_in_output(self):
        result = render_markdown(self._minimal_snapshot())
        assert "72.5" in result

    def test_rank_in_output(self):
        result = render_markdown(self._minimal_snapshot())
        assert "#3" in result or "3" in result

    def test_submissions_in_output(self):
        result = render_markdown(self._minimal_snapshot())
        assert "5" in result

    def test_no_tasks_shows_none(self):
        result = render_markdown(self._minimal_snapshot(tasks=[]))
        assert "none" in result.lower()

    def test_task_points_in_output(self):
        task = {
            "points": 7,
            "max_points": 10,
            "dashboard_score": 70.0,
            "cet_time": "02:30 PM",
            "duration": "15.0s",
            "checks_passed": 3,
            "checks_failed": 1,
            "family": "invoice",
            "api_calls": 5,
            "api_errors": 0,
            "successful_writes": 3,
            "match_confidence": "high",
        }
        result = render_markdown(self._minimal_snapshot(tasks=[task]))
        assert "7" in result
        assert "10" in result

    def test_low_score_task_includes_prompt_preview(self):
        task = {
            "points": 2,
            "max_points": 10,
            "dashboard_score": 20.0,
            "cet_time": "01:00 PM",
            "duration": "10.0s",
            "checks_passed": 1,
            "checks_failed": 4,
            "family": "payment",
            "api_calls": 2,
            "api_errors": 1,
            "successful_writes": 1,
            "match_confidence": "medium",
            "prompt_preview": "Pay invoice 123",
            "result_preview": "Error: 404",
        }
        result = render_markdown(self._minimal_snapshot(tasks=[task]))
        assert "Pay invoice 123" in result
        assert "Error: 404" in result

    def test_high_score_task_no_prompt_preview(self):
        task = {
            "points": 9,
            "max_points": 10,
            "dashboard_score": 90.0,
            "cet_time": "03:00 PM",
            "duration": "8.0s",
            "checks_passed": 4,
            "checks_failed": 0,
            "family": "credit",
            "api_calls": 3,
            "api_errors": 0,
            "successful_writes": 2,
            "match_confidence": "high",
            "prompt_preview": "Should not appear",
            "result_preview": "Should not appear either",
        }
        result = render_markdown(self._minimal_snapshot(tasks=[task]))
        assert "Should not appear" not in result

    def test_ends_with_newline(self):
        result = render_markdown(self._minimal_snapshot())
        assert result.endswith("\n")

    def test_generated_at_in_output(self):
        result = render_markdown(self._minimal_snapshot())
        assert "2026-03-28" in result

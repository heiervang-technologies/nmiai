"""Tests for tasks/accounting/project_progress.py — pure helper functions.

Covers: to_float, parse_time, fit_line, clamp, build_paired_points,
        latest_rows_by_family, best_dashboard_by_family.

All pure functions — no file system or matplotlib access needed.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting"))

from project_progress import (
    best_dashboard_by_family,
    build_paired_points,
    clamp,
    fit_line,
    latest_rows_by_family,
    parse_time,
    to_float,
)


# ---------------------------------------------------------------------------
# to_float
# ---------------------------------------------------------------------------

class TestToFloat:
    def test_none_returns_none(self):
        assert to_float(None) is None

    def test_empty_string_returns_none(self):
        assert to_float("") is None

    def test_whitespace_only_returns_none(self):
        assert to_float("   ") is None

    def test_valid_integer_string(self):
        assert to_float("42") == pytest.approx(42.0)

    def test_valid_float_string(self):
        assert to_float("3.14") == pytest.approx(3.14)

    def test_invalid_string_returns_none(self):
        assert to_float("not-a-number") is None

    def test_returns_float_type(self):
        result = to_float("5")
        assert isinstance(result, float)

    def test_negative_number(self):
        assert to_float("-7.5") == pytest.approx(-7.5)

    def test_zero_string(self):
        assert to_float("0") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# parse_time
# ---------------------------------------------------------------------------

class TestParseTime:
    def test_valid_iso_format(self):
        result = parse_time("2026-03-15T10:00:00", fallback_index=0)
        assert isinstance(result, datetime)
        assert result.year == 2026

    def test_z_suffix_handled(self):
        result = parse_time("2026-03-15T10:00:00Z", fallback_index=0)
        assert result.tzinfo is not None

    def test_none_returns_fallback_datetime(self):
        result = parse_time(None, fallback_index=0)
        assert isinstance(result, datetime)

    def test_invalid_string_returns_fallback(self):
        result = parse_time("not-a-date", fallback_index=5)
        assert isinstance(result, datetime)

    def test_fallback_index_affects_fallback_time(self):
        t0 = parse_time(None, fallback_index=0)
        t1 = parse_time(None, fallback_index=100)
        assert t1 > t0

    def test_returns_datetime(self):
        assert isinstance(parse_time("2026-01-01T00:00:00", 0), datetime)

    def test_empty_string_returns_fallback(self):
        result = parse_time("", fallback_index=0)
        assert isinstance(result, datetime)


# ---------------------------------------------------------------------------
# fit_line
# ---------------------------------------------------------------------------

class TestFitLine:
    def test_returns_tuple_of_three(self):
        result = fit_line([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_perfect_linear_relationship(self):
        # y = 2x + 1
        xs = [1.0, 2.0, 3.0, 4.0, 5.0]
        ys = [3.0, 5.0, 7.0, 9.0, 11.0]
        slope, intercept, rmse = fit_line(xs, ys)
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(1.0)
        assert rmse == pytest.approx(0.0, abs=1e-10)

    def test_slope_is_float(self):
        slope, intercept, rmse = fit_line([1.0, 2.0], [1.0, 2.0])
        assert isinstance(slope, float)
        assert isinstance(intercept, float)
        assert isinstance(rmse, float)

    def test_horizontal_line_slope_zero(self):
        xs = [1.0, 2.0, 3.0]
        ys = [5.0, 5.0, 5.0]
        slope, intercept, rmse = fit_line(xs, ys)
        assert slope == pytest.approx(0.0)
        assert intercept == pytest.approx(5.0)

    def test_rmse_nonnegative(self):
        _, _, rmse = fit_line([1.0, 2.0, 3.0], [1.0, 2.5, 3.0])
        assert rmse >= 0.0

    def test_all_same_x_returns_zero_slope(self):
        # var_x == 0 → slope = 0
        xs = [1.0, 1.0, 1.0]
        ys = [1.0, 2.0, 3.0]
        slope, _, _ = fit_line(xs, ys)
        assert slope == 0.0


# ---------------------------------------------------------------------------
# clamp
# ---------------------------------------------------------------------------

class TestClamp:
    def test_value_below_low_returns_low(self):
        assert clamp(-5.0, 0.0, 1.0) == 0.0

    def test_value_above_high_returns_high(self):
        assert clamp(10.0, 0.0, 1.0) == 1.0

    def test_value_in_range_unchanged(self):
        assert clamp(0.5, 0.0, 1.0) == 0.5

    def test_value_equal_to_low(self):
        assert clamp(0.0, 0.0, 1.0) == 0.0

    def test_value_equal_to_high(self):
        assert clamp(1.0, 0.0, 1.0) == 1.0

    def test_returns_float(self):
        assert isinstance(clamp(0.5, 0.0, 1.0), float)


# ---------------------------------------------------------------------------
# build_paired_points
# ---------------------------------------------------------------------------

class TestBuildPairedPoints:
    def _result_row(self, batch_id, family, proxy_clean_rate, **extra):
        row = {
            "batch_id": batch_id,
            "family": family,
            "proxy_clean_rate": str(proxy_clean_rate),
        }
        row.update({k: str(v) for k, v in extra.items()})
        return row

    def _dash_row(self, log_ts, family, dashboard_score):
        return {
            "log_ts": log_ts,
            "family": family,
            "dashboard_score": str(dashboard_score),
        }

    def test_empty_inputs_returns_empty(self):
        assert build_paired_points([], []) == []

    def test_matching_rows_paired(self):
        results = [self._result_row("ts1", "invoice", 0.8)]
        dashboard = [self._dash_row("ts1", "invoice", 0.75)]
        paired = build_paired_points(results, dashboard)
        assert len(paired) == 1
        assert paired[0]["proxy_clean_rate"] == pytest.approx(0.8)
        assert paired[0]["dashboard_score"] == pytest.approx(0.75)

    def test_no_matching_key_returns_empty(self):
        results = [self._result_row("ts1", "invoice", 0.8)]
        dashboard = [self._dash_row("ts2", "invoice", 0.75)]  # different log_ts
        paired = build_paired_points(results, dashboard)
        assert paired == []

    def test_missing_proxy_clean_rate_skipped(self):
        results = [{"batch_id": "ts1", "family": "invoice"}]  # no proxy_clean_rate
        dashboard = [self._dash_row("ts1", "invoice", 0.75)]
        paired = build_paired_points(results, dashboard)
        assert paired == []

    def test_family_field_in_result(self):
        results = [self._result_row("ts1", "employee", 0.9)]
        dashboard = [self._dash_row("ts1", "employee", 0.85)]
        paired = build_paired_points(results, dashboard)
        assert paired[0]["family"] == "employee"

    def test_attachment_present_parsed_as_bool(self):
        results = [self._result_row("ts1", "invoice", 0.8, attachment_present="True")]
        dashboard = [self._dash_row("ts1", "invoice", 0.7)]
        paired = build_paired_points(results, dashboard)
        assert paired[0]["attachment_present"] is True

    def test_attachment_absent_is_false(self):
        results = [self._result_row("ts1", "invoice", 0.8, attachment_present="false")]
        dashboard = [self._dash_row("ts1", "invoice", 0.7)]
        paired = build_paired_points(results, dashboard)
        assert paired[0]["attachment_present"] is False


# ---------------------------------------------------------------------------
# latest_rows_by_family
# ---------------------------------------------------------------------------

class TestLatestRowsByFamily:
    def test_empty_list_returns_empty_dict(self):
        assert latest_rows_by_family([]) == {}

    def test_single_row_returned(self):
        rows = [{"family": "invoice", "timestamp": "2026-01-01T00:00:00", "score": "0.8"}]
        result = latest_rows_by_family(rows)
        assert "invoice" in result
        assert result["invoice"]["score"] == "0.8"

    def test_returns_latest_row_by_timestamp(self):
        rows = [
            {"family": "invoice", "timestamp": "2026-01-01T00:00:00", "score": "0.5"},
            {"family": "invoice", "timestamp": "2026-03-01T00:00:00", "score": "0.9"},
        ]
        result = latest_rows_by_family(rows)
        assert result["invoice"]["score"] == "0.9"

    def test_multiple_families_separate(self):
        rows = [
            {"family": "invoice", "timestamp": "2026-01-01T00:00:00"},
            {"family": "employee", "timestamp": "2026-01-02T00:00:00"},
        ]
        result = latest_rows_by_family(rows)
        assert "invoice" in result
        assert "employee" in result

    def test_returns_dict(self):
        result = latest_rows_by_family([])
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# best_dashboard_by_family
# ---------------------------------------------------------------------------

class TestBestDashboardByFamily:
    def test_empty_returns_empty_dict(self):
        assert best_dashboard_by_family([]) == {}

    def test_single_row(self):
        rows = [{"family": "invoice", "dashboard_score": "0.8"}]
        result = best_dashboard_by_family(rows)
        assert result["invoice"] == pytest.approx(0.8)

    def test_keeps_best_score(self):
        rows = [
            {"family": "invoice", "dashboard_score": "0.5"},
            {"family": "invoice", "dashboard_score": "0.9"},
            {"family": "invoice", "dashboard_score": "0.7"},
        ]
        result = best_dashboard_by_family(rows)
        assert result["invoice"] == pytest.approx(0.9)

    def test_missing_score_skipped(self):
        rows = [
            {"family": "invoice"},  # no dashboard_score
            {"family": "invoice", "dashboard_score": "0.8"},
        ]
        result = best_dashboard_by_family(rows)
        assert result["invoice"] == pytest.approx(0.8)

    def test_multiple_families(self):
        rows = [
            {"family": "invoice", "dashboard_score": "0.8"},
            {"family": "employee", "dashboard_score": "0.6"},
        ]
        result = best_dashboard_by_family(rows)
        assert "invoice" in result
        assert "employee" in result

    def test_returns_dict(self):
        assert isinstance(best_dashboard_by_family([]), dict)

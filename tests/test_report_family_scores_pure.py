"""Tests for tasks/accounting/report_family_scores.py — pure helper functions.

Covers: to_float, best_dashboard_by_family, dashboard_stats_by_family,
        projection_by_family, priority_by_family, score_estimate,
        stabilized_estimate, opportunity_score.

All pure functions — no file system or network access required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "accounting"))

from report_family_scores import (
    best_dashboard_by_family,
    dashboard_stats_by_family,
    opportunity_score,
    priority_by_family,
    projection_by_family,
    score_estimate,
    stabilized_estimate,
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

    def test_whitespace_returns_none(self):
        assert to_float("   ") is None

    def test_valid_float_string(self):
        assert to_float("3.14") == pytest.approx(3.14)

    def test_valid_int_string(self):
        assert to_float("42") == pytest.approx(42.0)

    def test_invalid_returns_none(self):
        assert to_float("not-a-number") is None

    def test_float_input(self):
        assert to_float(0.75) == pytest.approx(0.75)

    def test_returns_float_type(self):
        result = to_float("5")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# best_dashboard_by_family
# ---------------------------------------------------------------------------

class TestBestDashboardByFamily:
    def test_empty_returns_empty(self):
        assert best_dashboard_by_family([]) == {}

    def test_single_row_captured(self):
        rows = [{"family": "invoice", "dashboard_score": "0.8", "log_ts": "ts1"}]
        result = best_dashboard_by_family(rows)
        assert "invoice" in result
        assert result["invoice"]["score"] == pytest.approx(0.8)

    def test_keeps_best_score(self):
        rows = [
            {"family": "invoice", "dashboard_score": "0.5"},
            {"family": "invoice", "dashboard_score": "0.9"},
        ]
        result = best_dashboard_by_family(rows)
        assert result["invoice"]["score"] == pytest.approx(0.9)

    def test_missing_score_skipped(self):
        rows = [{"family": "invoice"}]
        result = best_dashboard_by_family(rows)
        assert result == {}

    def test_multiple_families(self):
        rows = [
            {"family": "invoice", "dashboard_score": "0.8"},
            {"family": "employee", "dashboard_score": "0.6"},
        ]
        result = best_dashboard_by_family(rows)
        assert "invoice" in result
        assert "employee" in result


# ---------------------------------------------------------------------------
# dashboard_stats_by_family
# ---------------------------------------------------------------------------

class TestDashboardStatsByFamily:
    def test_empty_returns_empty(self):
        assert dashboard_stats_by_family([]) == {}

    def test_single_row_stats(self):
        rows = [{"family": "invoice", "dashboard_score": "0.8"}]
        result = dashboard_stats_by_family(rows)
        assert result["invoice"]["count"] == pytest.approx(1.0)
        assert result["invoice"]["best"] == pytest.approx(0.8)

    def test_multiple_rows_avg(self):
        rows = [
            {"family": "invoice", "dashboard_score": "0.6"},
            {"family": "invoice", "dashboard_score": "0.8"},
        ]
        result = dashboard_stats_by_family(rows)
        assert result["invoice"]["avg"] == pytest.approx(0.7)

    def test_best_is_max(self):
        rows = [
            {"family": "invoice", "dashboard_score": "0.5"},
            {"family": "invoice", "dashboard_score": "0.9"},
            {"family": "invoice", "dashboard_score": "0.7"},
        ]
        result = dashboard_stats_by_family(rows)
        assert result["invoice"]["best"] == pytest.approx(0.9)

    def test_missing_score_skipped(self):
        rows = [{"family": "invoice"}]
        result = dashboard_stats_by_family(rows)
        assert result == {}


# ---------------------------------------------------------------------------
# projection_by_family
# ---------------------------------------------------------------------------

class TestProjectionByFamily:
    def test_empty_payload_returns_empty(self):
        assert projection_by_family({}) == {}

    def test_extracts_family_rows(self):
        payload = {"family_projections": [{"family": "invoice", "projected": 0.75}]}
        result = projection_by_family(payload)
        assert "invoice" in result
        assert result["invoice"]["projected"] == 0.75

    def test_skips_rows_without_family(self):
        payload = {"family_projections": [{"projected": 0.5}]}
        result = projection_by_family(payload)
        assert result == {}

    def test_none_projections_returns_empty(self):
        payload = {"family_projections": None}
        result = projection_by_family(payload)
        assert result == {}


# ---------------------------------------------------------------------------
# priority_by_family
# ---------------------------------------------------------------------------

class TestPriorityByFamily:
    def test_empty_payload_returns_empty(self):
        assert priority_by_family({}) == {}

    def test_extracts_priority_targets(self):
        payload = {"priority_targets": [{"family": "employee", "score": 5.0}]}
        result = priority_by_family(payload)
        assert "employee" in result

    def test_skips_rows_without_family(self):
        payload = {"priority_targets": [{"score": 3.0}]}
        result = priority_by_family(payload)
        assert result == {}


# ---------------------------------------------------------------------------
# score_estimate
# ---------------------------------------------------------------------------

class TestScoreEstimate:
    def test_observed_returns_observed(self):
        value, source = score_estimate(0.8, 0.5)
        assert value == pytest.approx(0.8)
        assert source == "observed"

    def test_no_observed_uses_projected(self):
        value, source = score_estimate(None, 0.6)
        assert value == pytest.approx(0.6)
        assert source == "projected"

    def test_both_none_returns_none(self):
        value, source = score_estimate(None, None)
        assert value is None
        assert source == "none"

    def test_returns_tuple_of_two(self):
        result = score_estimate(0.5, 0.4)
        assert isinstance(result, tuple)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# stabilized_estimate
# ---------------------------------------------------------------------------

class TestStabilizedEstimate:
    def test_no_observed_returns_projected(self):
        result = stabilized_estimate([], 0.7)
        assert result == pytest.approx(0.7)

    def test_no_observed_no_projected_returns_none(self):
        result = stabilized_estimate([], None)
        assert result is None

    def test_single_observed_no_projected_returns_observed(self):
        result = stabilized_estimate([0.8], None)
        assert result == pytest.approx(0.8)

    def test_weighted_average(self):
        # (0.8 + 2.0 * 0.6) / (1 + 2.0) = (0.8 + 1.2) / 3 = 2.0/3 ≈ 0.667
        result = stabilized_estimate([0.8], 0.6, prior_weight=2.0)
        assert result == pytest.approx(2.0 / 3.0)

    def test_multiple_observations(self):
        result = stabilized_estimate([0.6, 0.8], 0.5, prior_weight=1.0)
        # (0.6 + 0.8 + 1.0 * 0.5) / (2 + 1) = 1.9 / 3
        assert result == pytest.approx(1.9 / 3.0)


# ---------------------------------------------------------------------------
# opportunity_score
# ---------------------------------------------------------------------------

class TestOpportunityScore:
    def test_none_priority_returns_none(self):
        assert opportunity_score(None, 30.0) is None

    def test_none_gap_returns_priority_score(self):
        result = opportunity_score(5.0, None)
        assert result == pytest.approx(5.0)

    def test_scales_by_gap(self):
        result = opportunity_score(4.0, 50.0)
        assert result == pytest.approx(4.0 * 50.0 / 100.0)

    def test_zero_gap_gives_zero(self):
        result = opportunity_score(5.0, 0.0)
        assert result == pytest.approx(0.0)

    def test_full_gap_preserves_priority(self):
        result = opportunity_score(3.0, 100.0)
        assert result == pytest.approx(3.0)

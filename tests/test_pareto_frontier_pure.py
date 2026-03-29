"""Pure-function tests for tools/pareto_frontier.py.

Covers: to_number, better, no_worse, dominates, build_frontier, build_progress

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_TOOLS_DIR = str(Path(__file__).resolve().parent.parent / "tools")
sys.path.insert(0, _TOOLS_DIR)

from pareto_frontier import (
    to_number,
    better,
    no_worse,
    dominates,
    build_frontier,
    build_progress,
)


# ---------------------------------------------------------------------------
# to_number
# ---------------------------------------------------------------------------

class TestToNumber:
    def test_plain_float(self):
        assert to_number("3.14") == pytest.approx(3.14)

    def test_integer_string(self):
        assert to_number("42") == pytest.approx(42.0)

    def test_negative(self):
        assert to_number("-1.5") == pytest.approx(-1.5)

    def test_whitespace_stripped(self):
        assert to_number("  0.9  ") == pytest.approx(0.9)

    def test_empty_string_returns_none(self):
        assert to_number("") is None

    def test_whitespace_only_returns_none(self):
        assert to_number("   ") is None

    def test_none_returns_none(self):
        assert to_number(None) is None

    def test_non_numeric_returns_none(self):
        assert to_number("abc") is None


# ---------------------------------------------------------------------------
# better
# ---------------------------------------------------------------------------

class TestBetter:
    def test_min_lower_is_better(self):
        assert better(1.0, 2.0, "min") is True

    def test_min_higher_is_not_better(self):
        assert better(2.0, 1.0, "min") is False

    def test_max_higher_is_better(self):
        assert better(2.0, 1.0, "max") is True

    def test_max_lower_is_not_better(self):
        assert better(1.0, 2.0, "max") is False

    def test_equal_values_not_better_min(self):
        assert better(1.0, 1.0, "min") is False

    def test_equal_values_not_better_max(self):
        assert better(1.0, 1.0, "max") is False


# ---------------------------------------------------------------------------
# no_worse
# ---------------------------------------------------------------------------

class TestNoWorse:
    def test_min_equal_is_no_worse(self):
        assert no_worse(1.0, 1.0, "min") is True

    def test_min_lower_is_no_worse(self):
        assert no_worse(0.5, 1.0, "min") is True

    def test_min_higher_is_worse(self):
        assert no_worse(2.0, 1.0, "min") is False

    def test_max_equal_is_no_worse(self):
        assert no_worse(1.0, 1.0, "max") is True

    def test_max_higher_is_no_worse(self):
        assert no_worse(2.0, 1.0, "max") is True

    def test_max_lower_is_worse(self):
        assert no_worse(0.5, 1.0, "max") is False


# ---------------------------------------------------------------------------
# dominates
# ---------------------------------------------------------------------------

class TestDominates:
    def _point(self, x, y, label=""):
        return {"x": x, "y": y, "label": label}

    def test_clearly_dominates_min_min(self):
        # a=(1,1) dominates b=(2,2) when minimizing both
        a = self._point(1.0, 1.0)
        b = self._point(2.0, 2.0)
        assert dominates(a, b, "min", "min") is True

    def test_no_domination_min_min_equal(self):
        a = self._point(1.0, 1.0)
        assert dominates(a, a, "min", "min") is False

    def test_no_domination_when_one_axis_worse(self):
        # a is better on x but worse on y
        a = self._point(0.5, 3.0)
        b = self._point(1.0, 1.0)
        assert dominates(a, b, "min", "min") is False

    def test_dominates_max_max(self):
        a = self._point(5.0, 5.0)
        b = self._point(3.0, 3.0)
        assert dominates(a, b, "max", "max") is True

    def test_none_x_returns_false(self):
        a = {"x": None, "y": 1.0}
        b = {"x": 1.0, "y": 1.0}
        assert dominates(a, b, "min", "min") is False

    def test_none_y_returns_false(self):
        a = {"x": 1.0, "y": None}
        b = {"x": 2.0, "y": 2.0}
        assert dominates(a, b, "min", "min") is False


# ---------------------------------------------------------------------------
# build_frontier
# ---------------------------------------------------------------------------

class TestBuildFrontier:
    def _point(self, x, y, label=""):
        return {"x": x, "y": y, "label": label}

    def test_single_point_is_frontier(self):
        pts = [self._point(1.0, 1.0)]
        frontier = build_frontier(pts, "min", "min")
        assert len(frontier) == 1

    def test_dominated_point_excluded(self):
        # a dominates b; only a should be on frontier
        pts = [self._point(0.5, 0.5, "a"), self._point(1.0, 1.0, "b")]
        frontier = build_frontier(pts, "min", "min")
        labels = {p["label"] for p in frontier}
        assert "a" in labels
        assert "b" not in labels

    def test_pareto_optimal_points_all_included(self):
        # Two points, each better on one axis: both on frontier
        pts = [self._point(0.5, 2.0, "a"), self._point(2.0, 0.5, "b")]
        frontier = build_frontier(pts, "min", "min")
        assert len(frontier) == 2

    def test_sorted_by_x_min(self):
        pts = [self._point(3.0, 1.0), self._point(1.0, 3.0), self._point(2.0, 2.0)]
        frontier = build_frontier(pts, "min", "min")
        xs = [p["x"] for p in frontier]
        assert xs == sorted(xs)  # ascending for min

    def test_none_coordinates_excluded(self):
        pts = [self._point(None, 1.0), self._point(1.0, None), self._point(0.5, 0.5, "ok")]
        frontier = build_frontier(pts, "min", "min")
        assert len(frontier) == 1
        assert frontier[0]["label"] == "ok"

    def test_empty_input(self):
        assert build_frontier([], "min", "min") == []


# ---------------------------------------------------------------------------
# build_progress
# ---------------------------------------------------------------------------

class TestBuildProgress:
    def _point(self, time, y, label=""):
        return {"x": 0.0, "y": y, "time": time, "label": label}

    def test_returns_same_length(self):
        pts = [self._point(i, float(i)) for i in range(5)]
        progress = build_progress(pts, "min")
        assert len(progress) == 5

    def test_best_y_tracks_minimum(self):
        pts = [self._point(0, 3.0), self._point(1, 1.0), self._point(2, 2.0)]
        progress = build_progress(pts, "min")
        assert progress[0]["best_y"] == pytest.approx(3.0)
        assert progress[1]["best_y"] == pytest.approx(1.0)
        assert progress[2]["best_y"] == pytest.approx(1.0)  # stays at 1.0

    def test_best_y_tracks_maximum(self):
        pts = [self._point(0, 1.0), self._point(1, 5.0), self._point(2, 3.0)]
        progress = build_progress(pts, "max")
        assert progress[2]["best_y"] == pytest.approx(5.0)

    def test_none_y_skipped(self):
        pts = [self._point(0, None), self._point(1, 2.0)]
        progress = build_progress(pts, "min")
        assert len(progress) == 1  # None-y point skipped

    def test_result_has_required_keys(self):
        pts = [self._point(0, 1.0, "m1")]
        result = build_progress(pts, "min")
        for key in ("time", "label", "current_y", "best_label", "best_y"):
            assert key in result[0]

    def test_current_y_equals_point_y(self):
        pts = [self._point(0, 4.0, "x")]
        result = build_progress(pts, "min")
        assert result[0]["current_y"] == pytest.approx(4.0)

    def test_empty_input(self):
        assert build_progress([], "min") == []

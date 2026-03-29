"""Tests for tools/pareto_frontier.py — pure helper functions."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

from pareto_frontier import (
    to_number,
    better,
    no_worse,
    dominates,
    build_frontier,
    build_progress,
    load_rows,
)


# ---------------------------------------------------------------------------
# to_number
# ---------------------------------------------------------------------------

class TestToNumber:
    def test_integer_string(self):
        assert to_number("42") == 42.0

    def test_float_string(self):
        assert abs(to_number("3.14") - 3.14) < 1e-10

    def test_negative(self):
        assert to_number("-7.5") == -7.5

    def test_strips_whitespace(self):
        assert to_number("  10  ") == 10.0

    def test_empty_string_returns_none(self):
        assert to_number("") is None

    def test_whitespace_only_returns_none(self):
        assert to_number("   ") is None

    def test_none_returns_none(self):
        assert to_number(None) is None

    def test_non_numeric_returns_none(self):
        assert to_number("abc") is None

    def test_zero_string(self):
        assert to_number("0") == 0.0


# ---------------------------------------------------------------------------
# better / no_worse
# ---------------------------------------------------------------------------

class TestBetter:
    def test_min_direction_lower_is_better(self):
        assert better(1.0, 2.0, "min") is True
        assert better(2.0, 1.0, "min") is False

    def test_max_direction_higher_is_better(self):
        assert better(3.0, 2.0, "max") is True
        assert better(2.0, 3.0, "max") is False

    def test_equal_is_not_better(self):
        assert better(1.0, 1.0, "min") is False
        assert better(1.0, 1.0, "max") is False


class TestNoWorse:
    def test_min_direction_lower_or_equal_ok(self):
        assert no_worse(1.0, 2.0, "min") is True
        assert no_worse(2.0, 2.0, "min") is True
        assert no_worse(3.0, 2.0, "min") is False

    def test_max_direction_higher_or_equal_ok(self):
        assert no_worse(3.0, 2.0, "max") is True
        assert no_worse(2.0, 2.0, "max") is True
        assert no_worse(1.0, 2.0, "max") is False


# ---------------------------------------------------------------------------
# dominates
# ---------------------------------------------------------------------------

def _pt(x, y, label="p"):
    return {"x": x, "y": y, "label": label}


class TestDominates:
    def test_strictly_better_on_both_axes(self):
        a = _pt(1.0, 5.0)
        b = _pt(2.0, 3.0)
        # min-x, max-y: a has lower x AND higher y → a dominates b
        assert dominates(a, b, "min", "max") is True
        assert dominates(b, a, "min", "max") is False

    def test_better_on_one_not_worse_on_other(self):
        a = _pt(1.0, 3.0)
        b = _pt(2.0, 3.0)
        # a is better on x (lower), equal on y → a dominates b (min-x, min-y)
        assert dominates(a, b, "min", "min") is True

    def test_equal_on_both_does_not_dominate(self):
        a = _pt(2.0, 3.0)
        b = _pt(2.0, 3.0)
        assert dominates(a, b, "min", "min") is False

    def test_none_x_does_not_dominate(self):
        a = _pt(None, 3.0)
        b = _pt(2.0, 3.0)
        assert dominates(a, b, "min", "min") is False

    def test_none_y_does_not_dominate(self):
        a = _pt(1.0, None)
        b = _pt(2.0, 3.0)
        assert dominates(a, b, "min", "min") is False

    def test_max_x_max_y(self):
        a = _pt(5.0, 5.0)
        b = _pt(3.0, 3.0)
        assert dominates(a, b, "max", "max") is True
        assert dominates(b, a, "max", "max") is False


# ---------------------------------------------------------------------------
# build_frontier
# ---------------------------------------------------------------------------

class TestBuildFrontier:
    def test_single_point_is_frontier(self):
        pts = [{"x": 1.0, "y": 2.0, "label": "a", "time": "0"}]
        result = build_frontier(pts, "min", "max")
        assert len(result) == 1

    def test_dominated_point_excluded(self):
        pts = [
            {"x": 1.0, "y": 5.0, "label": "a", "time": "0"},
            {"x": 2.0, "y": 3.0, "label": "b", "time": "1"},
        ]
        # min-x, max-y: 'a' dominates 'b'
        result = build_frontier(pts, "min", "max")
        labels = [p["label"] for p in result]
        assert "a" in labels
        assert "b" not in labels

    def test_pareto_front_two_points(self):
        """Two non-dominated points both appear on the frontier."""
        pts = [
            {"x": 1.0, "y": 3.0, "label": "a", "time": "0"},
            {"x": 2.0, "y": 5.0, "label": "b", "time": "1"},
        ]
        # min-x, max-y: a better on x, b better on y → neither dominates
        result = build_frontier(pts, "min", "max")
        assert len(result) == 2

    def test_none_values_excluded(self):
        pts = [
            {"x": None, "y": 3.0, "label": "bad", "time": "0"},
            {"x": 1.0, "y": 3.0, "label": "good", "time": "1"},
        ]
        result = build_frontier(pts, "min", "min")
        labels = [p["label"] for p in result]
        assert "bad" not in labels

    def test_sorted_by_x_min(self):
        pts = [
            {"x": 3.0, "y": 1.0, "label": "c", "time": "0"},
            {"x": 1.0, "y": 3.0, "label": "a", "time": "1"},
            {"x": 2.0, "y": 2.0, "label": "b", "time": "2"},
        ]
        result = build_frontier(pts, "min", "min")
        xs = [p["x"] for p in result]
        assert xs == sorted(xs)

    def test_empty_returns_empty(self):
        assert build_frontier([], "min", "max") == []


# ---------------------------------------------------------------------------
# build_progress
# ---------------------------------------------------------------------------

class TestBuildProgress:
    def test_best_y_tracks_improvement_max(self):
        pts = [
            {"y": 1.0, "label": "a", "time": "0"},
            {"y": 3.0, "label": "b", "time": "1"},
            {"y": 2.0, "label": "c", "time": "2"},
        ]
        result = build_progress(pts, "max")
        assert result[0]["best_label"] == "a"
        assert result[1]["best_label"] == "b"
        assert result[2]["best_label"] == "b"  # b still best

    def test_best_y_tracks_improvement_min(self):
        pts = [
            {"y": 5.0, "label": "a", "time": "0"},
            {"y": 3.0, "label": "b", "time": "1"},
            {"y": 4.0, "label": "c", "time": "2"},
        ]
        result = build_progress(pts, "min")
        assert result[2]["best_label"] == "b"

    def test_none_y_skipped(self):
        pts = [
            {"y": None, "label": "skip", "time": "0"},
            {"y": 2.0, "label": "a", "time": "1"},
        ]
        result = build_progress(pts, "max")
        assert len(result) == 1
        assert result[0]["label"] == "a"

    def test_empty_returns_empty(self):
        assert build_progress([], "max") == []

    def test_progress_length_equals_valid_points(self):
        pts = [{"y": float(i), "label": str(i), "time": str(i)} for i in range(5)]
        result = build_progress(pts, "max")
        assert len(result) == 5


# ---------------------------------------------------------------------------
# load_rows
# ---------------------------------------------------------------------------

class TestLoadRows:
    def test_loads_tsv_rows(self, tmp_path):
        p = tmp_path / "data.tsv"
        p.write_text("x\ty\tlabel\n1.0\t2.0\ta\n3.0\t4.0\tb\n")
        rows = load_rows(p)
        assert len(rows) == 2
        assert rows[0]["label"] == "a"
        assert rows[1]["x"] == "3.0"

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "empty.tsv"
        p.write_text("x\ty\n")
        rows = load_rows(p)
        assert rows == []

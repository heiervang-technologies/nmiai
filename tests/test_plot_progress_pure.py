"""Pure-function tests for accounting plot_progress.py.

Covers: to_float, grouped_family_series, cumulative_family_coverage.

All pure functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ACCT_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "accounting")
sys.path.insert(0, _ACCT_DIR)

from plot_progress import to_float, grouped_family_series, cumulative_family_coverage


# ---------------------------------------------------------------------------
# to_float
# ---------------------------------------------------------------------------

class TestToFloat:
    def test_valid_int_string(self):
        assert to_float("42") == 42.0

    def test_valid_float_string(self):
        result = to_float("3.14")
        assert abs(result - 3.14) < 1e-9

    def test_none_gives_none(self):
        assert to_float(None) is None

    def test_empty_string_gives_none(self):
        assert to_float("") is None

    def test_whitespace_only_gives_none(self):
        assert to_float("   ") is None

    def test_non_numeric_gives_none(self):
        assert to_float("abc") is None

    def test_negative_value(self):
        result = to_float("-1.5")
        assert result is not None
        assert abs(result + 1.5) < 1e-9

    def test_whitespace_stripped(self):
        result = to_float("  2.5  ")
        assert result is not None
        assert abs(result - 2.5) < 1e-9

    def test_zero(self):
        assert to_float("0") == 0.0

    def test_returns_float_type(self):
        result = to_float("1.0")
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# grouped_family_series
# ---------------------------------------------------------------------------

class TestGroupedFamilySeries:
    def test_single_row_single_family(self):
        rows = [{"family": "employee", "score": "0.8"}]
        result = grouped_family_series(rows, "score")
        assert "employee" in result
        assert result["employee"] == [(0, 0.8)]

    def test_multiple_rows_same_family(self):
        rows = [
            {"family": "employee", "score": "0.8"},
            {"family": "employee", "score": "0.9"},
        ]
        result = grouped_family_series(rows, "score")
        assert len(result["employee"]) == 2
        assert result["employee"][0] == (0, 0.8)
        assert result["employee"][1] == (1, 0.9)

    def test_multiple_families(self):
        rows = [
            {"family": "employee", "score": "0.8"},
            {"family": "timesheet", "score": "0.6"},
        ]
        result = grouped_family_series(rows, "score")
        assert "employee" in result
        assert "timesheet" in result

    def test_non_numeric_value_skipped(self):
        rows = [{"family": "employee", "score": "N/A"}]
        result = grouped_family_series(rows, "score")
        assert result == {}

    def test_missing_value_key_skipped(self):
        rows = [{"family": "employee"}]
        result = grouped_family_series(rows, "score")
        assert result == {}

    def test_missing_family_uses_unknown(self):
        rows = [{"score": "0.5"}]
        result = grouped_family_series(rows, "score")
        assert "unknown" in result

    def test_returns_dict(self):
        result = grouped_family_series([], "score")
        assert isinstance(result, dict)

    def test_empty_rows(self):
        result = grouped_family_series([], "score")
        assert result == {}

    def test_index_is_row_position(self):
        rows = [
            {"family": "a", "v": "0.1"},
            {"family": "b", "v": "0.2"},  # skipped in series "a"
            {"family": "a", "v": "0.3"},
        ]
        result = grouped_family_series(rows, "v")
        # First a-row has idx=0, second a-row has idx=2
        assert result["a"][0][0] == 0
        assert result["a"][1][0] == 2

    def test_tuple_structure(self):
        rows = [{"family": "f", "v": "1.0"}]
        result = grouped_family_series(rows, "v")
        assert isinstance(result["f"][0], tuple)
        assert len(result["f"][0]) == 2


# ---------------------------------------------------------------------------
# cumulative_family_coverage
# ---------------------------------------------------------------------------

class TestCumulativeFamilyCoverage:
    def test_empty_rows(self):
        result = cumulative_family_coverage([])
        assert result == []

    def test_single_family(self):
        rows = [{"family": "employee"}, {"family": "employee"}]
        result = cumulative_family_coverage(rows)
        assert result == [1, 1]

    def test_two_distinct_families(self):
        rows = [{"family": "employee"}, {"family": "timesheet"}]
        result = cumulative_family_coverage(rows)
        assert result == [1, 2]

    def test_monotonically_nondecreasing(self):
        rows = [
            {"family": "a"},
            {"family": "b"},
            {"family": "a"},
            {"family": "c"},
        ]
        result = cumulative_family_coverage(rows)
        for i in range(1, len(result)):
            assert result[i] >= result[i - 1]

    def test_returns_list(self):
        result = cumulative_family_coverage([{"family": "x"}])
        assert isinstance(result, list)

    def test_length_matches_rows(self):
        rows = [{"family": f"f{i}"} for i in range(5)]
        result = cumulative_family_coverage(rows)
        assert len(result) == 5

    def test_row_without_family_key_no_increment(self):
        rows = [{"family": "a"}, {}, {"family": "a"}]
        result = cumulative_family_coverage(rows)
        # Second row has no family → count stays 1
        assert result == [1, 1, 1]

    def test_final_count_equals_unique_families(self):
        rows = [
            {"family": "a"},
            {"family": "b"},
            {"family": "a"},
            {"family": "c"},
            {"family": "b"},
        ]
        result = cumulative_family_coverage(rows)
        assert result[-1] == 3  # a, b, c

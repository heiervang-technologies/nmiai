"""Tests for tasks/astar-island/regime_predictor.py — pure helper functions.

Covers: cell_code_to_class, cell_bucket, lookup.
All pure functions requiring no file I/O or GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR)

from regime_predictor import cell_bucket, cell_code_to_class, lookup


# ---------------------------------------------------------------------------
# cell_code_to_class
# ---------------------------------------------------------------------------

class TestCellCodeToClass:
    def test_ocean_0_maps_to_0(self):
        assert cell_code_to_class(0) == 0

    def test_ocean_10_maps_to_0(self):
        assert cell_code_to_class(10) == 0

    def test_plains_11_maps_to_0(self):
        assert cell_code_to_class(11) == 0

    def test_settlement_1_maps_to_1(self):
        assert cell_code_to_class(1) == 1

    def test_port_2_maps_to_2(self):
        assert cell_code_to_class(2) == 2

    def test_code_3_maps_to_3(self):
        assert cell_code_to_class(3) == 3

    def test_forest_4_maps_to_4(self):
        assert cell_code_to_class(4) == 4

    def test_mountain_5_maps_to_5(self):
        assert cell_code_to_class(5) == 5

    def test_unknown_maps_to_0(self):
        assert cell_code_to_class(99) == 0


# ---------------------------------------------------------------------------
# cell_bucket
# ---------------------------------------------------------------------------

class TestCellBucket:
    def test_ocean_returns_none(self):
        assert cell_bucket(10, 5.0, 2, 1, False) is None

    def test_mountain_returns_none(self):
        assert cell_bucket(5, 3.0, 0, 0, False) is None

    def test_returns_tuple_of_four(self):
        result = cell_bucket(1, 2.0, 1, 0, False)
        assert isinstance(result, tuple) and len(result) == 4

    def test_settlement_type_is_S(self):
        fine, _, _, _ = cell_bucket(1, 2.0, 0, 0, False)
        assert fine[0] == "S"

    def test_port_type_is_P(self):
        fine, _, _, _ = cell_bucket(2, 2.0, 1, 0, True)
        assert fine[0] == "P"

    def test_forest_type_is_F(self):
        fine, _, _, _ = cell_bucket(4, 5.0, 0, 0, False)
        assert fine[0] == "F"

    def test_plains_type_is_L(self):
        fine, _, _, _ = cell_bucket(11, 5.0, 0, 0, False)
        assert fine[0] == "L"

    def test_dist_clamped_at_15(self):
        fine, _, _, _ = cell_bucket(1, 100.0, 0, 0, False)
        assert fine[1] == 15

    def test_n_ocean_clamped_at_4(self):
        fine, _, _, _ = cell_bucket(1, 2.0, 99, 0, False)
        assert fine[2] == 4

    def test_n_civ_clamped_at_4(self):
        fine, _, _, _ = cell_bucket(1, 2.0, 0, 99, False)
        assert fine[3] == 4


# ---------------------------------------------------------------------------
# lookup
# ---------------------------------------------------------------------------

class TestLookup:
    def _tables(self, data: dict, level: int = 0, n_levels: int = 4) -> list[dict]:
        tables = [{} for _ in range(n_levels)]
        tables[level] = data
        return tables

    def _counts(self, data: dict, level: int = 0, n_levels: int = 4) -> list[dict]:
        counts = [{} for _ in range(n_levels)]
        counts[level] = data
        return counts

    def test_no_match_returns_none(self):
        tables = [{} for _ in range(4)]
        counts = [{} for _ in range(4)]
        result, count, level = lookup(tables, counts, keys=["k1", "k2", "k3", "k4"])
        assert result is None
        assert count == 0

    def test_level_0_hit_with_sufficient_count(self):
        arr = np.array([0.2, 0.3, 0.5])
        tables = self._tables({"key0": arr}, level=0)
        counts = self._counts({"key0": 10}, level=0)  # >= min_counts[0]=5
        result, count, level = lookup(tables, counts, keys=["key0", "k1", "k2", "k3"])
        assert np.array_equal(result, arr)
        assert level == 0

    def test_falls_through_to_level_2(self):
        arr = np.array([0.4, 0.6])
        tables = self._tables({"key2": arr}, level=2)
        counts = self._counts({"key2": 15}, level=2)  # >= min_counts[2]=10
        result, count, level = lookup(tables, counts, keys=["miss0", "miss1", "key2", "k3"])
        assert np.array_equal(result, arr)
        assert level == 2

    def test_level_0_below_min_count_falls_to_fallback(self):
        # Level 0 has key but count=3 (< min_counts[0]=5)
        # Should fail first pass but succeed in fallback pass
        arr = np.array([0.5, 0.5])
        tables = self._tables({"k": arr}, level=0)
        counts = self._counts({"k": 3}, level=0)
        result, count, level = lookup(tables, counts, keys=["k", "miss", "miss", "miss"])
        assert np.array_equal(result, arr)

    def test_returns_tuple_of_three(self):
        tables = [{} for _ in range(4)]
        counts = [{} for _ in range(4)]
        result = lookup(tables, counts, keys=["a", "b", "c", "d"])
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_fallback_level_is_3_on_no_match(self):
        tables = [{} for _ in range(4)]
        counts = [{} for _ in range(4)]
        _, _, level = lookup(tables, counts, keys=["a", "b", "c", "d"])
        assert level == 3

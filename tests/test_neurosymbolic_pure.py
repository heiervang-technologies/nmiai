"""Tests for tasks/astar-island/neurosymbolic_predictor.py — pure helper functions.

Covers: cell_type_str, make_keys.
Both are pure classification/key-building functions with no I/O or GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ASTAR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR)

from neurosymbolic_predictor import cell_type_str, make_keys


# ---------------------------------------------------------------------------
# cell_type_str
# ---------------------------------------------------------------------------

class TestCellTypeStr:
    def test_settlement_1_returns_S(self):
        assert cell_type_str(1) == "S"

    def test_port_2_returns_P(self):
        assert cell_type_str(2) == "P"

    def test_forest_4_returns_F(self):
        assert cell_type_str(4) == "F"

    def test_plains_11_returns_L(self):
        assert cell_type_str(11) == "L"

    def test_empty_0_returns_L(self):
        assert cell_type_str(0) == "L"

    def test_ocean_10_returns_none(self):
        assert cell_type_str(10) is None

    def test_mountain_5_returns_none(self):
        assert cell_type_str(5) is None

    def test_unknown_returns_none(self):
        assert cell_type_str(99) is None


# ---------------------------------------------------------------------------
# make_keys
# ---------------------------------------------------------------------------

class TestMakeKeys:
    def test_ocean_returns_none(self):
        assert make_keys(10, 5.0, 1, 0, 0, False) is None

    def test_mountain_returns_none(self):
        assert make_keys(5, 3.0, 0, 0, 0, False) is None

    def test_returns_list(self):
        result = make_keys(1, 2.0, 1, 0, 0, False)
        assert isinstance(result, list)

    def test_returns_six_levels(self):
        result = make_keys(1, 2.0, 1, 0, 0, False)
        assert len(result) == 6

    def test_all_keys_are_tuples(self):
        result = make_keys(1, 2.0, 1, 0, 0, False)
        assert all(isinstance(k, tuple) for k in result)

    def test_first_element_is_type_string(self):
        result = make_keys(1, 2.0, 0, 0, 0, False)
        assert result[0][0] == "S"

    def test_port_type_in_keys(self):
        result = make_keys(2, 1.0, 0, 0, 0, False)
        assert result[0][0] == "P"

    def test_coast_flag_in_first_key(self):
        with_coast = make_keys(1, 2.0, 1, 0, 0, True)
        without_coast = make_keys(1, 2.0, 1, 0, 0, False)
        # Coast flag (1 vs 0) should differ at position [-1] of fine key
        assert with_coast[0][-1] == 1
        assert without_coast[0][-1] == 0

    def test_dist_clamped_at_15(self):
        result = make_keys(1, 100.0, 0, 0, 0, False)
        # dist in first key should be clamped to 15
        assert result[0][1] == 15

    def test_n_ocean_clamped_at_4(self):
        result = make_keys(1, 2.0, 10, 0, 0, False)
        assert result[0][3] == 4

    def test_n_civ_clamped_at_5(self):
        result = make_keys(1, 2.0, 0, 10, 0, False)
        assert result[0][2] == 5

    def test_broad_key_uses_min_dist_8(self):
        result = make_keys(4, 15.0, 0, 0, 0, False)
        # Broad key (last) should be (type, min(d, 8)) = ("F", 8)
        assert result[-1] == ("F",) or result[-2][1] <= 8

    def test_singleton_key_at_last_level(self):
        result = make_keys(4, 5.0, 0, 0, 0, False)
        # Last key should be just the type tuple ("F",)
        assert result[-1] == ("F",)

"""Tests for tasks/astar-island/neighborhood_predictor.py — pure helper functions.

Covers: cell_to_type, dist_bin.
Both are pure classification/binning functions with no I/O or GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ASTAR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR)

from neighborhood_predictor import cell_to_type, dist_bin


# ---------------------------------------------------------------------------
# cell_to_type
# ---------------------------------------------------------------------------

class TestCellToType:
    def test_settlement_1_returns_settlement(self):
        assert cell_to_type(1) == "settlement"

    def test_port_2_returns_port(self):
        assert cell_to_type(2) == "port"

    def test_forest_4_returns_forest(self):
        assert cell_to_type(4) == "forest"

    def test_plains_11_returns_plains(self):
        assert cell_to_type(11) == "plains"

    def test_empty_0_returns_empty(self):
        assert cell_to_type(0) == "empty"

    def test_ocean_10_returns_none(self):
        assert cell_to_type(10) is None

    def test_mountain_5_returns_none(self):
        assert cell_to_type(5) is None

    def test_returns_string_or_none(self):
        result = cell_to_type(1)
        assert result is None or isinstance(result, str)

    def test_unknown_code_returns_none(self):
        assert cell_to_type(99) is None


# ---------------------------------------------------------------------------
# dist_bin
# ---------------------------------------------------------------------------

class TestDistBin:
    def test_distance_zero_is_bin_0(self):
        assert dist_bin(0) == 0

    def test_distance_one_is_bin_0(self):
        assert dist_bin(1) == 0

    def test_distance_two_is_bin_1(self):
        assert dist_bin(2) == 1

    def test_distance_three_is_bin_1(self):
        assert dist_bin(3) == 1

    def test_distance_four_is_bin_2(self):
        assert dist_bin(4) == 2

    def test_distance_six_is_bin_2(self):
        assert dist_bin(6) == 2

    def test_distance_seven_is_bin_3(self):
        assert dist_bin(7) == 3

    def test_large_distance_is_bin_3(self):
        assert dist_bin(100) == 3

    def test_returns_int(self):
        assert isinstance(dist_bin(5), int)

    def test_bin_values_are_0_to_3(self):
        for d in [0, 1, 2, 3, 4, 5, 6, 7, 8, 20]:
            b = dist_bin(d)
            assert 0 <= b <= 3

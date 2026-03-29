"""Pure-function tests for astar-island/solver.py and value_viewport_selector.py.

Covers:
  solver                  : full_coverage_viewports, cell_code_to_class
  value_viewport_selector : cell_entropy

All pure — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock requests before importing modules that use it at module level
sys.modules.setdefault("requests", MagicMock())
sys.modules.setdefault("regime_predictor", MagicMock())

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

import solver
import value_viewport_selector as vvs


# ---------------------------------------------------------------------------
# solver.full_coverage_viewports
# ---------------------------------------------------------------------------

class TestFullCoverageViewports:
    def test_returns_nine_viewports(self):
        viewports = solver.full_coverage_viewports()
        assert len(viewports) == 9

    def test_all_viewports_15x15(self):
        for vx, vy, vw, vh in solver.full_coverage_viewports():
            assert vw == 15
            assert vh == 15

    def test_returns_list_of_4_tuples(self):
        for vp in solver.full_coverage_viewports():
            assert isinstance(vp, tuple)
            assert len(vp) == 4

    def test_x_positions_cover_full_width(self):
        xs = {vx for vx, vy, vw, vh in solver.full_coverage_viewports()}
        # Must include 0 and reach at least x=25 (25+15=40)
        assert 0 in xs
        assert max(xs) + 15 >= 40

    def test_y_positions_cover_full_height(self):
        ys = {vy for vx, vy, vw, vh in solver.full_coverage_viewports()}
        assert 0 in ys
        assert max(ys) + 15 >= 40

    def test_nine_unique_positions(self):
        positions = [(vx, vy) for vx, vy, vw, vh in solver.full_coverage_viewports()]
        assert len(set(positions)) == 9


# ---------------------------------------------------------------------------
# solver.cell_code_to_class
# ---------------------------------------------------------------------------

class TestSolverCellCodeToClass:
    @pytest.mark.parametrize("code,expected", [
        (0, 0), (10, 0), (11, 0),
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    ])
    def test_known_codes(self, code, expected):
        assert solver.cell_code_to_class(code) == expected

    def test_unknown_returns_zero(self):
        assert solver.cell_code_to_class(99) == 0

    def test_ruin_maps_to_3(self):
        assert solver.cell_code_to_class(3) == 3

    def test_mountain_maps_to_5(self):
        assert solver.cell_code_to_class(5) == 5


# ---------------------------------------------------------------------------
# value_viewport_selector.cell_entropy
# ---------------------------------------------------------------------------

class TestCellEntropy:
    def test_uniform_maximum(self):
        p = np.ones(6) / 6
        expected = -np.sum(p * np.log(p))
        assert vvs.cell_entropy(p) == pytest.approx(expected)

    def test_deterministic_near_zero(self):
        # Floor at 1e-10 means residuals from 5 zero classes ≈ 5 * 1e-10 * |log(1e-10)|
        p = np.zeros(6)
        p[0] = 1.0
        result = vvs.cell_entropy(p)
        assert result == pytest.approx(0.0, abs=1e-7)

    def test_nonneg(self):
        p = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        assert vvs.cell_entropy(p) >= 0.0

    def test_more_diffuse_higher_entropy(self):
        p_conc = np.array([0.98, 0.004, 0.004, 0.004, 0.004, 0.004])
        p_unif = np.ones(6) / 6
        assert vvs.cell_entropy(p_unif) > vvs.cell_entropy(p_conc)

    def test_two_equal_half_is_log2(self):
        p = np.zeros(6)
        p[0] = 0.5
        p[1] = 0.5
        expected = np.log(2)
        assert vvs.cell_entropy(p) == pytest.approx(expected, rel=1e-6)

    def test_scalar_returned(self):
        p = np.ones(6) / 6
        result = vvs.cell_entropy(p)
        assert np.ndim(result) == 0

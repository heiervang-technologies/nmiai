"""Pure-function tests for astar-island/test_h3.py and test_deterministic_5phase.py.

Covers:
  test_h3.py                  : forest_never_grows, mountain_never_grows, no_isolated_civ_in_gt
  test_deterministic_5phase.py: get_regime

All pure numpy/scipy functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

from test_h3 import forest_never_grows, mountain_never_grows, no_isolated_civ_in_gt
from test_deterministic_5phase import get_regime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gt(h, w, dominant_class, n_classes=6):
    """Return (h, w, n_classes) gt with dominant_class having prob=1."""
    gt = np.zeros((h, w, n_classes))
    gt[:, :, dominant_class] = 1.0
    return gt


# ---------------------------------------------------------------------------
# forest_never_grows
# ---------------------------------------------------------------------------

class TestForestNeverGrows:
    def test_no_new_forest_returns_false(self):
        ig = np.zeros((3, 3), dtype=np.int32)
        gt = _make_gt(3, 3, 0)  # all class 0 (not forest)
        result = forest_never_grows(ig, gt)
        assert not result.any()

    def test_gt_forest_on_init_forest_returns_false(self):
        ig = np.full((2, 2), 4, dtype=np.int32)  # all forest
        gt = _make_gt(2, 2, 4)  # gt is forest too
        result = forest_never_grows(ig, gt)
        assert not result.any()

    def test_new_forest_returns_true(self):
        # initial grid has no forest, but gt predicts forest at one cell
        ig = np.zeros((3, 3), dtype=np.int32)
        gt = _make_gt(3, 3, 0)
        gt[1, 1, :] = 0
        gt[1, 1, 4] = 1.0  # cell (1,1) has forest in gt
        result = forest_never_grows(ig, gt)
        assert result[1, 1]
        # other cells should be False
        assert not result[0, 0]

    def test_returns_bool_array(self):
        ig = np.zeros((4, 4), dtype=np.int32)
        gt = _make_gt(4, 4, 0)
        result = forest_never_grows(ig, gt)
        assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# mountain_never_grows
# ---------------------------------------------------------------------------

class TestMountainNeverGrows:
    def test_no_new_mountain_returns_false(self):
        ig = np.zeros((3, 3), dtype=np.int32)
        gt = _make_gt(3, 3, 0)
        result = mountain_never_grows(ig, gt)
        assert not result.any()

    def test_mountain_stays_mountain_returns_false(self):
        ig = np.full((2, 2), 5, dtype=np.int32)
        gt = _make_gt(2, 2, 5)
        result = mountain_never_grows(ig, gt)
        assert not result.any()

    def test_new_mountain_returns_true(self):
        ig = np.zeros((3, 3), dtype=np.int32)
        gt = _make_gt(3, 3, 0)
        gt[0, 0, :] = 0
        gt[0, 0, 5] = 1.0  # mountain class=5 in gt but not in ig
        result = mountain_never_grows(ig, gt)
        assert result[0, 0]

    def test_returns_shape_hw(self):
        ig = np.zeros((5, 6), dtype=np.int32)
        gt = _make_gt(5, 6, 0)
        result = mountain_never_grows(ig, gt)
        assert result.shape == (5, 6)


# ---------------------------------------------------------------------------
# no_isolated_civ_in_gt
# ---------------------------------------------------------------------------

class TestNoIsolatedCivInGt:
    def test_no_civ_returns_false(self):
        ig = np.zeros((5, 5), dtype=np.int32)
        gt = _make_gt(5, 5, 0)
        result = no_isolated_civ_in_gt(ig, gt)
        assert not result.any()

    def test_isolated_settlement_returns_true(self):
        # 5x5 grid; civ only at center, no neighbors
        ig = np.zeros((5, 5), dtype=np.int32)
        gt = _make_gt(5, 5, 0)
        gt[2, 2, :] = 0
        gt[2, 2, 1] = 1.0  # settlement at center
        result = no_isolated_civ_in_gt(ig, gt)
        assert result[2, 2]

    def test_adjacent_civ_returns_false(self):
        ig = np.zeros((5, 5), dtype=np.int32)
        gt = _make_gt(5, 5, 0)
        # Two adjacent settlements → neither is isolated
        gt[2, 2, :] = 0; gt[2, 2, 1] = 1.0
        gt[2, 3, :] = 0; gt[2, 3, 1] = 1.0
        result = no_isolated_civ_in_gt(ig, gt)
        assert not result[2, 2]
        assert not result[2, 3]

    def test_port_class_2_also_counts(self):
        ig = np.zeros((5, 5), dtype=np.int32)
        gt = _make_gt(5, 5, 0)
        gt[0, 0, :] = 0; gt[0, 0, 2] = 1.0  # isolated port at corner
        result = no_isolated_civ_in_gt(ig, gt)
        assert result[0, 0]

    def test_returns_shape_hw(self):
        ig = np.zeros((4, 4), dtype=np.int32)
        gt = _make_gt(4, 4, 0)
        result = no_isolated_civ_in_gt(ig, gt)
        assert result.shape == (4, 4)


# ---------------------------------------------------------------------------
# get_regime
# ---------------------------------------------------------------------------

class TestGetRegime:
    def test_few_civ_returns_0(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        # < 30 civ cells
        grid[0, 0] = 1
        assert get_regime(grid) == 0

    def test_many_civ_returns_2(self):
        # > 150 civ cells
        grid = np.ones((15, 15), dtype=np.int32)  # 225 settlements
        assert get_regime(grid) == 2

    def test_moderate_civ_returns_1(self):
        # between 30 and 150
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[:6, :6] = 1  # 36 cells
        assert get_regime(grid) == 1

    def test_ports_count_as_civ(self):
        grid = np.zeros((20, 20), dtype=np.int32)
        grid[:8, :8] = 2  # 64 port cells → moderate
        assert get_regime(grid) == 1

    def test_exactly_30_is_moderate(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        # exactly 30 civ cells
        flat = grid.flatten()
        flat[:30] = 1
        grid = flat.reshape(10, 10)
        assert get_regime(grid) == 1

    def test_exactly_150_is_moderate(self):
        grid = np.zeros((20, 20), dtype=np.int32)
        flat = grid.flatten()
        flat[:150] = 1
        grid = flat.reshape(20, 20)
        assert get_regime(grid) == 1

    def test_empty_grid_returns_0(self):
        grid = np.zeros((10, 10), dtype=np.int32)
        assert get_regime(grid) == 0

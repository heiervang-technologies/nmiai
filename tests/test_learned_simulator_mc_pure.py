"""Tests for learned_simulator.py and cpu_monte_carlo.py pure helpers.

Covers:
  - learned_simulator: grid_to_class, count_neighbors_of_class,
      compute_features_vectorized, kl_divergence, entropy
  - cpu_monte_carlo: calc_wkl
All pure functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from learned_simulator import (
    compute_features_vectorized,
    count_neighbors_of_class,
    entropy,
    grid_to_class,
    kl_divergence,
)
from cpu_monte_carlo import calc_wkl


# ---------------------------------------------------------------------------
# learned_simulator.grid_to_class
# ---------------------------------------------------------------------------

class TestGridToClass:
    def test_ocean_maps_0(self):
        g = np.array([[10, 10]], dtype=np.int32)
        result = grid_to_class(g)
        assert (result == 0).all()

    def test_plains_maps_0(self):
        g = np.array([[11]], dtype=np.int32)
        assert grid_to_class(g)[0, 0] == 0

    def test_settlement_maps_1(self):
        g = np.array([[1]], dtype=np.int32)
        assert grid_to_class(g)[0, 0] == 1

    def test_port_maps_2(self):
        g = np.array([[2]], dtype=np.int32)
        assert grid_to_class(g)[0, 0] == 2

    def test_ruin_maps_3(self):
        g = np.array([[3]], dtype=np.int32)
        assert grid_to_class(g)[0, 0] == 3

    def test_forest_maps_4(self):
        g = np.array([[4]], dtype=np.int32)
        assert grid_to_class(g)[0, 0] == 4

    def test_mountain_maps_5(self):
        g = np.array([[5]], dtype=np.int32)
        assert grid_to_class(g)[0, 0] == 5


# ---------------------------------------------------------------------------
# learned_simulator.count_neighbors_of_class
# ---------------------------------------------------------------------------

class TestCountNeighborsOfClass:
    def test_isolated_cell_has_no_neighbors(self):
        cls_grid = np.zeros((5, 5), dtype=np.int32)
        cls_grid[2, 2] = 1
        result = count_neighbors_of_class(cls_grid, 1, 5, 5)
        assert result[2, 2] == 0

    def test_center_surrounded_by_class(self):
        cls_grid = np.ones((3, 3), dtype=np.int32)
        result = count_neighbors_of_class(cls_grid, 1, 3, 3)
        # Center (1,1) has 8 neighbors, all class 1
        assert result[1, 1] == 8

    def test_corner_has_fewer_neighbors(self):
        cls_grid = np.ones((3, 3), dtype=np.int32)
        result = count_neighbors_of_class(cls_grid, 1, 3, 3)
        # Corner (0,0) has 3 neighbors
        assert result[0, 0] == 3

    def test_missing_class_gives_zero(self):
        cls_grid = np.zeros((4, 4), dtype=np.int32)
        result = count_neighbors_of_class(cls_grid, 9, 4, 4)
        assert (result == 0).all()

    def test_output_shape(self):
        cls_grid = np.zeros((6, 8), dtype=np.int32)
        result = count_neighbors_of_class(cls_grid, 0, 6, 8)
        assert result.shape == (6, 8)


# ---------------------------------------------------------------------------
# learned_simulator.compute_features_vectorized
# ---------------------------------------------------------------------------

class TestComputeFeaturesVectorized:
    def _all_plains_grid(self, h=10, w=10):
        return np.full((h, w), 11, dtype=np.int32)

    def _grid_with_settlement(self, h=10, w=10):
        g = np.full((h, w), 11, dtype=np.int32)
        g[5, 5] = 1
        return g

    def test_returns_4_arrays(self):
        g = self._all_plains_grid()
        import learned_simulator as ls
        cls_g = ls.grid_to_class(g)
        result = compute_features_vectorized(cls_g, g, 10, 10)
        assert len(result) == 4

    def test_output_shapes(self):
        g = self._all_plains_grid(6, 8)
        import learned_simulator as ls
        cls_g = ls.grid_to_class(g)
        n_civ, n_forest, n_ocean, dist = compute_features_vectorized(cls_g, g, 6, 8)
        for arr in (n_civ, n_forest, n_ocean, dist):
            assert arr.shape == (6, 8)

    def test_no_civ_all_high_dist(self):
        g = self._all_plains_grid()
        import learned_simulator as ls
        cls_g = ls.grid_to_class(g)
        _, _, _, dist_bin = compute_features_vectorized(cls_g, g, 10, 10)
        # All cells far from civ → dist_bin=3
        assert (dist_bin == 3).all()

    def test_settlement_cell_dist_bin_0(self):
        g = self._grid_with_settlement()
        import learned_simulator as ls
        cls_g = ls.grid_to_class(g)
        _, _, _, dist_bin = compute_features_vectorized(cls_g, g, 10, 10)
        # Settlement itself at dist=0 → bin 0
        assert dist_bin[5, 5] == 0

    def test_adjacent_to_settlement_dist_bin_0(self):
        g = self._grid_with_settlement()
        import learned_simulator as ls
        cls_g = ls.grid_to_class(g)
        _, _, _, dist_bin = compute_features_vectorized(cls_g, g, 10, 10)
        # Cell at dist=1 → bin 0
        assert dist_bin[5, 6] == 0

    def test_ocean_neighbors_counted(self):
        g = np.full((5, 5), 11, dtype=np.int32)
        g[2, 0] = 10  # ocean on left edge
        import learned_simulator as ls
        cls_g = ls.grid_to_class(g)
        _, _, n_ocean, _ = compute_features_vectorized(cls_g, g, 5, 5)
        # Cell (2,1) should have 1 ocean neighbor
        assert n_ocean[2, 1] >= 1


# ---------------------------------------------------------------------------
# learned_simulator.kl_divergence
# ---------------------------------------------------------------------------

class TestKlDivergence:
    def test_identical_near_zero(self):
        p = np.full((3, 3, 6), 1.0 / 6)
        result = kl_divergence(p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_shape_collapses_class_axis(self):
        p = np.full((4, 5, 6), 1.0 / 6)
        result = kl_divergence(p, p)
        assert result.shape == (4, 5)

    def test_non_negative(self):
        p = np.full((2, 2, 6), 1.0 / 6)
        q = np.random.dirichlet(np.ones(6), size=(2, 2))
        q = q.reshape(2, 2, 6)
        result = kl_divergence(p, q)
        assert (result >= -1e-10).all()


# ---------------------------------------------------------------------------
# learned_simulator.entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_6_class(self):
        p = np.full((2, 2, 6), 1.0 / 6)
        result = entropy(p)
        np.testing.assert_allclose(result, np.log2(6), atol=1e-4)

    def test_deterministic_zero(self):
        p = np.zeros((2, 2, 6))
        p[:, :, 0] = 1.0
        result = entropy(p)
        np.testing.assert_allclose(result, 0.0, atol=1e-4)

    def test_shape_collapses_class_axis(self):
        p = np.full((3, 4, 6), 1.0 / 6)
        result = entropy(p)
        assert result.shape == (3, 4)

    def test_non_negative(self):
        p = np.random.dirichlet(np.ones(6), size=(3, 3)).reshape(3, 3, 6)
        result = entropy(p)
        assert (result >= -1e-10).all()


# ---------------------------------------------------------------------------
# cpu_monte_carlo.calc_wkl
# ---------------------------------------------------------------------------

class TestCalcWkl:
    def _uniform(self, h=5, w=5, c=6):
        return np.full((h, w, c), 1.0 / c)

    def _grid_plains(self, h=5, w=5):
        return np.full((h, w), 11, dtype=np.int32)

    def test_identical_near_zero(self):
        p = self._uniform()
        ig = self._grid_plains()
        result = calc_wkl(p, p, ig)
        assert abs(result) < 1e-4

    def test_returns_scalar(self):
        p = self._uniform()
        ig = self._grid_plains()
        result = calc_wkl(p, p, ig)
        assert np.isscalar(result) or result.ndim == 0

    def test_non_negative(self):
        p = self._uniform()
        q = np.random.dirichlet(np.ones(6), size=(5, 5)).reshape(5, 5, 6)
        ig = self._grid_plains()
        result = calc_wkl(p, q, ig)
        assert result >= 0.0

    def test_ocean_mountain_excluded(self):
        # Grid with all ocean/mountain should give nan or handled value
        p = self._uniform()
        ig = np.full((5, 5), 10, dtype=np.int32)  # all ocean → dynamic_mask all False
        ig[2, 2] = 11  # one non-ocean cell
        result = calc_wkl(p, p, ig)
        assert np.isfinite(result)

    def test_higher_divergence_for_different_dists(self):
        ig = self._grid_plains()
        p = self._uniform()
        q = np.zeros_like(p)
        q[:, :, 0] = 1.0  # deterministic, different from uniform
        result = calc_wkl(p, q, ig)
        result_same = calc_wkl(p, p, ig)
        assert result > result_same

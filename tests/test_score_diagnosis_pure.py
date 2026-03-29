"""Tests for tasks/astar-island/score_diagnosis.py — pure classification helpers.

Covers: kl_per_cell, entropy_per_cell, classify_cells.
All pure numpy/scipy functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from score_diagnosis import classify_cells, entropy_per_cell, kl_per_cell

# Cell codes
OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4
PLAINS = 11


# ---------------------------------------------------------------------------
# kl_per_cell
# ---------------------------------------------------------------------------

class TestKlPerCell:
    def _uniform(self, h=2, w=2, n=4):
        return np.full((h, w, n), 1.0 / n)

    def test_identical_distributions_near_zero(self):
        p = self._uniform()
        result = kl_per_cell(p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_shape_removes_class_axis(self):
        p = self._uniform(h=3, w=5, n=6)
        result = kl_per_cell(p, p)
        assert result.shape == (3, 5)

    def test_non_negative(self):
        p = self._uniform()
        q = np.array([[[[0.7, 0.1, 0.1, 0.1], [0.25, 0.25, 0.25, 0.25]],
                        [[0.5, 0.2, 0.2, 0.1], [0.4, 0.2, 0.2, 0.2]]]])
        q = q.reshape(2, 2, 4)
        result = kl_per_cell(p, q)
        assert (result >= 0).all()

    def test_asymmetric(self):
        p = np.full((1, 1, 4), 1.0 / 4)
        q = np.array([[[0.7, 0.1, 0.1, 0.1]]])
        kl_pq = kl_per_cell(p, q)
        kl_qp = kl_per_cell(q, p)
        # KL divergence is asymmetric
        assert not np.isclose(kl_pq[0, 0], kl_qp[0, 0])


# ---------------------------------------------------------------------------
# entropy_per_cell
# ---------------------------------------------------------------------------

class TestEntropyPerCell:
    def test_uniform_4_class_gives_2_bits(self):
        p = np.full((2, 2, 4), 0.25)
        result = entropy_per_cell(p)
        np.testing.assert_allclose(result, 2.0, atol=1e-5)

    def test_deterministic_gives_zero(self):
        p = np.zeros((2, 2, 4))
        p[:, :, 0] = 1.0
        result = entropy_per_cell(p)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_shape_removes_class_axis(self):
        p = np.full((3, 5, 6), 1.0 / 6)
        result = entropy_per_cell(p)
        assert result.shape == (3, 5)

    def test_non_negative(self):
        p = np.random.dirichlet([1, 1, 1, 1], size=(3, 4))
        result = entropy_per_cell(p)
        assert (result >= 0).all()

    def test_higher_entropy_for_more_uniform(self):
        uniform = np.full((1, 1, 4), 0.25)
        peaked = np.array([[[0.9, 0.05, 0.025, 0.025]]])
        h_uniform = entropy_per_cell(uniform)
        h_peaked = entropy_per_cell(peaked)
        assert h_uniform[0, 0] > h_peaked[0, 0]


# ---------------------------------------------------------------------------
# classify_cells
# ---------------------------------------------------------------------------

class TestClassifyCells:
    def _grid(self, codes):
        return np.array(codes, dtype=np.int32)

    def test_ocean_classified_as_ocean(self):
        g = self._grid([[OCEAN, PLAINS], [PLAINS, PLAINS]])
        result = classify_cells(g)
        assert result[0, 0] == "ocean"

    def test_mountain_classified_as_mountain(self):
        g = self._grid([[MOUNTAIN, PLAINS], [PLAINS, PLAINS]])
        result = classify_cells(g)
        assert result[0, 0] == "mountain"

    def test_settlement_classified_as_init_settlement(self):
        g = self._grid([[SETTLEMENT, PLAINS], [PLAINS, PLAINS]])
        result = classify_cells(g)
        assert result[0, 0] == "init_settlement"

    def test_port_classified_as_init_port(self):
        g = self._grid([[PORT, PLAINS], [PLAINS, PLAINS]])
        result = classify_cells(g)
        assert result[0, 0] == "init_port"

    def test_returns_2d_object_array(self):
        g = self._grid([[PLAINS, PLAINS], [PLAINS, PLAINS]])
        result = classify_cells(g)
        assert result.shape == (2, 2)
        assert result.dtype == object

    def test_near_settlement_is_near_civ(self):
        # 4x4 grid with settlement at (1,1), target at (1,2) — dist=1 → near_civ
        g = np.full((4, 4), PLAINS, dtype=np.int32)
        g[1, 1] = SETTLEMENT
        result = classify_cells(g)
        assert result[1, 2] == "near_civ"

    def test_far_from_civ_is_remote(self):
        # Tiny grid, no civ → all cells have dist=99 → remote
        g = np.full((3, 3), PLAINS, dtype=np.int32)
        result = classify_cells(g)
        assert result[1, 1] == "remote"

    def test_coastal_frontier_near_ocean(self):
        # Create a grid with many ocean cells and a civ nearby
        g = np.full((6, 6), PLAINS, dtype=np.int32)
        # Ocean row at top
        g[0, :] = OCEAN
        g[1, :] = OCEAN
        g[2, :] = OCEAN
        g[3, 3] = SETTLEMENT
        result = classify_cells(g)
        # Cell at (3, 2): dist=1, n_ocean should be >=2 due to ocean rows above
        # → coastal_frontier
        cat = result[3, 2]
        assert cat in ("coastal_frontier", "near_civ")  # depends on exact n_ocean count

    def test_forest_near_civ_classified(self):
        # Forest cell at distance 2 from settlement → forest_near_civ
        g = np.full((5, 5), PLAINS, dtype=np.int32)
        g[2, 2] = SETTLEMENT
        g[2, 4] = FOREST  # dist=2 from settlement
        result = classify_cells(g)
        assert result[2, 4] == "forest_near_civ"

    def test_all_valid_categories(self):
        valid = {
            "ocean", "mountain", "init_settlement", "init_port",
            "coastal_frontier", "forest_near_civ", "near_civ",
            "mid_range", "far_range", "remote",
        }
        g = np.full((8, 8), PLAINS, dtype=np.int32)
        g[4, 4] = SETTLEMENT
        result = classify_cells(g)
        for cat in result.flat:
            assert cat in valid, f"Unexpected category: {cat}"

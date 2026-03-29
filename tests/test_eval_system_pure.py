"""Tests for tasks/astar-island/eval_system.py — pure evaluation helpers.

Covers: kl_divergence_per_cell, entropy_per_cell, score_prediction, simulate_observations.
All pure numpy functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from eval_system import (
    entropy_per_cell,
    kl_divergence_per_cell,
    score_prediction,
    simulate_observations,
)


# ---------------------------------------------------------------------------
# kl_divergence_per_cell
# ---------------------------------------------------------------------------

class TestKlDivergencePerCell:
    def test_identical_near_zero(self):
        p = np.full((2, 2, 6), 1.0 / 6)
        result = kl_divergence_per_cell(p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_shape_removes_class_axis(self):
        p = np.full((3, 4, 6), 1.0 / 6)
        result = kl_divergence_per_cell(p, p)
        assert result.shape == (3, 4)

    def test_non_negative(self):
        p = np.full((2, 2, 6), 1.0 / 6)
        q = np.full((2, 2, 6), 1.0 / 6)
        q[0, 0, 0] += 0.5
        q /= q.sum(axis=2, keepdims=True)
        result = kl_divergence_per_cell(p, q)
        assert (result >= -1e-12).all()


# ---------------------------------------------------------------------------
# entropy_per_cell
# ---------------------------------------------------------------------------

class TestEntropyPerCell:
    def test_uniform_6_class_gives_log2_6_bits(self):
        p = np.full((2, 2, 6), 1.0 / 6)
        result = entropy_per_cell(p)
        np.testing.assert_allclose(result, np.log2(6), atol=1e-5)

    def test_deterministic_gives_zero(self):
        p = np.zeros((2, 2, 6))
        p[:, :, 0] = 1.0
        result = entropy_per_cell(p)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_shape_removes_class_axis(self):
        p = np.full((3, 5, 6), 1.0 / 6)
        result = entropy_per_cell(p)
        assert result.shape == (3, 5)


# ---------------------------------------------------------------------------
# score_prediction
# ---------------------------------------------------------------------------

class TestScorePrediction:
    def _uniform(self, h=3, w=3, c=6):
        return np.full((h, w, c), 1.0 / c)

    def test_returns_dict(self):
        p = self._uniform()
        result = score_prediction(p, p)
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        p = self._uniform()
        result = score_prediction(p, p)
        for key in ("mean_kl", "mean_weighted_kl", "mean_entropy", "dynamic_cells"):
            assert key in result

    def test_identical_gives_near_zero_kl(self):
        p = self._uniform()
        result = score_prediction(p, p)
        assert abs(result["mean_kl"]) < 1e-4

    def test_dynamic_cells_positive_for_uniform(self):
        p = self._uniform()
        result = score_prediction(p, p)
        # Uniform distributions have entropy > 0.01 → should be dynamic
        assert result["dynamic_cells"] > 0

    def test_all_static_grid_gives_zero_dynamic_cells(self):
        # Deterministic grid has entropy=0 → no dynamic cells
        p = np.zeros((3, 3, 6))
        p[:, :, 0] = 1.0
        result = score_prediction(p, p)
        assert result["dynamic_cells"] == 0

    def test_mean_kl_non_negative(self):
        p = self._uniform()
        q = np.random.dirichlet([1.0] * 6, size=(3, 3))
        result = score_prediction(p, q)
        assert result["mean_kl"] >= 0.0

    def test_kl_array_shape(self):
        p = self._uniform(h=4, w=5)
        result = score_prediction(p, p)
        assert result["kl_array"].shape == (4, 5)


# ---------------------------------------------------------------------------
# simulate_observations
# ---------------------------------------------------------------------------

class TestSimulateObservations:
    def _uniform_gt(self, h=10, w=10, c=6):
        return np.full((h, w, c), 1.0 / c)

    def _simple_grid(self, h=10, w=10):
        g = np.full((h, w), 11, dtype=np.int32)  # all plains
        g[4, 4] = 1  # one settlement
        return g

    def test_returns_list(self):
        gt = self._uniform_gt()
        ig = self._simple_grid()
        result = simulate_observations(gt, ig, n_viewports=3)
        assert isinstance(result, list)

    def test_n_viewports_observations(self):
        gt = self._uniform_gt()
        ig = self._simple_grid()
        result = simulate_observations(gt, ig, n_viewports=5)
        assert len(result) == 5

    def test_each_obs_has_expected_keys(self):
        gt = self._uniform_gt()
        ig = self._simple_grid()
        for obs in simulate_observations(gt, ig, n_viewports=2):
            assert "viewport_x" in obs
            assert "viewport_y" in obs
            assert "grid" in obs

    def test_grid_cells_are_valid_codes(self):
        gt = self._uniform_gt()
        ig = self._simple_grid()
        valid_codes = {0, 1, 2, 3, 4, 5}  # Empty, Settlement, Port, Ruin, Forest, Mountain
        for obs in simulate_observations(gt, ig, n_viewports=3, rng=np.random.default_rng(42)):
            for row in obs["grid"]:
                for cell in row:
                    assert cell in valid_codes

    def test_seeded_rng_reproducible(self):
        gt = self._uniform_gt()
        ig = self._simple_grid()
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        obs1 = simulate_observations(gt, ig, n_viewports=3, rng=rng1)
        obs2 = simulate_observations(gt, ig, n_viewports=3, rng=rng2)
        for o1, o2 in zip(obs1, obs2):
            assert o1["viewport_x"] == o2["viewport_x"]
            assert o1["viewport_y"] == o2["viewport_y"]
            assert o1["grid"] == o2["grid"]

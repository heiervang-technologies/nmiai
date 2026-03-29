"""Pure-function tests for astar-island gpu_mc_solver.py.

Covers: quantization_snap, apply_structural_zeros, observation_overlay, compute_obs_ll.

All pure functions — no GPU, file system, or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

from gpu_mc_solver import (
    quantization_snap,
    apply_structural_zeros,
    observation_overlay,
    compute_obs_ll,
)

OCEAN, MOUNTAIN = 10, 5
N_CLASSES = 6
GRID_SIZE = 40


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform(h: int = GRID_SIZE, w: int = GRID_SIZE) -> np.ndarray:
    return np.full((h, w, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float64)


def _plains_grid(h: int = GRID_SIZE, w: int = GRID_SIZE) -> np.ndarray:
    return np.full((h, w), 11, dtype=np.int32)


def _obs(vx: int = 0, vy: int = 0, w: int = 5, h: int = 5, code: int = 1) -> dict:
    """Minimal observation dict."""
    return {
        "viewport_x": vx,
        "viewport_y": vy,
        "grid": np.full((h, w), code, dtype=np.int32).tolist(),
    }


# ---------------------------------------------------------------------------
# quantization_snap
# ---------------------------------------------------------------------------

class TestQuantizationSnap:
    def test_output_shape(self):
        pred = _uniform()
        result = quantization_snap(pred)
        assert result.shape == pred.shape

    def test_sums_to_one(self):
        pred = _uniform()
        result = quantization_snap(pred)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((GRID_SIZE, GRID_SIZE)), atol=1e-5)

    def test_nonneg(self):
        pred = _uniform()
        result = quantization_snap(pred)
        assert (result >= 0).all()

    def test_multiples_of_target_sum(self):
        # With target_sum=200, each cell's distribution should be quantized
        pred = _uniform()
        result = quantization_snap(pred, target_sum=200)
        # Each value should be close to a multiple of 1/200
        scaled = result * 200
        rounded = np.round(scaled)
        np.testing.assert_allclose(scaled, rounded, atol=0.01)

    def test_returns_float(self):
        pred = _uniform()
        result = quantization_snap(pred)
        assert result.dtype in (np.float64, np.float32)

    def test_custom_target_sum(self):
        # target_sum=100 should also produce valid distributions
        pred = _uniform()
        result = quantization_snap(pred, target_sum=100)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((GRID_SIZE, GRID_SIZE)), atol=1e-5)


# ---------------------------------------------------------------------------
# apply_structural_zeros
# ---------------------------------------------------------------------------

class TestApplyStructuralZeros:
    def test_mountain_gets_class5(self):
        pred = _uniform(5, 5)
        ig = _plains_grid(5, 5)
        ig[2, 2] = MOUNTAIN
        result = apply_structural_zeros(pred.copy(), ig)
        np.testing.assert_allclose(result[2, 2], [0, 0, 0, 0, 0, 1])

    def test_ocean_gets_class0(self):
        pred = _uniform(5, 5)
        ig = _plains_grid(5, 5)
        ig[1, 1] = OCEAN
        result = apply_structural_zeros(pred.copy(), ig)
        np.testing.assert_allclose(result[1, 1], [1, 0, 0, 0, 0, 0])

    def test_shape_preserved(self):
        pred = _uniform()
        ig = _plains_grid()
        result = apply_structural_zeros(pred.copy(), ig)
        assert result.shape == pred.shape

    def test_no_civ_all_far(self):
        # No settlement/port → all cells at dist=100 → plains/empty cells get [1,0,0,0,0,0]
        pred = _uniform(5, 5)
        ig = np.full((5, 5), 11, dtype=np.int32)  # all plains
        result = apply_structural_zeros(pred.copy(), ig)
        # All plains far from civ → class-0 forced
        expected = np.tile([1, 0, 0, 0, 0, 0], (5, 5, 1))
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_forest_far_from_civ(self):
        pred = _uniform(5, 5)
        ig = np.full((5, 5), 4, dtype=np.int32)  # all forest
        result = apply_structural_zeros(pred.copy(), ig)
        # Forest far from civ → [0,0,0,0,1,0]
        np.testing.assert_allclose(result, np.tile([0, 0, 0, 0, 1, 0], (5, 5, 1)), atol=1e-6)

    def test_nonneg_output(self):
        pred = _uniform()
        ig = _plains_grid()
        ig[5, 5] = MOUNTAIN
        ig[10, 10] = OCEAN
        result = apply_structural_zeros(pred.copy(), ig)
        assert (result >= 0).all()


# ---------------------------------------------------------------------------
# observation_overlay
# ---------------------------------------------------------------------------

class TestObservationOverlay:
    def test_no_observations_unchanged(self):
        pred = _uniform()
        result = observation_overlay(pred.copy(), [])
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((GRID_SIZE, GRID_SIZE)), atol=1e-5)

    def test_output_shape(self):
        pred = _uniform()
        result = observation_overlay(pred.copy(), [_obs()])
        assert result.shape == pred.shape

    def test_sums_to_one(self):
        pred = _uniform()
        obs = [_obs(0, 0, 5, 5, 1)]
        result = observation_overlay(pred.copy(), obs)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((GRID_SIZE, GRID_SIZE)), atol=1e-5)

    def test_nonneg(self):
        pred = _uniform()
        result = observation_overlay(pred.copy(), [_obs(0, 0, 5, 5, 1)])
        assert (result >= 0).all()

    def test_strong_observation_updates_probability(self):
        # With tau=1, a single settlement observation should increase class-1 probability
        pred = _uniform()
        obs = [_obs(0, 0, 1, 1, 1)]  # 1×1 observation of settlement (code=1)
        result = observation_overlay(pred.copy(), obs, tau=1)
        # Class 1 probability at observed cell should be higher than uniform 1/6
        assert result[0, 0, 1] > 1.0 / N_CLASSES

    def test_ocean_code_maps_to_class0(self):
        pred = _uniform()
        obs = [_obs(0, 0, 1, 1, OCEAN)]
        result = observation_overlay(pred.copy(), obs, tau=1)
        assert result[0, 0, 0] > 1.0 / N_CLASSES

    def test_multiple_observations(self):
        pred = _uniform()
        obs = [_obs(0, 0, 3, 3, 1), _obs(5, 5, 3, 3, 4)]
        result = observation_overlay(pred.copy(), obs, tau=5)
        assert result.shape == pred.shape
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((GRID_SIZE, GRID_SIZE)), atol=1e-5)


# ---------------------------------------------------------------------------
# compute_obs_ll
# ---------------------------------------------------------------------------

class TestComputeObsLl:
    def test_no_observations_gives_zero(self):
        pred = _uniform()
        result = compute_obs_ll(pred, [])
        assert result == 0.0

    def test_returns_float(self):
        pred = _uniform()
        result = compute_obs_ll(pred, [_obs()])
        assert isinstance(result, float)

    def test_nonpositive_log_likelihood(self):
        # Log-likelihood is always <= 0 (probabilities <= 1, log(p) <= 0)
        pred = _uniform()
        result = compute_obs_ll(pred, [_obs(0, 0, 3, 3, 1)])
        assert result <= 0

    def test_high_confidence_pred_gives_higher_ll(self):
        # Prediction that matches observation (class 1) should give better LL
        # than uniform when we observe settlement cells
        pred_conf = np.full((GRID_SIZE, GRID_SIZE, N_CLASSES), 0.01, dtype=np.float64)
        pred_conf[:, :, 1] = 0.95  # high confidence on class 1
        pred_conf /= pred_conf.sum(axis=-1, keepdims=True)

        pred_unif = _uniform()
        obs = [_obs(0, 0, 3, 3, 1)]  # observe settlements

        ll_conf = compute_obs_ll(pred_conf, obs)
        ll_unif = compute_obs_ll(pred_unif, obs)
        assert ll_conf > ll_unif

    def test_out_of_bounds_observation_ignored(self):
        # If viewport + grid exceeds 40x40, those cells are silently skipped
        pred = _uniform()
        obs = [_obs(38, 38, 5, 5, 1)]  # overflows 40x40 boundary
        result = compute_obs_ll(pred, obs)
        # Should not raise
        assert isinstance(result, float)

    def test_ocean_code_maps_to_class0(self):
        pred = np.zeros((GRID_SIZE, GRID_SIZE, N_CLASSES), dtype=np.float64)
        pred[:, :, 0] = 1.0  # all mass on class 0
        obs = [_obs(0, 0, 2, 2, OCEAN)]
        result = compute_obs_ll(pred, obs)
        # pred has high prob on class 0 for ocean obs → high LL
        assert result > compute_obs_ll(_uniform(), obs)

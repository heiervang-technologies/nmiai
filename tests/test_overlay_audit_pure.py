"""Pure-function tests for astar-island overlay_audit_script.py.

Covers: cell_code_to_class, entropy, kl_divergence, score_prediction,
        count_observations, overlay_dirichlet, overlay_mle.

All pure functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

from overlay_audit_script import (
    cell_code_to_class,
    entropy,
    kl_divergence,
    score_prediction,
    count_observations,
    overlay_dirichlet,
    overlay_mle,
)

N_CLASSES = 6
OCEAN, MOUNTAIN = 10, 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform(h: int = 5, w: int = 5) -> np.ndarray:
    return np.full((h, w, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float64)


def _plains_grid(h: int = 5, w: int = 5) -> np.ndarray:
    return np.full((h, w), 11, dtype=np.int32)


def _obs(vx: int = 0, vy: int = 0, code: int = 1, h: int = 2, w: int = 2) -> dict:
    return {
        "viewport_x": vx,
        "viewport_y": vy,
        "grid": [[code] * w for _ in range(h)],
    }


# ---------------------------------------------------------------------------
# cell_code_to_class
# ---------------------------------------------------------------------------

class TestCellCodeToClass:
    def test_settlement(self):
        assert cell_code_to_class(1) == 1

    def test_port(self):
        assert cell_code_to_class(2) == 2

    def test_forest(self):
        assert cell_code_to_class(4) == 4

    def test_mountain(self):
        assert cell_code_to_class(5) == 5

    def test_empty_codes_give_0(self):
        assert cell_code_to_class(0) == 0
        assert cell_code_to_class(11) == 0

    def test_ocean_gives_0(self):
        assert cell_code_to_class(10) == 0

    def test_unknown_code(self):
        # Should return 0 or handle gracefully
        result = cell_code_to_class(99)
        assert isinstance(result, int)


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_max_entropy(self):
        p = np.full((4, 4, N_CLASSES), 1.0 / N_CLASSES)
        result = entropy(p)
        expected = np.log2(N_CLASSES)
        np.testing.assert_allclose(result, np.full((4, 4), expected), atol=1e-5)

    def test_deterministic_zero_entropy(self):
        p = np.zeros((4, 4, N_CLASSES))
        p[:, :, 0] = 1.0
        result = entropy(p)
        np.testing.assert_allclose(result, np.zeros((4, 4)), atol=1e-4)

    def test_nonneg(self):
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(N_CLASSES), size=(4, 4))
        assert (entropy(p) >= -1e-9).all()

    def test_output_shape(self):
        p = _uniform(3, 4)
        assert entropy(p).shape == (3, 4)


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------

class TestKlDivergence:
    def test_identical_gives_zero(self):
        p = _uniform()
        result = kl_divergence(p, p)
        np.testing.assert_allclose(result, np.zeros((5, 5)), atol=1e-6)

    def test_nonneg(self):
        rng = np.random.default_rng(42)
        p = _uniform()
        q = rng.dirichlet(np.ones(N_CLASSES), size=(5, 5))
        assert (kl_divergence(p, q) >= -1e-9).all()

    def test_output_shape(self):
        p = _uniform(3, 4)
        assert kl_divergence(p, p).shape == (3, 4)


# ---------------------------------------------------------------------------
# score_prediction
# ---------------------------------------------------------------------------

class TestScorePrediction:
    def test_identical_gives_zero(self):
        p = _uniform()
        result = score_prediction(p, p.copy())
        assert abs(result) < 1e-4

    def test_nonneg(self):
        rng = np.random.default_rng(42)
        p = _uniform()
        q = rng.dirichlet(np.ones(N_CLASSES), size=(5, 5))
        result = score_prediction(p, q)
        assert result >= -1e-9

    def test_returns_float(self):
        p = _uniform()
        assert isinstance(score_prediction(p, p.copy()), float)

    def test_all_static_gives_zero(self):
        # Deterministic gt → entropy=0 → dynamic mask empty → 0.0
        p = np.zeros((4, 4, N_CLASSES))
        p[:, :, 0] = 1.0
        result = score_prediction(p, _uniform(4, 4))
        assert result == 0.0


# ---------------------------------------------------------------------------
# count_observations
# ---------------------------------------------------------------------------

class TestCountObservations:
    def test_empty_obs(self):
        pred = _uniform()
        counts, obs_count = count_observations(pred, [])
        np.testing.assert_allclose(counts, np.zeros_like(pred))
        np.testing.assert_allclose(obs_count, np.zeros((5, 5)))

    def test_obs_increments_class(self):
        pred = _uniform()
        obs = [_obs(0, 0, 1, 1, 1)]  # settlement (class 1) at (0,0)
        counts, obs_count = count_observations(pred, obs)
        assert counts[0, 0, 1] == 1.0
        assert obs_count[0, 0] == 1.0

    def test_multiple_obs_accumulate(self):
        pred = _uniform()
        obs = [_obs(0, 0, 1, 1, 1), _obs(0, 0, 1, 1, 1)]
        counts, obs_count = count_observations(pred, obs)
        assert counts[0, 0, 1] == 2.0
        assert obs_count[0, 0] == 2.0

    def test_out_of_bounds_skipped(self):
        pred = _uniform()
        obs = [_obs(3, 3, 1, 5, 5)]  # overflows 5×5 grid
        counts, obs_count = count_observations(pred, obs)
        # Should not raise
        assert isinstance(obs_count, np.ndarray)


# ---------------------------------------------------------------------------
# overlay_dirichlet
# ---------------------------------------------------------------------------

class TestOverlayDirichlet:
    def test_no_obs_unchanged(self):
        pred = _uniform()
        ig = _plains_grid()
        result = overlay_dirichlet(pred.copy(), ig, [], tau=5)
        np.testing.assert_allclose(result, pred, atol=1e-6)

    def test_output_shape(self):
        pred = _uniform()
        ig = _plains_grid()
        result = overlay_dirichlet(pred.copy(), ig, [_obs()], tau=5)
        assert result.shape == pred.shape

    def test_sums_to_one(self):
        pred = _uniform()
        ig = _plains_grid()
        obs = [_obs(0, 0, 1, 2, 2), _obs(0, 0, 1, 2, 2), _obs(0, 0, 1, 2, 2)]
        result = overlay_dirichlet(pred.copy(), ig, obs, tau=5)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((5, 5)), atol=1e-5)

    def test_nonneg(self):
        pred = _uniform()
        ig = _plains_grid()
        obs = [_obs()] * 5
        result = overlay_dirichlet(pred.copy(), ig, obs, tau=5)
        assert (result >= 0).all()

    def test_min_samples_gate(self):
        # With min_samples=3, fewer than 3 obs should leave cell unchanged
        pred = _uniform()
        ig = _plains_grid()
        # Only 2 observations — below min_samples=3
        obs = [_obs(0, 0, 1, 1, 1), _obs(0, 0, 1, 1, 1)]
        result = overlay_dirichlet(pred.copy(), ig, obs, tau=5, min_samples=3)
        np.testing.assert_allclose(result[0, 0], pred[0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# overlay_mle
# ---------------------------------------------------------------------------

class TestOverlayMle:
    def test_no_obs_unchanged(self):
        pred = _uniform()
        ig = _plains_grid()
        result = overlay_mle(pred.copy(), ig, [])
        np.testing.assert_allclose(result, pred, atol=1e-6)

    def test_output_shape(self):
        pred = _uniform()
        ig = _plains_grid()
        result = overlay_mle(pred.copy(), ig, [_obs()])
        assert result.shape == pred.shape

    def test_sums_to_one(self):
        pred = _uniform()
        ig = _plains_grid()
        obs = [_obs(0, 0, 1, 1, 1)] * 4
        result = overlay_mle(pred.copy(), ig, obs)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((5, 5)), atol=1e-5)

    def test_nonneg(self):
        pred = _uniform()
        ig = _plains_grid()
        result = overlay_mle(pred.copy(), ig, [_obs()] * 4)
        assert (result >= 0).all()

    def test_many_settlement_obs_concentrates_class1(self):
        pred = _uniform()
        ig = _plains_grid()
        # 5 settlement observations at (0,0)
        obs = [_obs(0, 0, 1, 1, 1)] * 5
        result = overlay_mle(pred.copy(), ig, obs, min_samples=3)
        assert result[0, 0, 1] > 1.0 / N_CLASSES

"""Pure-function tests for astar-island/round_optimizer.py and combined_predictor.py.

Covers:
  round_optimizer  : kl_divergence, entropy, compute_wkl
  combined_predictor: cell_code_to_class, detect_survival

All pure — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Mock heavy / side-effectful dependencies before import
_rp_mock = MagicMock()
sys.modules.setdefault("requests", MagicMock())
sys.modules.setdefault("regime_predictor", _rp_mock)
sys.modules.setdefault("spatial_model", MagicMock())
# scipy is available but combined_predictor imports it; mock to stay pure
_scipy_mock = MagicMock()
sys.modules.setdefault("scipy", _scipy_mock)
sys.modules.setdefault("scipy.ndimage", _scipy_mock.ndimage)

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

import round_optimizer as ro


# ---------------------------------------------------------------------------
# round_optimizer.kl_divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    """KL(p || q) summed over the class axis (axis=2)."""

    def _uniform(self, h=2, w=2):
        arr = np.ones((h, w, 6)) / 6
        return arr

    def test_identical_distributions_zero(self):
        p = self._uniform()
        result = ro.kl_divergence(p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_shape_is_hw(self):
        p = self._uniform(3, 4)
        result = ro.kl_divergence(p, p)
        assert result.shape == (3, 4)

    def test_nonneg(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(6), size=(5, 5))
        q = rng.dirichlet(np.ones(6), size=(5, 5))
        result = ro.kl_divergence(p, q)
        assert (result >= 0).all()

    def test_asymmetric(self):
        p = np.array([[[0.9, 0.02, 0.02, 0.02, 0.02, 0.02]]])
        q = np.array([[[1/6, 1/6, 1/6, 1/6, 1/6, 1/6]]])
        kl_pq = ro.kl_divergence(p, q)
        kl_qp = ro.kl_divergence(q, p)
        assert not np.isclose(kl_pq, kl_qp)

    def test_concentrated_larger_than_uniform(self):
        p_conc = np.array([[[0.98, 0.004, 0.004, 0.004, 0.004, 0.004]]])
        q_unif = np.ones((1, 1, 6)) / 6
        assert ro.kl_divergence(p_conc, q_unif)[0, 0] > 1.0


# ---------------------------------------------------------------------------
# round_optimizer.entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_maximum(self):
        p = np.ones((1, 1, 6)) / 6
        result = ro.entropy(p)
        np.testing.assert_allclose(result, np.log2(6), atol=1e-9)

    def test_deterministic_near_zero(self):
        # entropy() floors p at 1e-10 before summing, so residuals from
        # the other 5 zero classes contribute ~5 * 1e-10 * log2(1e-10) ≈ 1.7e-8.
        p = np.zeros((1, 1, 6))
        p[0, 0, 0] = 1.0
        result = ro.entropy(p)
        np.testing.assert_allclose(result, 0.0, atol=1e-7)

    def test_shape_is_hw(self):
        p = np.ones((3, 5, 6)) / 6
        result = ro.entropy(p)
        assert result.shape == (3, 5)

    def test_nonneg(self):
        p = np.ones((4, 4, 6)) / 6
        assert (ro.entropy(p) >= 0).all()

    def test_more_diffuse_higher_entropy(self):
        p_conc = np.array([[[0.9, 0.02, 0.02, 0.02, 0.02, 0.02]]])
        p_unif = np.ones((1, 1, 6)) / 6
        assert ro.entropy(p_unif)[0, 0] > ro.entropy(p_conc)[0, 0]

    def test_two_class_half_is_one_bit(self):
        p = np.zeros((1, 1, 6))
        p[0, 0, 0] = 0.5
        p[0, 0, 1] = 0.5
        result = ro.entropy(p)
        np.testing.assert_allclose(result[0, 0], 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# round_optimizer.compute_wkl
# ---------------------------------------------------------------------------

class TestComputeWKL:
    def _make_grid(self, h=5, w=5):
        """Simple uniform gt, perfect pred."""
        arr = np.ones((h, w, 6)) / 6
        return arr

    def test_perfect_prediction_zero(self):
        gt = self._make_grid()
        pred = self._make_grid()
        result = ro.compute_wkl(gt, pred)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_returns_float(self):
        gt = self._make_grid()
        pred = self._make_grid()
        assert isinstance(ro.compute_wkl(gt, pred), float)

    def test_nonneg(self):
        rng = np.random.default_rng(42)
        gt = rng.dirichlet(np.ones(6), size=(4, 4))
        pred = rng.dirichlet(np.ones(6), size=(4, 4))
        assert ro.compute_wkl(gt, pred) >= 0.0

    def test_deterministic_gt_returns_zero(self):
        # All-deterministic gt → H=0 → no dynamic cells → returns 0.0
        gt = np.zeros((4, 4, 6))
        gt[:, :, 0] = 1.0
        pred = np.ones((4, 4, 6)) / 6
        assert ro.compute_wkl(gt, pred) == pytest.approx(0.0)

    def test_bad_pred_worse_than_perfect(self):
        gt = np.ones((3, 3, 6)) / 6
        pred_good = np.ones((3, 3, 6)) / 6
        pred_bad = np.zeros((3, 3, 6))
        pred_bad[:, :, 0] = 1.0
        assert ro.compute_wkl(gt, pred_bad) > ro.compute_wkl(gt, pred_good)


# ---------------------------------------------------------------------------
# combined_predictor: cell_code_to_class
# ---------------------------------------------------------------------------

# Import combined_predictor separately (needs scipy mock patches applied above)
# scipy.ndimage functions are called at import-time via the module body;
# the MagicMock means all calls return MagicMock objects, which is fine since
# we only test the pure helper functions here.
try:
    import combined_predictor as cp
    _CP_AVAILABLE = True
except Exception:
    _CP_AVAILABLE = False


@pytest.mark.skipif(not _CP_AVAILABLE, reason="combined_predictor not importable")
class TestCombinedCellCodeToClass:
    @pytest.mark.parametrize("code,expected", [
        (0, 0), (10, 0), (11, 0),
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    ])
    def test_known_codes(self, code, expected):
        assert cp.cell_code_to_class(code) == expected

    def test_unknown_returns_zero(self):
        assert cp.cell_code_to_class(99) == 0


@pytest.mark.skipif(not _CP_AVAILABLE, reason="combined_predictor not importable")
class TestDetectSurvival:
    def _make_obs(self, vx, vy, cells):
        """Create a minimal observation dict with a 1×1 viewport."""
        return {"viewport_x": vx, "viewport_y": vy, "grid": [[cells]]}

    def test_no_observations_returns_default(self):
        ig = [[1, 0], [0, 0]]
        assert cp.detect_survival(ig, []) == pytest.approx(0.3)

    def test_settlement_survived(self):
        # 1x1 grid with a settlement at (0,0)
        ig = [[1]]
        obs = [self._make_obs(0, 0, 1)]  # cell=1 at (0,0) → survived
        result = cp.detect_survival(ig, obs)
        assert result == pytest.approx(1.0)

    def test_settlement_did_not_survive(self):
        ig = [[1]]
        obs = [self._make_obs(0, 0, 0)]  # cell=0 → not settlement
        result = cp.detect_survival(ig, obs)
        assert result == pytest.approx(0.0)

    def test_partial_survival(self):
        # 2x1 grid: settlements at (0,0) and (0,1)
        ig = [[1, 1]]
        # first viewport covers (0,0) → survived; second covers (0,1) → not survived
        obs = [
            {"viewport_x": 0, "viewport_y": 0, "grid": [[1, 0]]},
        ]
        result = cp.detect_survival(ig, obs)
        assert result == pytest.approx(0.5)

    def test_non_settlement_cells_ignored(self):
        # initial grid has settlement at (0,0) only
        ig = [[1, 4], [0, 0]]
        # observation shows settlement survived, forest cell also present
        obs = [{"viewport_x": 0, "viewport_y": 0, "grid": [[1, 4], [0, 0]]}]
        result = cp.detect_survival(ig, obs)
        assert result == pytest.approx(1.0)

    def test_no_settlement_in_viewport_returns_default(self):
        # settlement at (1,1) but observation only covers (0,0)
        ig = [[0, 0], [0, 1]]
        obs = [self._make_obs(0, 0, 0)]
        result = cp.detect_survival(ig, obs)
        assert result == pytest.approx(0.3)

    def test_result_in_range(self):
        ig = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        obs = [{"viewport_x": 0, "viewport_y": 0, "grid": [[1, 0, 1], [0, 1, 0], [1, 0, 1]]}]
        result = cp.detect_survival(ig, obs)
        assert 0.0 <= result <= 1.0

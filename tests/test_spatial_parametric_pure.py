"""Pure-function tests for spatial_model.py and parametric_predictor.py.

Covers:
  spatial_model: extract_features, predict_logodds
  parametric_predictor: exp_decay, _cell_type, classify_round, _eval_curve

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

import spatial_model as sm
import parametric_predictor as pp


GRID_SIZE = 40
N_CLASSES = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_grid(size: int = GRID_SIZE, code: int = 11) -> np.ndarray:
    g = np.full((size, size), code, dtype=np.int32)
    g[5, 5] = 1    # settlement
    g[5, 6] = 2    # port
    g[10, 10] = 4  # forest
    g[0, 0] = 10   # ocean
    g[-1, -1] = 5  # mountain
    return g


# ---------------------------------------------------------------------------
# spatial_model.extract_features
# ---------------------------------------------------------------------------

class TestExtractFeatures:
    EXPECTED_KEYS = {
        "ds", "do", "df", "dc", "dm", "coast",
        "ns", "nf", "no", "nm",
        "sd5", "sd9", "sd15", "fd5", "fd9", "od5",
        "de",
    }

    def test_returns_dict_and_grid(self):
        feats, ig = sm.extract_features(_uniform_grid())
        assert isinstance(feats, dict)
        assert isinstance(ig, np.ndarray)

    def test_required_keys(self):
        feats, _ = sm.extract_features(_uniform_grid())
        assert self.EXPECTED_KEYS.issubset(feats.keys())

    def test_shapes_match_grid(self):
        g = _uniform_grid()
        feats, _ = sm.extract_features(g)
        for key, arr in feats.items():
            assert arr.shape == (GRID_SIZE, GRID_SIZE), f"Wrong shape for {key}"

    def test_coast_nonzero_near_ocean(self):
        g = _uniform_grid()
        feats, _ = sm.extract_features(g)
        assert feats["coast"].any()

    def test_dist_settle_zero_at_settlement(self):
        g = _uniform_grid()
        feats, _ = sm.extract_features(g)
        assert feats["ds"][5, 5] == pytest.approx(0.0)

    def test_dist_ocean_zero_at_ocean(self):
        g = _uniform_grid()
        feats, _ = sm.extract_features(g)
        assert feats["do"][0, 0] == pytest.approx(0.0)

    def test_all_ocean_grid_no_settle_dist(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 10, dtype=np.int32)
        feats, _ = sm.extract_features(g)
        # No settlement — dist_settle should be max fill (40)
        assert (feats["ds"] == 40.0).all()

    def test_densities_nonneg(self):
        g = _uniform_grid()
        feats, _ = sm.extract_features(g)
        for key in ("sd5", "sd9", "sd15", "fd5", "fd9", "od5"):
            assert (feats[key] >= 0).all()

    def test_dist_edge_zero_at_border(self):
        g = _uniform_grid()
        feats, _ = sm.extract_features(g)
        # dist_edge: min distance to any border → 0 on all edges
        assert feats["de"][0, :].max() == pytest.approx(0.0)
        assert feats["de"][:, 0].max() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# spatial_model.predict_logodds
# ---------------------------------------------------------------------------

class TestPredictLogodds:
    def test_sums_to_one(self):
        n_feat = 5
        n_samples = 10
        X = np.random.rand(n_samples, n_feat)
        W = np.random.rand(n_feat, N_CLASSES - 1)
        probs = sm.predict_logodds(X, W)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_shape(self):
        X = np.random.rand(8, 5)
        W = np.random.rand(5, N_CLASSES - 1)
        probs = sm.predict_logodds(X, W)
        assert probs.shape == (8, N_CLASSES)

    def test_nonneg(self):
        X = np.random.rand(6, 4)
        W = np.random.rand(4, N_CLASSES - 1)
        probs = sm.predict_logodds(X, W)
        assert (probs >= 0).all()

    def test_zero_weights_uniform(self):
        X = np.ones((5, 3))
        W = np.zeros((3, N_CLASSES - 1))
        probs = sm.predict_logodds(X, W)
        # All logits = 0 → uniform over N_CLASSES
        np.testing.assert_allclose(probs, 1.0 / N_CLASSES, atol=1e-6)

    def test_large_positive_weight_biases_classes(self):
        X = np.ones((3, 1))
        W = np.zeros((1, N_CLASSES - 1))
        W[0, 0] = 100.0  # class 1 gets huge logit
        probs = sm.predict_logodds(X, W)
        # Class 1 should dominate
        assert probs[0, 1] > 0.99


# ---------------------------------------------------------------------------
# parametric_predictor.exp_decay
# ---------------------------------------------------------------------------

class TestExpDecay:
    def test_at_zero_returns_a(self):
        assert pp.exp_decay(0.0, 2.0, 0.5) == pytest.approx(2.0)

    def test_decreases_with_distance(self):
        assert pp.exp_decay(1.0, 1.0, 0.5) < pp.exp_decay(0.0, 1.0, 0.5)

    def test_negative_a_negative_result(self):
        result = pp.exp_decay(0.0, -1.0, 0.5)
        assert result == pytest.approx(-1.0)

    def test_large_distance_approaches_zero(self):
        assert abs(pp.exp_decay(1000.0, 1.0, 1.0)) < 1e-10


# ---------------------------------------------------------------------------
# parametric_predictor._cell_type
# ---------------------------------------------------------------------------

class TestCellType:
    def test_settlement_is_S(self):
        assert pp._cell_type(pp.SETTLEMENT) == "S"

    def test_port_is_P(self):
        assert pp._cell_type(pp.PORT) == "P"

    def test_forest_is_F(self):
        assert pp._cell_type(4) == "F"  # FOREST = 4

    def test_plains_is_L(self):
        assert pp._cell_type(11) == "L"

    def test_unknown_is_L(self):
        assert pp._cell_type(99) == "L"


# ---------------------------------------------------------------------------
# parametric_predictor.classify_round
# ---------------------------------------------------------------------------

class TestClassifyRound:
    def _make_round_data(self, settle_prob: float, size: int = 10) -> dict:
        """Minimal round_data dict for classify_round."""
        ig = np.full((size, size), 11, dtype=np.int32)
        ig[5, 5] = 1  # settlement at center
        gt = np.zeros((size, size, N_CLASSES), dtype=np.float64)
        gt[:, :, 0] = 1.0 - settle_prob
        gt[:, :, 1] = settle_prob
        return {0: {"initial_grid": ig, "ground_truth": gt}}

    def test_low_frontier_rate_harsh(self):
        # Very low settlement probability → harsh
        rd = self._make_round_data(settle_prob=0.01)
        result = pp.classify_round(rd)
        assert result == "harsh"

    def test_high_frontier_rate_prosperous(self):
        # Very high settlement probability → prosperous
        rd = self._make_round_data(settle_prob=0.5)
        result = pp.classify_round(rd)
        assert result == "prosperous"

    def test_mid_frontier_rate_moderate(self):
        # Mid settlement probability → moderate
        rd = self._make_round_data(settle_prob=0.12)
        result = pp.classify_round(rd)
        assert result == "moderate"

    def test_returns_string(self):
        rd = self._make_round_data(settle_prob=0.1)
        assert isinstance(pp.classify_round(rd), str)


# ---------------------------------------------------------------------------
# parametric_predictor._eval_curve
# ---------------------------------------------------------------------------

class TestEvalCurve:
    def test_at_zero_is_A_plus_C(self):
        A, B, C = 0.5, 0.3, 0.1
        result = pp._eval_curve((A, B, C), 0.0)
        assert result == pytest.approx(A + C)

    def test_large_distance_approaches_C(self):
        A, B, C = 1.0, 2.0, 0.05
        result = pp._eval_curve((A, B, C), 100.0)
        assert result == pytest.approx(C, abs=1e-6)

    def test_decreasing_with_positive_A_and_B(self):
        params = (1.0, 0.5, 0.0)
        assert pp._eval_curve(params, 1.0) < pp._eval_curve(params, 0.0)

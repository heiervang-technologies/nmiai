"""Tests for tasks/astar-island/predictor.py — pure helper functions.

Covers: encode_initial_type, feature_matrix_from_maps, entropy_weight,
        weighted_kl_divergence, tau_from_prior.
All pure numpy functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from predictor import (
    encode_initial_type,
    entropy_weight,
    extract_feature_maps,
    feature_matrix_from_maps,
    tau_from_prior,
    weighted_kl_divergence,
)


# ---------------------------------------------------------------------------
# encode_initial_type
# ---------------------------------------------------------------------------

class TestEncodeInitialType:
    def test_empty_and_plains_map_to_0(self):
        init = np.array([[0, 11]], dtype=np.int32)
        result = encode_initial_type(init)
        np.testing.assert_array_equal(result, [[0, 0]])

    def test_settlement_maps_to_1(self):
        init = np.array([[1]], dtype=np.int32)
        assert encode_initial_type(init)[0, 0] == 1

    def test_port_maps_to_2(self):
        init = np.array([[2]], dtype=np.int32)
        assert encode_initial_type(init)[0, 0] == 2

    def test_forest_maps_to_4(self):
        init = np.array([[4]], dtype=np.int32)
        assert encode_initial_type(init)[0, 0] == 4

    def test_mountain_maps_to_5(self):
        init = np.array([[5]], dtype=np.int32)
        assert encode_initial_type(init)[0, 0] == 5

    def test_ocean_maps_to_6(self):
        init = np.array([[10]], dtype=np.int32)
        assert encode_initial_type(init)[0, 0] == 6

    def test_output_shape_matches_input(self):
        init = np.full((3, 4), 11, dtype=np.int32)
        result = encode_initial_type(init)
        assert result.shape == (3, 4)

    def test_output_dtype_int32(self):
        init = np.ones((2, 2), dtype=np.int32)
        result = encode_initial_type(init)
        assert result.dtype == np.int32

    def test_mixed_grid(self):
        init = np.array([[0, 1, 2], [4, 5, 10]], dtype=np.int32)
        result = encode_initial_type(init)
        expected = np.array([[0, 1, 2], [4, 5, 6]], dtype=np.int32)
        np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# feature_matrix_from_maps
# ---------------------------------------------------------------------------

class TestFeatureMatrixFromMaps:
    def _simple_maps(self, h=4, w=4):
        g = np.full((h, w), 11, dtype=np.int32)
        g[1, 1] = 1  # settlement
        return extract_feature_maps(g)

    def test_returns_ndarray(self):
        maps = self._simple_maps()
        result = feature_matrix_from_maps(maps)
        assert isinstance(result, np.ndarray)

    def test_shape_h_w_7_features(self):
        maps = self._simple_maps(h=4, w=5)
        result = feature_matrix_from_maps(maps)
        assert result.shape == (4, 5, 7)

    def test_contains_init_type(self):
        maps = self._simple_maps()
        result = feature_matrix_from_maps(maps)
        # First feature should match init_type
        np.testing.assert_array_equal(result[:, :, 0], maps["init_type"])


# ---------------------------------------------------------------------------
# entropy_weight
# ---------------------------------------------------------------------------

class TestEntropyWeight:
    def test_uniform_6_class_has_highest_weight(self):
        y = np.array([[1.0 / 6.0] * 6])
        result = entropy_weight(y)
        assert result[0] > 0.25

    def test_deterministic_gives_floor_weight(self):
        y = np.zeros((1, 6))
        y[0, 0] = 1.0
        result = entropy_weight(y)
        # Entropy=0 → weight = 0.25 + 0 = 0.25
        assert result[0] == pytest.approx(0.25, abs=1e-5)

    def test_weight_always_at_least_0_25(self):
        y = np.random.dirichlet([1.0] * 6, size=5)
        result = entropy_weight(y)
        assert (result >= 0.25).all()

    def test_shape_matches_input_rows(self):
        y = np.ones((4, 6)) / 6.0
        result = entropy_weight(y)
        assert result.shape == (4,)

    def test_higher_entropy_gives_higher_weight(self):
        uniform = np.array([[1.0 / 6.0] * 6])
        peaked = np.array([[0.9, 0.02, 0.02, 0.02, 0.02, 0.02]])
        w_uniform = entropy_weight(uniform)[0]
        w_peaked = entropy_weight(peaked)[0]
        assert w_uniform > w_peaked


# ---------------------------------------------------------------------------
# tau_from_prior
# ---------------------------------------------------------------------------

class TestTauFromPrior:
    def test_ocean_init_type_6_returns_100(self):
        prior = np.array([1.0 / 6.0] * 6)
        result = tau_from_prior(prior, init_type=6)
        assert result == pytest.approx(100.0)

    def test_mountain_init_type_5_returns_100(self):
        prior = np.array([1.0 / 6.0] * 6)
        result = tau_from_prior(prior, init_type=5)
        assert result == pytest.approx(100.0)

    def test_uniform_prior_gives_high_tau(self):
        prior = np.array([1.0 / 6.0] * 6)
        result = tau_from_prior(prior, init_type=0)
        # uniform → max entropy → tau = 1 + 4 * log(6)/log(6) = 5.0
        assert result == pytest.approx(5.0, abs=1e-4)

    def test_deterministic_prior_gives_min_tau(self):
        prior = np.zeros(6)
        prior[0] = 1.0
        result = tau_from_prior(prior, init_type=0)
        # entropy=0 → tau = 1.0
        assert result == pytest.approx(1.0, abs=1e-4)

    def test_tau_between_1_and_5_for_normal_types(self):
        prior = np.array([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        result = tau_from_prior(prior, init_type=1)
        assert 1.0 <= result <= 5.0 + 1e-5


# ---------------------------------------------------------------------------
# weighted_kl_divergence
# ---------------------------------------------------------------------------

class TestWeightedKlDivergence:
    def _uniform(self, h=3, w=3, n=6):
        return np.full((h, w, n), 1.0 / n)

    def test_identical_near_zero(self):
        p = self._uniform()
        result = weighted_kl_divergence(p, p)
        assert abs(result) < 1e-6

    def test_returns_float(self):
        p = self._uniform()
        result = weighted_kl_divergence(p, p)
        assert isinstance(result, float)

    def test_non_negative(self):
        p = self._uniform()
        q = np.random.dirichlet([1.0] * 6, size=(3, 3))
        result = weighted_kl_divergence(p, q)
        assert result >= 0.0

    def test_larger_for_more_different(self):
        p = self._uniform()
        # q1 close to p
        q1 = p.copy()
        q1[:, :, 0] += 0.05
        q1 /= q1.sum(axis=2, keepdims=True)
        # q2 very different
        q2 = np.zeros_like(p)
        q2[:, :, 0] = 1.0

        d1 = weighted_kl_divergence(p, q1)
        d2 = weighted_kl_divergence(p, q2)
        assert d2 > d1

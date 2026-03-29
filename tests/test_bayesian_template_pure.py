"""Tests for tasks/astar-island/bayesian_template_predictor.py — pure helpers.

Covers: cell_bucket, cell_code_to_class, posterior_weights,
kl_divergence, entropy.
All pure functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from bayesian_template_predictor import (
    N_CLASSES,
    cell_bucket,
    cell_code_to_class,
    entropy,
    kl_divergence,
    posterior_weights,
)


# ---------------------------------------------------------------------------
# cell_bucket
# ---------------------------------------------------------------------------

class TestCellBucket:
    def test_ocean_gives_none(self):
        assert cell_bucket(10, 3.0, 0, 0, False) is None

    def test_mountain_gives_none(self):
        assert cell_bucket(5, 3.0, 0, 0, False) is None

    def test_returns_4_tuple(self):
        result = cell_bucket(11, 3.0, 0, 0, False)
        assert len(result) == 4

    def test_settlement_type_is_S(self):
        fine, _, _, _ = cell_bucket(1, 0.0, 0, 0, False)
        assert fine[0] == "S"

    def test_port_type_is_P(self):
        fine, _, _, _ = cell_bucket(2, 0.0, 0, 0, False)
        assert fine[0] == "P"

    def test_forest_type_is_F(self):
        fine, _, _, _ = cell_bucket(4, 5.0, 0, 0, False)
        assert fine[0] == "F"

    def test_plains_type_is_L(self):
        fine, _, _, _ = cell_bucket(11, 5.0, 0, 0, False)
        assert fine[0] == "L"

    def test_distance_clamped_to_15(self):
        fine, _, _, _ = cell_bucket(11, 100.0, 0, 0, False)
        assert fine[1] == 15

    def test_distance_at_zero(self):
        fine, _, _, _ = cell_bucket(11, 0.0, 0, 0, False)
        assert fine[1] == 0

    def test_coast_flag_propagated(self):
        fine1, _, _, _ = cell_bucket(11, 3.0, 2, 0, True)
        fine2, _, _, _ = cell_bucket(11, 3.0, 2, 0, False)
        assert fine1[-1] == 1
        assert fine2[-1] == 0

    def test_n_ocean_clamped_at_4(self):
        fine, _, _, _ = cell_bucket(11, 3.0, 10, 0, False)
        assert fine[2] == 4

    def test_broad_key_dist_clamped_at_8(self):
        _, _, _, broad = cell_bucket(11, 15.0, 0, 0, False)
        assert broad[1] <= 8

    def test_ruin_type_is_L(self):
        # Ruin (3) has no special type → defaults to "L"
        fine, _, _, _ = cell_bucket(3, 4.0, 0, 0, False)
        assert fine[0] == "L"


# ---------------------------------------------------------------------------
# cell_code_to_class
# ---------------------------------------------------------------------------

class TestCellCodeToClass:
    def test_empty_0(self):
        assert cell_code_to_class(0) == 0

    def test_ocean_0(self):
        assert cell_code_to_class(10) == 0

    def test_plains_0(self):
        assert cell_code_to_class(11) == 0

    def test_settlement_1(self):
        assert cell_code_to_class(1) == 1

    def test_port_2(self):
        assert cell_code_to_class(2) == 2

    def test_ruin_3(self):
        assert cell_code_to_class(3) == 3

    def test_forest_4(self):
        assert cell_code_to_class(4) == 4

    def test_mountain_5(self):
        assert cell_code_to_class(5) == 5

    def test_unknown_0(self):
        assert cell_code_to_class(99) == 0


# ---------------------------------------------------------------------------
# posterior_weights
# ---------------------------------------------------------------------------

class TestPosteriorWeights:
    def test_uniform_prior_equal_log_liks(self):
        log_liks = {1: 0.0, 2: 0.0, 3: 0.0}
        result = posterior_weights(log_liks, [1, 2, 3])
        for rn in [1, 2, 3]:
            assert abs(result[rn] - 1.0 / 3) < 1e-6

    def test_dominant_template_gets_high_weight(self):
        log_liks = {1: 0.0, 2: -100.0, 3: -100.0}
        result = posterior_weights(log_liks, [1, 2, 3])
        assert result[1] > 0.99

    def test_weights_sum_to_one(self):
        log_liks = {1: -2.0, 2: -1.0, 3: -0.5}
        result = posterior_weights(log_liks, [1, 2, 3])
        assert abs(sum(result.values()) - 1.0) < 1e-6

    def test_returns_dict(self):
        log_liks = {1: 0.0}
        result = posterior_weights(log_liks, [1])
        assert isinstance(result, dict)

    def test_single_template_gets_weight_one(self):
        log_liks = {5: -3.0}
        result = posterior_weights(log_liks, [5])
        assert abs(result[5] - 1.0) < 1e-6

    def test_all_float_values(self):
        log_liks = {1: -1.0, 2: -2.0}
        result = posterior_weights(log_liks, [1, 2])
        for v in result.values():
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------

class TestKlDivergence:
    def test_identical_near_zero(self):
        p = np.full((3, 3, N_CLASSES), 1.0 / N_CLASSES)
        result = kl_divergence(p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_shape_collapses_class_axis(self):
        p = np.full((4, 5, N_CLASSES), 1.0 / N_CLASSES)
        result = kl_divergence(p, p)
        assert result.shape == (4, 5)

    def test_non_negative(self):
        p = np.full((2, 2, N_CLASSES), 1.0 / N_CLASSES)
        q = np.random.dirichlet(np.ones(N_CLASSES), size=(2, 2)).reshape(2, 2, N_CLASSES)
        result = kl_divergence(p, q)
        assert (result >= -1e-10).all()


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_6_class_log2_6(self):
        p = np.full((2, 2, N_CLASSES), 1.0 / N_CLASSES)
        result = entropy(p)
        np.testing.assert_allclose(result, np.log2(N_CLASSES), atol=1e-4)

    def test_deterministic_zero(self):
        p = np.zeros((2, 2, N_CLASSES))
        p[:, :, 0] = 1.0
        result = entropy(p)
        np.testing.assert_allclose(result, 0.0, atol=1e-4)

    def test_shape_collapses_class_axis(self):
        p = np.full((3, 4, N_CLASSES), 1.0 / N_CLASSES)
        result = entropy(p)
        assert result.shape == (3, 4)

    def test_non_negative(self):
        p = np.random.dirichlet(np.ones(N_CLASSES), size=(3, 3)).reshape(3, 3, N_CLASSES)
        result = entropy(p)
        assert (result >= -1e-10).all()

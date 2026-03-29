"""Tests for tasks/astar-island pure helpers in ensemble_predictor and parametric_predictor.

Covers:
- ensemble_predictor: floor_renorm, weighted_kl_divergence, cell_entropy, round_number_from_path
- parametric_predictor: exp_decay, _cell_type

All pure functions — no file I/O or GPU needed.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR)

from ensemble_predictor import (
    cell_entropy,
    floor_renorm,
    round_number_from_path,
    weighted_kl_divergence,
)
from parametric_predictor import _cell_type, exp_decay


# ---------------------------------------------------------------------------
# floor_renorm
# ---------------------------------------------------------------------------

class TestFloorRenorm:
    def test_returns_ndarray(self):
        p = np.array([0.5, 0.3, 0.2])
        result = floor_renorm(p)
        assert isinstance(result, np.ndarray)

    def test_sums_to_one_single_cell(self):
        p = np.array([[0.5, 0.3, 0.2]])
        result = floor_renorm(p)
        assert result.sum(axis=-1)[0] == pytest.approx(1.0)

    def test_zero_elements_raised_above_zero(self):
        # After flooring, renorm may push values below floor, but they'll be > 0
        p = np.array([[1.0, 0.0, 0.0]])
        result = floor_renorm(p, floor=0.01)
        assert result[0, 1] > 0.0
        assert result[0, 2] > 0.0

    def test_already_above_floor_unchanged_relative_order(self):
        p = np.array([[0.6, 0.3, 0.1]])
        result = floor_renorm(p, floor=0.01)
        assert result[0, 0] > result[0, 1] > result[0, 2]

    def test_all_values_above_floor_after_call(self):
        p = np.zeros((2, 4)) + 0.001
        result = floor_renorm(p, floor=0.05)
        assert np.all(result >= 0.05)

    def test_shape_preserved(self):
        p = np.ones((3, 4, 6)) / 6
        result = floor_renorm(p)
        assert result.shape == (3, 4, 6)


# ---------------------------------------------------------------------------
# weighted_kl_divergence
# ---------------------------------------------------------------------------

class TestWeightedKlDivergence:
    def test_returns_float(self):
        p = np.ones((2, 2, 4)) / 4
        result = weighted_kl_divergence(p, p)
        assert isinstance(result, float)

    def test_identical_distributions_near_zero(self):
        p = np.ones((3, 3, 6)) / 6
        result = weighted_kl_divergence(p, p)
        assert result == pytest.approx(0.0, abs=1e-8)

    def test_nonnegative(self):
        p = np.random.default_rng(1).dirichlet(np.ones(4), size=(2, 2))
        q = np.random.default_rng(2).dirichlet(np.ones(4), size=(2, 2))
        result = weighted_kl_divergence(p, q)
        assert result >= 0.0

    def test_more_different_higher_wkl(self):
        target = np.ones((2, 2, 4)) / 4
        slightly_off = np.full((2, 2, 4), 0.24)
        slightly_off[:, :, 0] = 0.28
        very_off = np.zeros((2, 2, 4))
        very_off[:, :, 0] = 1.0
        wkl_slight = weighted_kl_divergence(target, slightly_off)
        wkl_very = weighted_kl_divergence(target, very_off)
        assert wkl_very > wkl_slight


# ---------------------------------------------------------------------------
# cell_entropy
# ---------------------------------------------------------------------------

class TestCellEntropy:
    def test_returns_ndarray(self):
        p = np.ones((2, 3, 4)) / 4
        result = cell_entropy(p)
        assert isinstance(result, np.ndarray)

    def test_output_shape_hw(self):
        p = np.ones((4, 5, 6)) / 6
        result = cell_entropy(p)
        assert result.shape == (4, 5)

    def test_uniform_gives_positive_entropy(self):
        p = np.ones((1, 1, 4)) / 4
        result = cell_entropy(p)
        assert result[0, 0] > 0.0

    def test_deterministic_near_zero_entropy(self):
        p = np.zeros((1, 1, 4))
        p[0, 0, 0] = 1.0
        result = cell_entropy(p)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_nonnegative(self):
        p = np.random.default_rng(42).dirichlet(np.ones(6), size=(3, 3))
        result = cell_entropy(p)
        assert np.all(result >= 0.0)


# ---------------------------------------------------------------------------
# round_number_from_path
# ---------------------------------------------------------------------------

class TestRoundNumberFromPath:
    def test_basic_round_seed_path(self):
        p = Path("ground_truth/round3_seed7.json")
        assert round_number_from_path(p) == 3

    def test_round_12(self):
        p = Path("/some/dir/round12_seed0.json")
        assert round_number_from_path(p) == 12

    def test_returns_int(self):
        p = Path("round1_seed1.json")
        assert isinstance(round_number_from_path(p), int)


# ---------------------------------------------------------------------------
# exp_decay (parametric_predictor)
# ---------------------------------------------------------------------------

class TestExpDecay:
    def test_at_zero_distance_returns_a(self):
        result = exp_decay(0.0, a=2.0, b=0.5)
        assert result == pytest.approx(2.0)

    def test_decay_at_positive_distance(self):
        result = exp_decay(1.0, a=1.0, b=1.0)
        assert result == pytest.approx(math.exp(-1.0))

    def test_positive_result_for_positive_a(self):
        result = exp_decay(5.0, a=3.0, b=0.1)
        assert result > 0.0

    def test_larger_b_decays_faster(self):
        d = 2.0
        slow = exp_decay(d, a=1.0, b=0.1)
        fast = exp_decay(d, a=1.0, b=1.0)
        assert fast < slow

    def test_returns_float_like(self):
        result = exp_decay(1.0, a=1.0, b=1.0)
        assert isinstance(float(result), float)


# ---------------------------------------------------------------------------
# _cell_type (parametric_predictor)
# ---------------------------------------------------------------------------

class TestCellType:
    def test_settlement_is_S(self):
        assert _cell_type(1) == "S"

    def test_port_is_P(self):
        assert _cell_type(2) == "P"

    def test_forest_is_F(self):
        assert _cell_type(4) == "F"

    def test_plains_is_L(self):
        assert _cell_type(11) == "L"

    def test_empty_is_L(self):
        assert _cell_type(0) == "L"

    def test_unknown_is_L(self):
        assert _cell_type(99) == "L"

    def test_returns_string(self):
        assert isinstance(_cell_type(1), str)

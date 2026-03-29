"""Pure-function tests for astar-island calibrated_predictor.py.

Covers:
  entropy_bits         : Shannon entropy in bits
  temperature_scale    : softmax with temperature
  fit_temperature      : binary-search temperature to match entropy
  shrink_temperature   : Bayesian shrinkage toward T=1
  assign_bucket        : hierarchical fine→mid→coarse→type fallback

All pure functions — no file system, network, or GT data.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock neighborhood_predictor before importing calibrated_predictor
_nbr_mock = MagicMock()
_nbr_mock.N_CLASSES = 6
sys.modules.setdefault("neighborhood_predictor", _nbr_mock)

_ASTAR_DIR = str(
    Path(__file__).resolve().parent.parent / "tasks" / "astar-island"
)
sys.path.insert(0, _ASTAR_DIR)

from calibrated_predictor import (
    entropy_bits,
    temperature_scale,
    fit_temperature,
    shrink_temperature,
    assign_bucket,
)


# ---------------------------------------------------------------------------
# entropy_bits
# ---------------------------------------------------------------------------

class TestEntropyBits:
    def test_uniform_distribution(self):
        # Uniform over 6 classes → log2(6) bits
        prob = np.full(6, 1.0 / 6)
        result = entropy_bits(prob)
        assert result == pytest.approx(np.log2(6), abs=1e-6)

    def test_deterministic_near_zero(self):
        # One class = 1.0, rest = 0 → near 0 bits (floor at 1e-12)
        prob = np.zeros(6)
        prob[0] = 1.0
        result = entropy_bits(prob)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_two_class_uniform(self):
        # Uniform over 2 classes → 1.0 bit
        prob = np.array([0.5, 0.5])
        result = entropy_bits(prob)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        p = rng.dirichlet(np.ones(6))
        assert entropy_bits(p) >= 0.0

    def test_returns_float(self):
        prob = np.full(6, 1.0 / 6)
        result = entropy_bits(prob)
        assert isinstance(result, float)

    def test_clips_zeros(self):
        # Should not raise even with zero entries
        prob = np.array([1.0, 0.0, 0.0])
        result = entropy_bits(prob)
        assert np.isfinite(result)


# ---------------------------------------------------------------------------
# temperature_scale
# ---------------------------------------------------------------------------

class TestTemperatureScale:
    def test_t1_preserves_distribution(self):
        prob = np.array([0.5, 0.3, 0.2])
        out = temperature_scale(prob, 1.0)
        np.testing.assert_allclose(out, prob, atol=1e-6)

    def test_high_temperature_flattens(self):
        # High T → more uniform
        prob = np.array([0.9, 0.05, 0.05])
        out = temperature_scale(prob, 100.0)
        # All classes should be closer to each other
        assert np.max(out) - np.min(out) < np.max(prob) - np.min(prob)

    def test_low_temperature_sharpens(self):
        # Low T → more peaked
        prob = np.array([0.6, 0.3, 0.1])
        out = temperature_scale(prob, 0.1)
        assert out[0] > prob[0]

    def test_sums_to_one(self):
        prob = np.array([0.4, 0.3, 0.2, 0.1])
        out = temperature_scale(prob, 2.0)
        assert np.sum(out) == pytest.approx(1.0, abs=1e-6)

    def test_non_negative(self):
        prob = np.array([0.7, 0.2, 0.1])
        out = temperature_scale(prob, 0.5)
        assert np.all(out >= 0.0)

    def test_returns_array(self):
        prob = np.array([0.5, 0.5])
        out = temperature_scale(prob, 1.0)
        assert isinstance(out, np.ndarray)


# ---------------------------------------------------------------------------
# fit_temperature
# ---------------------------------------------------------------------------

class TestFitTemperature:
    def _uniform_prob(self, n=6):
        return np.full(n, 1.0 / n)

    def _peaked_prob(self):
        p = np.array([0.8, 0.1, 0.05, 0.025, 0.015, 0.01])
        return p

    def test_already_close_returns_one(self):
        prob = self._uniform_prob()
        target = entropy_bits(prob)
        t = fit_temperature(prob, target)
        assert t == pytest.approx(1.0, abs=1e-3)

    def test_higher_target_returns_t_gt_1(self):
        # Peaked distribution → need T > 1 to raise entropy
        prob = self._peaked_prob()
        current = entropy_bits(prob)
        target = current + 0.5
        t = fit_temperature(prob, target)
        assert t > 1.0

    def test_lower_target_returns_t_lt_1(self):
        # Uniform → need T < 1 to lower entropy
        prob = self._uniform_prob()
        current = entropy_bits(prob)
        target = current - 0.3
        t = fit_temperature(prob, target)
        assert t < 1.0

    def test_achieved_entropy_matches_target(self):
        prob = self._peaked_prob()
        current = entropy_bits(prob)
        target = current + 0.4
        t = fit_temperature(prob, target)
        achieved = entropy_bits(temperature_scale(prob, t))
        assert achieved == pytest.approx(target, abs=1e-3)

    def test_returns_positive_float(self):
        prob = self._uniform_prob()
        t = fit_temperature(prob, 1.0)
        assert isinstance(t, float)
        assert t > 0.0


# ---------------------------------------------------------------------------
# shrink_temperature
# ---------------------------------------------------------------------------

class TestShrinkTemperature:
    def test_zero_support_returns_one(self):
        # alpha = 0 / (0 + shrink) = 0 → exp(0) = 1
        result = shrink_temperature(raw_temperature=2.0, support=0, shrink=10)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_high_support_approaches_raw(self):
        # support >> shrink → alpha → 1 → exp(log(T)) = T
        result = shrink_temperature(raw_temperature=3.0, support=10000, shrink=1)
        assert result == pytest.approx(3.0, abs=0.01)

    def test_t1_unchanged(self):
        # T=1 → log(1)=0 → any alpha gives exp(0)=1
        result = shrink_temperature(raw_temperature=1.0, support=5, shrink=10)
        assert result == pytest.approx(1.0, abs=1e-6)

    def test_returns_float(self):
        result = shrink_temperature(2.0, 10, 5)
        assert isinstance(result, float)

    def test_positive_output(self):
        result = shrink_temperature(0.5, 3, 5)
        assert result > 0.0

    def test_partial_shrink(self):
        # alpha = 5 / (5+5) = 0.5 → exp(0.5 * log(T))
        import math
        T = 4.0
        shrink = 5
        support = 5
        expected = math.exp(0.5 * math.log(T))
        result = shrink_temperature(T, support, shrink)
        assert result == pytest.approx(expected, abs=1e-6)


# ---------------------------------------------------------------------------
# assign_bucket
# ---------------------------------------------------------------------------

class TestAssignBucket:
    def _make_tables(self):
        fine_table = {(0, 1, 2, 1, 2): np.full(6, 1.0 / 6)}
        mid_table = {(0, 1, 0, 2): np.full(6, 1.0 / 6)}
        coarse_table = {(0, 2): np.full(6, 1.0 / 6)}
        type_table = {(0,): np.full(6, 1.0 / 6)}
        return fine_table, mid_table, coarse_table, type_table

    def test_fine_match_returns_fine_level(self):
        ft, mt, ct, tt = self._make_tables()
        (level, key), prob = assign_bucket(0, 1, 2, 1, 2, ft, mt, ct, tt)
        assert level == "fine"

    def test_mid_fallback_when_no_fine(self):
        ft, mt, ct, tt = self._make_tables()
        # ns=0, no=1, db=2 → fine key (0,0,2,1,2) absent
        # mid key: ns_bin=0, no_bin=1 (no>0), db=2 → (0, 0, 1, 2)
        mt[(0, 0, 1, 2)] = np.full(6, 1.0 / 6)
        (level, key), prob = assign_bucket(0, 0, 2, 1, 2, ft, mt, ct, tt)
        assert level == "mid"

    def test_coarse_fallback(self):
        ft, mt, ct, tt = self._make_tables()
        # Keys that miss fine and mid
        (level, key), prob = assign_bucket(0, 3, 3, 2, 2, ft, mt, ct, tt)
        assert level == "coarse"

    def test_type_fallback(self):
        ft, mt, ct, tt = self._make_tables()
        # Remove coarse entry to force type fallback
        del ct[(0, 2)]
        (level, key), prob = assign_bucket(0, 3, 3, 2, 2, ft, mt, ct, tt)
        assert level == "type"

    def test_returns_probability_array(self):
        ft, mt, ct, tt = self._make_tables()
        (level, key), prob = assign_bucket(0, 1, 2, 1, 2, ft, mt, ct, tt)
        assert isinstance(prob, np.ndarray)
        assert len(prob) == 6

    def test_probability_sums_to_one(self):
        ft, mt, ct, tt = self._make_tables()
        (level, key), prob = assign_bucket(0, 1, 2, 1, 2, ft, mt, ct, tt)
        assert np.sum(prob) == pytest.approx(1.0, abs=1e-6)

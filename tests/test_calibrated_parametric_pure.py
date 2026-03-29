"""Tests for calibrated_predictor.py and parametric_predictor.py pure helpers.

Covers:
  - calibrated_predictor: entropy_bits, temperature_scale, fit_temperature,
      assign_bucket, shrink_temperature, choose_temperature
  - parametric_predictor: exp_decay, _cell_type, _eval_curve, compute_features
All pure functions — no file system or network access.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from calibrated_predictor import (
    assign_bucket,
    choose_temperature,
    entropy_bits,
    fit_temperature,
    shrink_temperature,
    temperature_scale,
)
from parametric_predictor import (
    _cell_type,
    _eval_curve,
    exp_decay,
)


# ---------------------------------------------------------------------------
# calibrated_predictor.entropy_bits
# ---------------------------------------------------------------------------

class TestEntropyBits:
    def test_uniform_6_class(self):
        prob = np.full(6, 1.0 / 6)
        result = entropy_bits(prob)
        assert abs(result - math.log2(6)) < 1e-4

    def test_deterministic_gives_zero(self):
        prob = np.zeros(6)
        prob[0] = 1.0
        result = entropy_bits(prob)
        assert abs(result) < 1e-4

    def test_2_class_uniform_gives_one_bit(self):
        prob = np.array([0.5, 0.5])
        assert abs(entropy_bits(prob) - 1.0) < 1e-4

    def test_returns_float(self):
        prob = np.full(6, 1.0 / 6)
        assert isinstance(entropy_bits(prob), float)

    def test_non_negative(self):
        prob = np.array([0.2, 0.3, 0.1, 0.15, 0.15, 0.1])
        assert entropy_bits(prob) >= 0.0


# ---------------------------------------------------------------------------
# calibrated_predictor.temperature_scale
# ---------------------------------------------------------------------------

class TestTemperatureScale:
    def _uniform(self):
        return np.full(6, 1.0 / 6)

    def test_temperature_1_preserves_dist(self):
        p = np.array([0.1, 0.2, 0.3, 0.15, 0.15, 0.1])
        result = temperature_scale(p, 1.0)
        np.testing.assert_allclose(result, p / p.sum(), atol=1e-5)

    def test_output_sums_to_one(self):
        p = self._uniform()
        result = temperature_scale(p, 2.0)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_high_temperature_flattens(self):
        p = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
        flat = temperature_scale(p, 20.0)
        sharp = temperature_scale(p, 0.1)
        assert entropy_bits(flat) > entropy_bits(sharp)

    def test_low_temperature_sharpens(self):
        p = np.full(6, 1.0 / 6)
        # Already uniform — low temp keeps it uniform
        result = temperature_scale(p, 0.5)
        np.testing.assert_allclose(result, p, atol=1e-5)

    def test_output_all_positive(self):
        p = np.array([0.5, 0.0, 0.3, 0.1, 0.05, 0.05])
        result = temperature_scale(p, 1.0)
        assert (result > 0).all()


# ---------------------------------------------------------------------------
# calibrated_predictor.fit_temperature
# ---------------------------------------------------------------------------

class TestFitTemperature:
    def test_already_matching_returns_one(self):
        prob = np.full(6, 1.0 / 6)
        target = entropy_bits(prob)
        result = fit_temperature(prob, target)
        assert abs(result - 1.0) < 1e-3

    def test_higher_target_gives_temperature_above_one(self):
        # Sharp distribution; we want higher entropy → temperature > 1
        prob = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
        high_target = entropy_bits(prob) + 0.5
        result = fit_temperature(prob, high_target)
        assert result > 1.0

    def test_lower_target_gives_temperature_below_one(self):
        # Flat distribution; we want lower entropy → temperature < 1
        prob = np.full(6, 1.0 / 6)
        low_target = entropy_bits(prob) - 0.5
        result = fit_temperature(prob, low_target)
        assert result < 1.0

    def test_returns_float(self):
        prob = np.full(6, 1.0 / 6)
        assert isinstance(fit_temperature(prob, 2.0), float)

    def test_scaled_entropy_near_target(self):
        prob = np.array([0.7, 0.1, 0.1, 0.05, 0.025, 0.025])
        target = 1.5
        t = fit_temperature(prob, target)
        scaled = temperature_scale(prob, t)
        assert abs(entropy_bits(scaled) - target) < 0.05


# ---------------------------------------------------------------------------
# calibrated_predictor.assign_bucket
# ---------------------------------------------------------------------------

class TestAssignBucket:
    def _tables(self):
        fine_table = {("S", 2, 1, 0, 2): np.full(6, 1.0 / 6)}
        mid_table = {("S", 1, 0, 2): np.full(6, 1.0 / 6)}
        coarse_table = {("S", 2): np.full(6, 1.0 / 6)}
        type_table = {("S",): np.full(6, 1.0 / 6)}
        return fine_table, mid_table, coarse_table, type_table

    def test_fine_key_hit(self):
        ft, mt, ct, tt = self._tables()
        bucket_id, prob = assign_bucket("S", 2, 1, 0, 2, ft, mt, ct, tt)
        assert bucket_id[0] == "fine"

    def test_mid_key_fallback(self):
        ft, mt, ct, tt = self._tables()
        # ns=1 maps to ns_bin=1, same as ns=2→ns_bin=1, coast(no>0)=0
        bucket_id, prob = assign_bucket("S", 1, 0, 0, 2, ft, mt, ct, tt)
        assert bucket_id[0] == "mid"

    def test_coarse_fallback(self):
        ft, mt, ct, tt = self._tables()
        ft.clear()
        mt.clear()
        bucket_id, prob = assign_bucket("S", 0, 0, 0, 2, ft, mt, ct, tt)
        assert bucket_id[0] == "coarse"

    def test_type_fallback(self):
        ft, mt, ct, tt = self._tables()
        ft.clear()
        mt.clear()
        ct.clear()
        bucket_id, prob = assign_bucket("S", 0, 0, 0, 2, ft, mt, ct, tt)
        assert bucket_id[0] == "type"

    def test_returns_tuple_and_array(self):
        ft, mt, ct, tt = self._tables()
        bucket_id, prob = assign_bucket("S", 2, 1, 0, 2, ft, mt, ct, tt)
        assert isinstance(bucket_id, tuple)
        assert isinstance(prob, np.ndarray)


# ---------------------------------------------------------------------------
# calibrated_predictor.shrink_temperature
# ---------------------------------------------------------------------------

class TestShrinkTemperature:
    def test_high_support_approaches_raw(self):
        # With support >> shrink, alpha → 1 → shrink_temp → raw_temp
        result = shrink_temperature(2.0, support=10000, shrink=1)
        assert abs(result - 2.0) < 0.01

    def test_low_support_shrinks_toward_one(self):
        # With support << shrink, alpha → 0 → shrink_temp → exp(0) = 1
        result = shrink_temperature(4.0, support=0, shrink=100)
        assert abs(result - 1.0) < 0.05

    def test_returns_float(self):
        assert isinstance(shrink_temperature(1.5, 20, 30), float)

    def test_raw_one_always_one(self):
        # exp(alpha * log(1)) = 1 regardless of support
        for support in [0, 5, 100]:
            assert abs(shrink_temperature(1.0, support, 30) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# calibrated_predictor.choose_temperature
# ---------------------------------------------------------------------------

class TestChooseTemperature:
    def _bucket_temps(self):
        return {
            "bucket": {("fine", ("S", 2, 1, 0, 2)): 1.5},
            "dist": {("fine", "S", 2): 1.3},
            "type": {("fine", "S"): 1.1},
            "level": {("fine",): 1.0},
        }

    def test_bucket_exact_match(self):
        bt = self._bucket_temps()
        bid = ("fine", ("S", 2, 1, 0, 2))
        assert choose_temperature(bid, bt) == 1.5

    def test_fallback_to_dist(self):
        bt = self._bucket_temps()
        # Use key not in bucket table but in dist table
        bid = ("fine", ("S", 0, 0, 0, 2))  # fine key that misses bucket exact
        bt["bucket"] = {}
        result = choose_temperature(bid, bt)
        assert result == 1.3

    def test_fallback_to_type(self):
        bt = self._bucket_temps()
        bt["bucket"] = {}
        bt["dist"] = {}
        bid = ("fine", ("S", 0, 0, 0, 2))
        result = choose_temperature(bid, bt)
        assert result == 1.1

    def test_fallback_to_level(self):
        bt = self._bucket_temps()
        bt["bucket"] = {}
        bt["dist"] = {}
        bt["type"] = {}
        bid = ("fine", ("S", 0, 0, 0, 2))
        result = choose_temperature(bid, bt)
        assert result == 1.0

    def test_missing_level_returns_default_one(self):
        bt = {"bucket": {}, "dist": {}, "type": {}, "level": {}}
        bid = ("coarse", ("F", 3))
        result = choose_temperature(bid, bt)
        assert result == 1.0


# ---------------------------------------------------------------------------
# parametric_predictor.exp_decay
# ---------------------------------------------------------------------------

class TestExpDecay:
    def test_at_zero_returns_a_plus_c(self):
        # exp_decay(0, a, b) = a*exp(0) + ... but signature is just a*exp(-b*d)
        # Looking at code: exp_decay(d, a, b) = a * exp(-b * d)
        assert abs(exp_decay(0, 3.0, 0.5) - 3.0) < 1e-9

    def test_decays_with_distance(self):
        assert exp_decay(5, 1.0, 0.2) < exp_decay(0, 1.0, 0.2)

    def test_zero_a_gives_zero(self):
        assert exp_decay(10, 0.0, 1.0) == 0.0

    def test_large_b_decays_fast(self):
        assert exp_decay(3, 1.0, 5.0) < 0.01

    def test_returns_numpy_compatible(self):
        d = np.array([0.0, 1.0, 2.0])
        result = exp_decay(d, 1.0, 0.5)
        assert result.shape == (3,)


# ---------------------------------------------------------------------------
# parametric_predictor._cell_type
# ---------------------------------------------------------------------------

class TestCellType:
    def test_settlement_gives_S(self):
        assert _cell_type(1) == "S"

    def test_port_gives_P(self):
        assert _cell_type(2) == "P"

    def test_forest_gives_F(self):
        assert _cell_type(4) == "F"

    def test_default_gives_L(self):
        for code in [0, 3, 5, 10, 11, 99]:
            assert _cell_type(code) == "L"


# ---------------------------------------------------------------------------
# parametric_predictor._eval_curve
# ---------------------------------------------------------------------------

class TestEvalCurve:
    def test_at_zero_returns_a_plus_c(self):
        params = (2.0, 0.5, 0.1)
        result = _eval_curve(params, 0)
        assert abs(result - 2.1) < 1e-9

    def test_decays_with_distance(self):
        params = (1.0, 0.3, 0.05)
        assert _eval_curve(params, 10) < _eval_curve(params, 0)

    def test_converges_to_c_at_infinity(self):
        params = (5.0, 2.0, 0.3)
        result = _eval_curve(params, 100)
        assert abs(result - 0.3) < 1e-3

    def test_zero_a_returns_c(self):
        params = (0.0, 1.0, 0.5)
        assert abs(_eval_curve(params, 5) - 0.5) < 1e-9

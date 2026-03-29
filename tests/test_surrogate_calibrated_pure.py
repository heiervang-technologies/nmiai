"""Pure-function tests for surrogate_mc.py and calibrated_predictor.py.

Covers:
  surrogate_mc: normalize, cell_code_to_class, cell_type_token, quantize_dist,
                quantize_neighbor_civ, base_bucket_keys, spatial_bucket_keys,
                apply_constraints, structured_fallback, kl_divergence, entropy,
                score_prediction, score_from_wkl
  calibrated_predictor: entropy_bits, temperature_scale, fit_temperature,
                        assign_bucket, shrink_temperature

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

from surrogate_mc import (
    normalize,
    cell_code_to_class,
    cell_type_token,
    quantize_dist,
    quantize_neighbor_civ,
    base_bucket_keys,
    spatial_bucket_keys,
    apply_constraints,
    structured_fallback,
    kl_divergence,
    entropy,
    score_prediction,
    score_from_wkl,
    FLOOR, N_CLASSES, OCEAN, MOUNTAIN, SETTLEMENT, PORT, FOREST, PLAINS,
    OCEAN_DIST, MOUNTAIN_DIST,
)

import calibrated_predictor as cp


# ---------------------------------------------------------------------------
# surrogate_mc.normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_sums_to_one(self):
        prob = np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0])
        result = normalize(prob)
        assert abs(result.sum() - 1.0) < 1e-9

    def test_floor_applied(self):
        # FLOOR is applied before normalization so all output entries are > 0
        prob = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = normalize(prob)
        assert (result > 0).all()

    def test_uniform_on_zero_input(self):
        prob = np.zeros(N_CLASSES)
        result = normalize(prob)
        assert abs(result.sum() - 1.0) < 1e-9

    def test_positive_values_preserved_relative(self):
        prob = np.array([2.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        result = normalize(prob)
        assert result[0] > result[1]


# ---------------------------------------------------------------------------
# surrogate_mc.cell_code_to_class
# ---------------------------------------------------------------------------

class TestCellCodeToClass:
    @pytest.mark.parametrize("code,expected", [
        (0, 0), (10, 0), (11, 0),  # ocean-like / empty
        (1, 1),   # settlement
        (2, 2),   # port
        (3, 3),   # ruin
        (4, 4),   # forest
        (5, 5),   # mountain
    ])
    def test_known_codes(self, code, expected):
        assert cell_code_to_class(code) == expected

    def test_unknown_code_returns_zero(self):
        assert cell_code_to_class(99) == 0


# ---------------------------------------------------------------------------
# surrogate_mc.cell_type_token
# ---------------------------------------------------------------------------

class TestCellTypeToken:
    def test_settlement_is_S(self):
        assert cell_type_token(SETTLEMENT) == "S"

    def test_port_is_P(self):
        assert cell_type_token(PORT) == "P"

    def test_forest_is_F(self):
        assert cell_type_token(FOREST) == "F"

    def test_plains_is_L(self):
        assert cell_type_token(PLAINS) == "L"

    def test_empty_is_L(self):
        assert cell_type_token(0) == "L"


# ---------------------------------------------------------------------------
# surrogate_mc.quantize_dist
# ---------------------------------------------------------------------------

class TestQuantizeDist:
    def test_zero(self):
        assert quantize_dist(0.0) == 0

    def test_rounds_down(self):
        assert quantize_dist(3.9) == 3

    def test_clamps_at_15(self):
        assert quantize_dist(100.0) == 15

    def test_clamps_at_0_for_negative(self):
        assert quantize_dist(-5.0) == 0

    def test_exact_integer(self):
        assert quantize_dist(7.0) == 7


# ---------------------------------------------------------------------------
# surrogate_mc.quantize_neighbor_civ
# ---------------------------------------------------------------------------

class TestQuantizeNeighborCiv:
    @pytest.mark.parametrize("n,expected", [
        (0, 0),
        (1, 1),
        (2, 2), (3, 2),
        (4, 3), (8, 3),
    ])
    def test_bins(self, n, expected):
        assert quantize_neighbor_civ(n) == expected


# ---------------------------------------------------------------------------
# surrogate_mc.base_bucket_keys
# ---------------------------------------------------------------------------

class TestBaseBucketKeys:
    def test_returns_none_for_ocean(self):
        assert base_bucket_keys(OCEAN, 5.0, 2, True) is None

    def test_returns_none_for_mountain(self):
        assert base_bucket_keys(MOUNTAIN, 5.0, 0, False) is None

    def test_returns_4_keys_for_land(self):
        keys = base_bucket_keys(SETTLEMENT, 3.0, 1, True)
        assert keys is not None
        assert len(keys) == 4

    def test_keys_first_is_finest(self):
        keys = base_bucket_keys(FOREST, 2.0, 2, False)
        # finest key has 4 components, broadest has 2
        assert len(keys[0]) > len(keys[-1])

    def test_coast_reflected_in_keys(self):
        keys_coast = base_bucket_keys(PLAINS, 2.0, 1, True)
        keys_no_coast = base_bucket_keys(PLAINS, 2.0, 1, False)
        assert keys_coast != keys_no_coast


# ---------------------------------------------------------------------------
# surrogate_mc.spatial_bucket_keys
# ---------------------------------------------------------------------------

class TestSpatialBucketKeys:
    def test_returns_none_for_ocean(self):
        assert spatial_bucket_keys(OCEAN, 3.0, 2, True, 2) is None

    def test_returns_4_keys_for_land(self):
        keys = spatial_bucket_keys(SETTLEMENT, 3.0, 1, True, 1)
        assert keys is not None
        assert len(keys) == 4

    def test_different_n_civ_gives_different_keys(self):
        keys0 = spatial_bucket_keys(PLAINS, 3.0, 1, False, 0)
        keys3 = spatial_bucket_keys(PLAINS, 3.0, 1, False, 3)
        assert keys0 != keys3


# ---------------------------------------------------------------------------
# surrogate_mc.apply_constraints
# ---------------------------------------------------------------------------

class TestApplyConstraints:
    def test_ocean_returns_ocean_dist(self):
        prob = np.ones(N_CLASSES, dtype=np.float64) / N_CLASSES
        result = apply_constraints(prob, OCEAN, 0.0, False)
        np.testing.assert_array_equal(result, OCEAN_DIST)

    def test_mountain_returns_mountain_dist(self):
        prob = np.ones(N_CLASSES, dtype=np.float64) / N_CLASSES
        result = apply_constraints(prob, MOUNTAIN, 0.0, False)
        np.testing.assert_array_equal(result, MOUNTAIN_DIST)

    def test_no_mountain_class_for_non_mountain(self):
        prob = np.ones(N_CLASSES, dtype=np.float64)
        result = apply_constraints(prob, PLAINS, 1.0, False)
        assert result[5] == 0.0 or result[5] < FLOOR + 1e-9

    def test_no_port_if_not_coastal(self):
        prob = np.ones(N_CLASSES, dtype=np.float64)
        result = apply_constraints(prob, PLAINS, 1.0, coast=False)
        assert result[2] < FLOOR + 1e-9

    def test_port_allowed_if_coastal(self):
        prob = np.array([0.1, 0.1, 0.5, 0.1, 0.1, 0.0])
        result = apply_constraints(prob, PLAINS, 1.0, coast=True)
        assert result[2] > 0.0

    def test_forest_far_from_civ_is_pure_forest(self):
        prob = np.ones(N_CLASSES, dtype=np.float64) / N_CLASSES
        result = apply_constraints(prob, FOREST, 15.0, False)
        assert result[4] == pytest.approx(1.0)

    def test_sums_to_one_for_regular_cell(self):
        prob = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        result = apply_constraints(prob, SETTLEMENT, 5.0, True)
        assert abs(result.sum() - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# surrogate_mc.structured_fallback
# ---------------------------------------------------------------------------

class TestStructuredFallback:
    def test_forest_far_returns_pure_forest(self):
        result = structured_fallback(FOREST, 15.0, False)
        assert result[4] == pytest.approx(1.0)

    def test_plains_far_returns_pure_empty(self):
        result = structured_fallback(PLAINS, 15.0, False)
        assert result[0] == pytest.approx(1.0)

    def test_returns_normalized_array(self):
        for code in (FOREST, SETTLEMENT, PORT, PLAINS):
            result = structured_fallback(code, 3.0, False)
            assert abs(result.sum() - 1.0) < 1e-6

    def test_coast_changes_port_probability(self):
        coast = structured_fallback(PORT, 3.0, True)
        no_coast = structured_fallback(PORT, 3.0, False)
        assert coast[2] > no_coast[2]


# ---------------------------------------------------------------------------
# surrogate_mc.kl_divergence
# ---------------------------------------------------------------------------

class TestKLDivergence:
    def test_identical_distributions_zero(self):
        p = np.ones((3, 3, N_CLASSES)) / N_CLASSES
        result = kl_divergence(p, p.copy())
        np.testing.assert_allclose(result, np.zeros((3, 3)), atol=1e-6)

    def test_nonneg(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(N_CLASSES), size=(4, 4))
        q = rng.dirichlet(np.ones(N_CLASSES), size=(4, 4))
        assert (kl_divergence(p, q) >= 0).all()

    def test_shape(self):
        p = np.ones((5, 7, N_CLASSES)) / N_CLASSES
        q = np.ones((5, 7, N_CLASSES)) / N_CLASSES
        assert kl_divergence(p, q).shape == (5, 7)


# ---------------------------------------------------------------------------
# surrogate_mc.entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_max_entropy(self):
        p = np.ones((2, 2, N_CLASSES)) / N_CLASSES
        H = entropy(p)
        expected = np.log2(N_CLASSES)
        np.testing.assert_allclose(H, np.full((2, 2), expected), atol=1e-6)

    def test_degenerate_near_zero(self):
        p = np.zeros((2, 2, N_CLASSES))
        p[:, :, 0] = 1.0
        H = entropy(p)
        assert (H < 0.01).all()

    def test_shape(self):
        p = np.ones((3, 4, N_CLASSES)) / N_CLASSES
        assert entropy(p).shape == (3, 4)


# ---------------------------------------------------------------------------
# surrogate_mc.score_prediction
# ---------------------------------------------------------------------------

class TestScorePrediction:
    def test_identical_returns_zero_wkl(self):
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.ones(N_CLASSES), size=(5, 5))
        result = score_prediction(p, p.copy())
        assert result["wkl"] == pytest.approx(0.0, abs=1e-5)

    def test_returns_required_keys(self):
        rng = np.random.default_rng(2)
        p = rng.dirichlet(np.ones(N_CLASSES), size=(4, 4))
        q = rng.dirichlet(np.ones(N_CLASSES), size=(4, 4))
        result = score_prediction(p, q)
        for key in ("wkl", "kl", "dynamic_cells"):
            assert key in result

    def test_dynamic_cells_nonneg(self):
        rng = np.random.default_rng(3)
        p = rng.dirichlet(np.ones(N_CLASSES), size=(4, 4))
        q = rng.dirichlet(np.ones(N_CLASSES), size=(4, 4))
        assert score_prediction(p, q)["dynamic_cells"] >= 0

    def test_all_static_gt_returns_zero_wkl(self):
        # Degenerate p (all mass on one class) → entropy=0 → no dynamic cells
        p = np.zeros((4, 4, N_CLASSES))
        p[:, :, 0] = 1.0
        q = np.ones((4, 4, N_CLASSES)) / N_CLASSES
        result = score_prediction(p, q)
        assert result["wkl"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# surrogate_mc.score_from_wkl
# ---------------------------------------------------------------------------

class TestScoreFromWkl:
    def test_zero_wkl_gives_100(self):
        assert score_from_wkl(0.0) == pytest.approx(100.0)

    def test_wkl_one_gives_zero(self):
        assert score_from_wkl(1.0) == pytest.approx(0.0)

    def test_large_wkl_clamped_at_zero(self):
        assert score_from_wkl(5.0) == pytest.approx(0.0)

    def test_mid_wkl(self):
        assert score_from_wkl(0.5) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# calibrated_predictor.entropy_bits
# ---------------------------------------------------------------------------

class TestEntropyBits:
    def test_uniform_max(self):
        prob = np.ones(N_CLASSES) / N_CLASSES
        result = cp.entropy_bits(prob)
        assert result == pytest.approx(np.log2(N_CLASSES), rel=1e-5)

    def test_degenerate_near_zero(self):
        prob = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = cp.entropy_bits(prob)
        assert result == pytest.approx(0.0, abs=1e-4)

    def test_returns_float(self):
        assert isinstance(cp.entropy_bits(np.ones(N_CLASSES) / N_CLASSES), float)


# ---------------------------------------------------------------------------
# calibrated_predictor.temperature_scale
# ---------------------------------------------------------------------------

class TestTemperatureScale:
    def test_temperature_1_identity(self):
        prob = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        result = cp.temperature_scale(prob, 1.0)
        np.testing.assert_allclose(result, prob / prob.sum(), atol=1e-6)

    def test_high_temperature_flattens(self):
        prob = np.array([0.9, 0.05, 0.01, 0.01, 0.01, 0.02])
        flat = cp.temperature_scale(prob, 10.0)
        assert cp.entropy_bits(flat) > cp.entropy_bits(prob)

    def test_low_temperature_sharpens(self):
        prob = np.array([0.4, 0.25, 0.15, 0.1, 0.05, 0.05])
        sharp = cp.temperature_scale(prob, 0.3)
        assert cp.entropy_bits(sharp) < cp.entropy_bits(prob)

    def test_sums_to_one(self):
        prob = np.array([0.3, 0.2, 0.2, 0.1, 0.1, 0.1])
        result = cp.temperature_scale(prob, 2.0)
        assert abs(result.sum() - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# calibrated_predictor.fit_temperature
# ---------------------------------------------------------------------------

class TestFitTemperature:
    def test_returns_float(self):
        prob = np.array([0.4, 0.25, 0.15, 0.1, 0.05, 0.05])
        t = cp.fit_temperature(prob, 2.0)
        assert isinstance(t, float)

    def test_target_entropy_achieved(self):
        prob = np.array([0.8, 0.1, 0.04, 0.03, 0.02, 0.01])
        target = 1.5
        t = cp.fit_temperature(prob, target)
        scaled = cp.temperature_scale(prob, t)
        assert cp.entropy_bits(scaled) == pytest.approx(target, abs=0.05)

    def test_already_at_target_returns_one(self):
        prob = np.ones(N_CLASSES) / N_CLASSES
        target = cp.entropy_bits(prob)
        t = cp.fit_temperature(prob, target)
        assert t == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# calibrated_predictor.shrink_temperature
# ---------------------------------------------------------------------------

class TestShrinkTemperature:
    def test_returns_float(self):
        result = cp.shrink_temperature(2.0, 50, 100)
        assert isinstance(result, float)

    def test_high_support_approaches_raw(self):
        raw = 2.0
        result = cp.shrink_temperature(raw, support=10000, shrink=1)
        assert result == pytest.approx(raw, rel=0.01)

    def test_low_support_shrinks_toward_one(self):
        # With support → 0, alpha → 0, exp(0) = 1
        result = cp.shrink_temperature(3.0, support=0, shrink=100)
        assert result == pytest.approx(1.0, abs=0.01)

    def test_nonneg(self):
        assert cp.shrink_temperature(1.5, 10, 20) > 0


# ---------------------------------------------------------------------------
# calibrated_predictor.assign_bucket
# ---------------------------------------------------------------------------

class TestAssignBucket:
    def _make_tables(self):
        prob = np.ones(N_CLASSES, dtype=float) / N_CLASSES
        # Fine table only has ("S", 2, 1, 1, 0)
        fine_table = {("S", 2, 1, 1, 0): prob}
        mid_table = {("S", 0, 1, 0): prob}
        coarse_table = {("S", 2): prob}
        type_table = {("S",): prob}
        return fine_table, mid_table, coarse_table, type_table

    def test_returns_tuple_bucket_and_prob(self):
        ft, mt, ct, tt = self._make_tables()
        bucket_id, result_prob = cp.assign_bucket("S", 0, 1, 1, 0, ft, mt, ct, tt)
        assert isinstance(bucket_id, tuple)
        assert isinstance(result_prob, np.ndarray)
        assert result_prob.shape == (N_CLASSES,)

    def test_fine_match_returned(self):
        ft, mt, ct, tt = self._make_tables()
        bucket_id, _ = cp.assign_bucket("S", 2, 1, 1, 0, ft, mt, ct, tt)
        assert bucket_id[0] == "fine"

    def test_type_fallback(self):
        # No fine/mid/coarse match → falls through to type_table
        ft, mt, ct, tt = {}, {}, {}, {("L",): np.ones(N_CLASSES) / N_CLASSES}
        bucket_id, _ = cp.assign_bucket("L", 99, 0, False, 0, ft, mt, ct, tt)
        assert bucket_id[0] == "type"

"""Pure-function tests for bayesian_template_predictor.py and ensemble_predictor.py.

Covers:
  bayesian_template_predictor: cell_bucket, cell_code_to_class, lookup,
                                posterior_weights, kl_divergence, entropy
  ensemble_predictor: floor_renorm, weighted_kl_divergence, cell_entropy,
                      round_number_from_path

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

# ensemble_predictor imports neighborhood_predictor and predictor at module level
for _m in ("neighborhood_predictor", "requests", "sklearn",
           "sklearn.ensemble", "sklearn.linear_model"):
    sys.modules.setdefault(_m, MagicMock())

import bayesian_template_predictor as btp
import ensemble_predictor as ep


N_CLASSES = 6


# ---------------------------------------------------------------------------
# bayesian_template_predictor.cell_bucket
# ---------------------------------------------------------------------------

class TestCellBucket:
    def test_ocean_returns_none(self):
        assert btp.cell_bucket(btp.OCEAN, 5.0, 2, 1, True) is None

    def test_mountain_returns_none(self):
        assert btp.cell_bucket(btp.MOUNTAIN, 3.0, 0, 0, False) is None

    def test_returns_4_keys_for_settlement(self):
        keys = btp.cell_bucket(btp.SETTLEMENT, 3.0, 1, 2, True)
        assert keys is not None
        assert len(keys) == 4

    def test_token_settlement_is_S(self):
        fine_key = btp.cell_bucket(btp.SETTLEMENT, 2.0, 0, 0, False)[0]
        assert fine_key[0] == "S"

    def test_token_port_is_P(self):
        fine_key = btp.cell_bucket(btp.PORT, 2.0, 1, 0, True)[0]
        assert fine_key[0] == "P"

    def test_token_forest_is_F(self):
        fine_key = btp.cell_bucket(btp.FOREST, 5.0, 0, 0, False)[0]
        assert fine_key[0] == "F"

    def test_token_plains_is_L(self):
        fine_key = btp.cell_bucket(btp.PLAINS, 5.0, 0, 0, False)[0]
        assert fine_key[0] == "L"

    def test_dist_clamped_at_15(self):
        keys_far = btp.cell_bucket(btp.PLAINS, 100.0, 0, 0, False)
        keys_15 = btp.cell_bucket(btp.PLAINS, 15.0, 0, 0, False)
        assert keys_far[0] == keys_15[0]

    def test_coast_reflected_in_fine_key(self):
        keys_coast = btp.cell_bucket(btp.PLAINS, 2.0, 1, 0, True)
        keys_no_coast = btp.cell_bucket(btp.PLAINS, 2.0, 1, 0, False)
        assert keys_coast[0] != keys_no_coast[0]

    def test_broad_key_clips_dist_at_8(self):
        keys_near = btp.cell_bucket(btp.SETTLEMENT, 2.0, 0, 0, False)
        keys_far = btp.cell_bucket(btp.SETTLEMENT, 10.0, 0, 0, False)
        # broad key (index 3) should be the same when dist ≥ 8
        assert keys_far[3] == btp.cell_bucket(btp.SETTLEMENT, 8.0, 0, 0, False)[3]


# ---------------------------------------------------------------------------
# bayesian_template_predictor.cell_code_to_class
# ---------------------------------------------------------------------------

class TestCellCodeToClass:
    @pytest.mark.parametrize("code,expected", [
        (0, 0), (10, 0), (11, 0),
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    ])
    def test_known_codes(self, code, expected):
        assert btp.cell_code_to_class(code) == expected

    def test_unknown_returns_zero(self):
        assert btp.cell_code_to_class(99) == 0


# ---------------------------------------------------------------------------
# bayesian_template_predictor.lookup
# ---------------------------------------------------------------------------

class TestLookup:
    def _make_tables_and_counts(self):
        prob = np.ones(N_CLASSES) / N_CLASSES
        tables = [
            {("S", 2, 0, 1, 1): prob},    # fine level 0
            {("S", 2, 1, 0): prob},         # mid level 1
            {("S", 2, 0): prob},             # coarse level 2
            {("S", 2): prob},                # broad level 3
        ]
        counts = [
            {("S", 2, 0, 1, 1): 10},
            {("S", 2, 1, 0): 5},
            {("S", 2, 0): 3},
            {("S", 2): 1},
        ]
        return tables, counts

    def test_returns_match_at_fine_level(self):
        tables, counts = self._make_tables_and_counts()
        keys = (("S", 2, 0, 1, 1), ("S", 2, 1, 0), ("S", 2, 0), ("S", 2))
        prob, count, level = btp.lookup(tables, counts, keys)
        assert level == 0
        assert count == 10

    def test_falls_back_to_broad(self):
        tables, counts = self._make_tables_and_counts()
        # Only broad matches
        tables[0] = {}
        tables[1] = {}
        tables[2] = {}
        keys = (("S", 99, 0, 0, 0), ("S", 99, 0, 0), ("S", 99, 0), ("S", 2))
        prob, count, level = btp.lookup(tables, counts, keys)
        assert level == 3

    def test_no_match_returns_none(self):
        tables = [{}, {}, {}, {}]
        counts = [{}, {}, {}, {}]
        keys = (("X",), ("X",), ("X",), ("X",))
        prob, count, level = btp.lookup(tables, counts, keys)
        assert prob is None

    def test_support_threshold_enforced(self):
        prob = np.ones(N_CLASSES) / N_CLASSES
        # fine level has count=3 which is < default min_count=5
        tables = [{("S", 2, 0, 1, 1): prob}, {("S", 2, 1, 0): prob}, {}, {}]
        counts = [{("S", 2, 0, 1, 1): 3}, {("S", 2, 1, 0): 10}, {}, {}]
        keys = (("S", 2, 0, 1, 1), ("S", 2, 1, 0), ("X",), ("X",))
        _, _, level = btp.lookup(tables, counts, keys, min_counts=(5, 8, 10, 1))
        assert level == 1  # fine skipped, mid passes


# ---------------------------------------------------------------------------
# bayesian_template_predictor.posterior_weights
# ---------------------------------------------------------------------------

class TestPosteriorWeights:
    def test_returns_dict_keyed_by_rounds(self):
        log_liks = {1: -10.0, 2: -8.0, 3: -12.0}
        weights = btp.posterior_weights(log_liks, [1, 2, 3])
        assert set(weights.keys()) == {1, 2, 3}

    def test_weights_sum_to_one(self):
        log_liks = {1: -10.0, 2: -8.0, 3: -12.0}
        weights = btp.posterior_weights(log_liks, [1, 2, 3])
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_higher_log_lik_gets_higher_weight(self):
        log_liks = {1: -5.0, 2: -20.0}
        weights = btp.posterior_weights(log_liks, [1, 2])
        assert weights[1] > weights[2]

    def test_identical_log_liks_uniform(self):
        log_liks = {1: -5.0, 2: -5.0, 3: -5.0}
        weights = btp.posterior_weights(log_liks, [1, 2, 3])
        for v in weights.values():
            assert v == pytest.approx(1 / 3, rel=1e-5)

    def test_single_round_weight_one(self):
        log_liks = {7: -999.0}
        weights = btp.posterior_weights(log_liks, [7])
        assert weights[7] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# bayesian_template_predictor.kl_divergence + entropy
# ---------------------------------------------------------------------------

class TestBtpKLAndEntropy:
    def test_kl_identical_zero(self):
        p = np.ones((3, 3, N_CLASSES)) / N_CLASSES
        kl = btp.kl_divergence(p, p.copy())
        np.testing.assert_allclose(kl, 0.0, atol=1e-6)

    def test_kl_nonneg(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(N_CLASSES), size=(4, 4))
        q = rng.dirichlet(np.ones(N_CLASSES), size=(4, 4))
        assert (btp.kl_divergence(p, q) >= 0).all()

    def test_entropy_uniform_max(self):
        p = np.ones((2, 2, N_CLASSES)) / N_CLASSES
        H = btp.entropy(p)
        np.testing.assert_allclose(H, np.log2(N_CLASSES), atol=1e-6)

    def test_entropy_shape(self):
        p = np.ones((5, 7, N_CLASSES)) / N_CLASSES
        assert btp.entropy(p).shape == (5, 7)


# ---------------------------------------------------------------------------
# ensemble_predictor.floor_renorm
# ---------------------------------------------------------------------------

class TestFloorRenorm:
    def test_sums_to_one(self):
        pred = np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        result = ep.floor_renorm(pred)
        assert abs(result.sum() - 1.0) < 1e-9

    def test_floor_applied(self):
        # Floor is applied before normalization so all entries are > 0
        pred = np.array([[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
        result = ep.floor_renorm(pred, floor=0.05)
        assert (result > 0).all()

    def test_shape_preserved(self):
        pred = np.random.rand(4, 4, N_CLASSES)
        pred /= pred.sum(axis=-1, keepdims=True)
        assert ep.floor_renorm(pred).shape == (4, 4, N_CLASSES)

    def test_relative_order_preserved(self):
        pred = np.array([[[0.9, 0.05, 0.01, 0.01, 0.01, 0.02]]])
        result = ep.floor_renorm(pred, floor=0.001)
        assert result[0, 0, 0] > result[0, 0, 1]


# ---------------------------------------------------------------------------
# ensemble_predictor.weighted_kl_divergence
# ---------------------------------------------------------------------------

class TestWeightedKLDivergence:
    def test_identical_returns_zero(self):
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.ones(N_CLASSES), size=(5, 5))
        assert ep.weighted_kl_divergence(p, p.copy()) == pytest.approx(0.0, abs=1e-6)

    def test_returns_scalar_float(self):
        rng = np.random.default_rng(2)
        p = rng.dirichlet(np.ones(N_CLASSES), size=(4, 4))
        q = rng.dirichlet(np.ones(N_CLASSES), size=(4, 4))
        result = ep.weighted_kl_divergence(p, q)
        assert isinstance(result, float)

    def test_nonneg(self):
        rng = np.random.default_rng(3)
        p = rng.dirichlet(np.ones(N_CLASSES), size=(5, 5))
        q = rng.dirichlet(np.ones(N_CLASSES), size=(5, 5))
        assert ep.weighted_kl_divergence(p, q) >= 0


# ---------------------------------------------------------------------------
# ensemble_predictor.cell_entropy
# ---------------------------------------------------------------------------

class TestCellEntropy:
    def test_uniform_max_entropy(self):
        p = np.ones((3, 3, N_CLASSES)) / N_CLASSES
        H = ep.cell_entropy(p)
        np.testing.assert_allclose(H, np.log(N_CLASSES), atol=1e-6)

    def test_shape(self):
        p = np.ones((5, 7, N_CLASSES)) / N_CLASSES
        assert ep.cell_entropy(p).shape == (5, 7)

    def test_degenerate_near_zero(self):
        p = np.zeros((2, 2, N_CLASSES))
        p[:, :, 0] = 1.0
        H = ep.cell_entropy(p)
        assert (H < 0.01).all()


# ---------------------------------------------------------------------------
# ensemble_predictor.round_number_from_path
# ---------------------------------------------------------------------------

class TestRoundNumberFromPath:
    def test_extracts_round_number(self):
        p = Path("ground_truth/round12_seed3.json")
        assert ep.round_number_from_path(p) == 12

    def test_single_digit_round(self):
        p = Path("round3_seed0.json")
        assert ep.round_number_from_path(p) == 3

    def test_large_round_number(self):
        p = Path("/some/dir/round100_seed7.json")
        assert ep.round_number_from_path(p) == 100

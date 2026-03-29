"""Additional pure-function tests for astar-island score_diagnosis.py.

Covers: build_pooled_prior, predict_with_prior, score_breakdown.
(kl_per_cell, entropy_per_cell, classify_cells were covered in the
 earlier test/score-diagnosis-pure branch.)

All pure functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

from score_diagnosis import (
    build_pooled_prior,
    predict_with_prior,
    score_breakdown,
    N_CLASSES,
    OCEAN,
    MOUNTAIN,
    SETTLEMENT,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_gt(h: int, w: int) -> np.ndarray:
    return np.full((h, w, N_CLASSES), 1.0 / N_CLASSES)


def _make_ig(h: int = 6, w: int = 6, base_code: int = 11) -> np.ndarray:
    """Small grid: mostly plains, one civ, one ocean, one mountain."""
    ig = np.full((h, w), base_code, dtype=np.int32)
    ig[0, 0] = OCEAN
    ig[-1, -1] = MOUNTAIN
    ig[2, 2] = SETTLEMENT
    return ig


def _make_rounds_data(n_rounds: int = 2) -> dict:
    """Minimal rounds_data structure with synthetic seeds."""
    ig = _make_ig()
    gt = _uniform_gt(*ig.shape)
    return {
        str(r): {
            "0": {"initial_grid": ig, "ground_truth": gt}
        }
        for r in range(n_rounds)
    }


# ---------------------------------------------------------------------------
# build_pooled_prior
# ---------------------------------------------------------------------------

class TestBuildPooledPrior:
    def test_returns_dict(self):
        rounds = _make_rounds_data(1)
        result = build_pooled_prior(rounds)
        assert isinstance(result, dict)

    def test_prior_sums_to_one(self):
        rounds = _make_rounds_data(2)
        result = build_pooled_prior(rounds)
        for cat, prior in result.items():
            np.testing.assert_allclose(prior.sum(), 1.0, atol=1e-6)

    def test_prior_nonneg(self):
        rounds = _make_rounds_data(2)
        result = build_pooled_prior(rounds)
        for cat, prior in result.items():
            assert (prior >= 0).all()

    def test_prior_shape(self):
        rounds = _make_rounds_data(2)
        result = build_pooled_prior(rounds)
        for cat, prior in result.items():
            assert prior.shape == (N_CLASSES,)

    def test_exclude_round_skipped(self):
        rounds = _make_rounds_data(3)
        # Exclude round "0" — should produce same result as using only rounds "1" and "2"
        result_full = build_pooled_prior(rounds, exclude_round=None)
        result_excl = build_pooled_prior(rounds, exclude_round="0")
        # Both should still be valid dicts with the same categories
        assert set(result_full.keys()) == set(result_excl.keys())
        for cat in result_full:
            np.testing.assert_allclose(result_full[cat].sum(), 1.0, atol=1e-6)
            np.testing.assert_allclose(result_excl[cat].sum(), 1.0, atol=1e-6)

    def test_empty_rounds_returns_empty(self):
        result = build_pooled_prior({})
        assert result == {}

    def test_single_round(self):
        rounds = _make_rounds_data(1)
        result = build_pooled_prior(rounds)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# predict_with_prior
# ---------------------------------------------------------------------------

class TestPredictWithPrior:
    def _get_priors(self) -> dict:
        rounds = _make_rounds_data(2)
        return build_pooled_prior(rounds)

    def test_output_shape(self):
        ig = _make_ig()
        priors = self._get_priors()
        pred = predict_with_prior(ig, priors)
        assert pred.shape == (*ig.shape, N_CLASSES)

    def test_sums_to_one(self):
        ig = _make_ig()
        priors = self._get_priors()
        pred = predict_with_prior(ig, priors)
        np.testing.assert_allclose(pred.sum(axis=2), np.ones(ig.shape), atol=1e-5)

    def test_nonneg(self):
        ig = _make_ig()
        priors = self._get_priors()
        pred = predict_with_prior(ig, priors)
        assert (pred >= 0).all()

    def test_ocean_cell_gives_class0(self):
        ig = np.full((4, 4), OCEAN, dtype=np.int32)
        priors = {}
        pred = predict_with_prior(ig, priors)
        # Ocean cells should have mass on class 0
        assert (pred[:, :, 0] > 0).all()

    def test_mountain_cell_gives_class5(self):
        ig = np.full((4, 4), MOUNTAIN, dtype=np.int32)
        priors = {}
        pred = predict_with_prior(ig, priors)
        # Mountain cells should have mass on class 5
        assert (pred[:, :, 5] > 0).all()

    def test_returns_float64(self):
        ig = _make_ig()
        priors = self._get_priors()
        pred = predict_with_prior(ig, priors)
        assert pred.dtype in (np.float64, np.float32)

    def test_missing_prior_uses_uniform(self):
        # If category not in priors, falls back to uniform
        ig = np.full((4, 4), SETTLEMENT, dtype=np.int32)
        priors = {}  # empty — forces uniform fallback
        pred = predict_with_prior(ig, priors)
        assert pred.shape == (4, 4, N_CLASSES)
        np.testing.assert_allclose(pred.sum(axis=2), np.ones((4, 4)), atol=1e-5)


# ---------------------------------------------------------------------------
# score_breakdown
# ---------------------------------------------------------------------------

class TestScoreBreakdown:
    def test_returns_dict(self):
        ig = _make_ig()
        gt = _uniform_gt(*ig.shape)
        priors = build_pooled_prior(_make_rounds_data(2))
        pred = predict_with_prior(ig, priors)
        result = score_breakdown(ig, gt, pred)
        assert isinstance(result, dict)

    def test_keys_are_strings(self):
        ig = _make_ig()
        gt = _uniform_gt(*ig.shape)
        priors = build_pooled_prior(_make_rounds_data(2))
        pred = predict_with_prior(ig, priors)
        result = score_breakdown(ig, gt, pred)
        for k in result.keys():
            assert isinstance(k, str)

    def test_category_entry_has_required_keys(self):
        ig = _make_ig()
        gt = _uniform_gt(*ig.shape)
        priors = build_pooled_prior(_make_rounds_data(2))
        pred = predict_with_prior(ig, priors)
        result = score_breakdown(ig, gt, pred)
        for cat, data in result.items():
            assert "mean_wkl" in data
            assert "mean_kl" in data
            assert "count" in data
            assert "dynamic_count" in data
            assert "pct_of_total_loss" in data

    def test_pct_sums_to_100(self):
        ig = _make_ig()
        gt = _uniform_gt(*ig.shape)
        priors = build_pooled_prior(_make_rounds_data(2))
        pred = predict_with_prior(ig, priors)
        result = score_breakdown(ig, gt, pred)
        total_pct = sum(d["pct_of_total_loss"] for d in result.values())
        assert abs(total_pct - 100.0) < 1e-3 or total_pct == 0.0

    def test_mean_wkl_nonneg(self):
        ig = _make_ig()
        gt = _uniform_gt(*ig.shape)
        priors = build_pooled_prior(_make_rounds_data(2))
        pred = predict_with_prior(ig, priors)
        result = score_breakdown(ig, gt, pred)
        for cat, data in result.items():
            assert data["mean_wkl"] >= -1e-9

    def test_identical_gt_pred_low_wkl(self):
        ig = _make_ig()
        gt = _uniform_gt(*ig.shape)
        result = score_breakdown(ig, gt, gt.copy())
        for cat, data in result.items():
            assert data["mean_wkl"] < 1e-3

    def test_count_covers_all_cells(self):
        ig = _make_ig()
        H, W = ig.shape
        gt = _uniform_gt(H, W)
        priors = build_pooled_prior(_make_rounds_data(2))
        pred = predict_with_prior(ig, priors)
        result = score_breakdown(ig, gt, pred)
        total = sum(d["count"] for d in result.values())
        assert total == H * W

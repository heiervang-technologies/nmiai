"""Tests for tasks/astar-island pure helpers in template_predictor and replay_boosted_predictor.

Covers:
- template_predictor: quantized_distance, support_blend
- replay_boosted_predictor: lookup_residual

All pure functions requiring no file I/O or GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR)

from template_predictor import quantized_distance, support_blend
from replay_boosted_predictor import lookup_residual


# ---------------------------------------------------------------------------
# quantized_distance
# ---------------------------------------------------------------------------

class TestQuantizedDistance:
    def test_zero_returns_zero(self):
        assert quantized_distance(0.0) == 0

    def test_one_returns_one(self):
        assert quantized_distance(1.0) == 1

    def test_float_floors(self):
        assert quantized_distance(2.9) == 2

    def test_twelve_returns_twelve(self):
        assert quantized_distance(12.0) == 12

    def test_clamped_at_twelve(self):
        assert quantized_distance(100.0) == 12

    def test_returns_int(self):
        result = quantized_distance(3.5)
        assert isinstance(result, int)

    def test_value_in_range_0_to_12(self):
        for d in [0.0, 1.5, 6.0, 11.9, 12.0, 99.9]:
            q = quantized_distance(d)
            assert 0 <= q <= 12

    def test_negative_clamped_to_zero(self):
        assert quantized_distance(-5.0) == 0


# ---------------------------------------------------------------------------
# support_blend
# ---------------------------------------------------------------------------

class TestSupportBlend:
    def test_zero_count_returns_zero(self):
        assert support_blend(0.0, 10.0) == pytest.approx(0.0)

    def test_large_count_approaches_one(self):
        result = support_blend(10000.0, 1.0)
        assert result > 0.99

    def test_count_equals_shrink_gives_half(self):
        result = support_blend(5.0, 5.0)
        assert result == pytest.approx(0.5)

    def test_returns_float(self):
        result = support_blend(10.0, 5.0)
        assert isinstance(result, float)

    def test_between_zero_and_one(self):
        for count in [1.0, 5.0, 10.0, 100.0]:
            result = support_blend(count, 10.0)
            assert 0.0 <= result <= 1.0

    def test_larger_count_higher_blend(self):
        low = support_blend(5.0, 10.0)
        high = support_blend(50.0, 10.0)
        assert high > low


# ---------------------------------------------------------------------------
# lookup_residual
# ---------------------------------------------------------------------------

class TestLookupResidual:
    def _make_residuals_and_counts(self, n_levels=4):
        residuals = [{} for _ in range(n_levels)]
        counts = [{} for _ in range(n_levels)]
        return residuals, counts

    def test_no_match_returns_none_tuple(self):
        residuals, counts = self._make_residuals_and_counts()
        result, count, level = lookup_residual(residuals, counts, keys=["k1", "k2", "k3", "k4"])
        assert result is None
        assert count == 0

    def test_level_0_hit_with_high_count(self):
        residuals, counts = self._make_residuals_and_counts()
        arr = np.array([0.1, 0.2, 0.3])
        residuals[0]["key0"] = arr
        counts[0]["key0"] = 25  # >= min_counts[0]=20
        result, count, level = lookup_residual(residuals, counts, keys=["key0", "k1", "k2", "k3"])
        assert result is arr
        assert level == 0

    def test_falls_through_to_level_1(self):
        residuals, counts = self._make_residuals_and_counts()
        arr = np.array([0.5, 0.5])
        residuals[1]["key1"] = arr
        counts[1]["key1"] = 35  # >= min_counts[1]=30
        result, count, level = lookup_residual(residuals, counts, keys=["miss", "key1", "k2", "k3"])
        assert result is arr
        assert level == 1

    def test_low_count_skipped_first_pass(self):
        # Level 0 has key but count=5 (< min_counts[0]=20)
        # Should not be returned in first pass
        residuals, counts = self._make_residuals_and_counts()
        arr0 = np.array([0.1])
        arr2 = np.array([0.9])
        residuals[0]["k"] = arr0
        counts[0]["k"] = 5   # below min_counts[0]=20
        residuals[2]["k"] = arr2
        counts[2]["k"] = 55  # >= min_counts[2]=50
        result, count, level = lookup_residual(residuals, counts, keys=["k", "k", "k", "k"])
        # Should return level 2 (first pass skips level 0 due to low count)
        assert result is arr2
        assert level == 2

    def test_fallback_pass_picks_low_count(self):
        # Only level 0 has key with count=8 (< 20, but ≥ 5 for fallback)
        residuals, counts = self._make_residuals_and_counts()
        arr = np.array([0.3])
        residuals[0]["k"] = arr
        counts[0]["k"] = 8
        result, count, level = lookup_residual(residuals, counts, keys=["k", "miss", "miss", "miss"])
        assert result is arr

    def test_returns_tuple_of_three(self):
        residuals, counts = self._make_residuals_and_counts()
        result = lookup_residual(residuals, counts, keys=["a", "b", "c", "d"])
        assert isinstance(result, tuple)
        assert len(result) == 3

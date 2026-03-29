"""Pure-function tests for astar-island/test_quantization.py snap helpers.

Covers: naive_snap, expected_value_snap

Both are pure numpy functions that convert continuous probability arrays
into quantized distributions summing to 1. No I/O, no GPU, no randomness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

from test_quantization import naive_snap, expected_value_snap


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform(h=2, w=2, c=6):
    arr = np.ones((h, w, c)) / c
    return arr


# ---------------------------------------------------------------------------
# naive_snap
# ---------------------------------------------------------------------------

class TestNaiveSnap:
    def test_output_shape_preserved(self):
        p = _uniform(3, 4, 6)
        result = naive_snap(p)
        assert result.shape == (3, 4, 6)

    def test_sums_to_one_per_cell(self):
        p = _uniform(2, 2, 6)
        result = naive_snap(p)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-9)

    def test_all_values_positive(self):
        # naive_snap applies a floor of 1e-6 before renormalizing
        p = np.zeros((1, 1, 6))
        p[0, 0, 0] = 1.0
        result = naive_snap(p)
        assert (result > 0).all()

    def test_uniform_stays_close_to_uniform(self):
        p = _uniform(1, 1, 6)
        result = naive_snap(p, target_sum=600)
        np.testing.assert_allclose(result[0, 0], 1/6, atol=0.01)

    def test_custom_target_sum_still_sums_to_one(self):
        p = _uniform(2, 2, 6)
        result = naive_snap(p, target_sum=100)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-9)

    def test_counts_rounded_correctly(self):
        # With target_sum=6 and uniform 6-class, each class gets exactly 1
        p = _uniform(1, 1, 6)
        result = naive_snap(p, target_sum=6)
        # Each snapped value should be 1/6 ± floor
        np.testing.assert_allclose(result[0, 0], 1/6, atol=0.05)

    def test_float_dtype(self):
        p = _uniform(1, 1, 6)
        result = naive_snap(p)
        assert result.dtype in (np.float32, np.float64, float)

    def test_concentrated_distribution_preserved(self):
        p = np.zeros((1, 1, 6))
        p[0, 0, 0] = 1.0
        result = naive_snap(p, target_sum=200)
        # Class 0 should get almost all probability
        assert result[0, 0, 0] > 0.95


# ---------------------------------------------------------------------------
# expected_value_snap
# ---------------------------------------------------------------------------

class TestExpectedValueSnap:
    def test_output_shape_preserved(self):
        p = _uniform(3, 4, 6)
        result = expected_value_snap(p)
        assert result.shape == (3, 4, 6)

    def test_sums_to_one_per_cell(self):
        p = _uniform(2, 2, 6)
        result = expected_value_snap(p)
        np.testing.assert_allclose(result.sum(axis=-1), 1.0, atol=1e-9)

    def test_all_values_positive(self):
        # EV snap adds epsilon to all counts → always > 0
        p = np.zeros((1, 1, 6))
        p[0, 0, 0] = 1.0
        result = expected_value_snap(p)
        assert (result > 0).all()

    def test_uniform_stays_close_to_uniform(self):
        p = _uniform(1, 1, 6)
        result = expected_value_snap(p, target_sum=600)
        np.testing.assert_allclose(result[0, 0], 1/6, atol=0.01)

    def test_custom_epsilon(self):
        p = _uniform(1, 1, 6)
        r1 = expected_value_snap(p, epsilon=1e-5)
        r2 = expected_value_snap(p, epsilon=0.1)
        # Both should still sum to 1
        np.testing.assert_allclose(r1.sum(axis=-1), 1.0, atol=1e-9)
        np.testing.assert_allclose(r2.sum(axis=-1), 1.0, atol=1e-9)

    def test_concentrated_distribution_preserved(self):
        p = np.zeros((1, 1, 6))
        p[0, 0, 0] = 1.0
        result = expected_value_snap(p, target_sum=200)
        assert result[0, 0, 0] > 0.9

    def test_float_dtype(self):
        p = _uniform(1, 1, 6)
        result = expected_value_snap(p)
        assert result.dtype in (np.float32, np.float64, float)


# ---------------------------------------------------------------------------
# Comparison: naive_snap vs expected_value_snap
# ---------------------------------------------------------------------------

class TestSnapComparison:
    def test_both_give_positive_results(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(6), size=(5, 5))
        assert (naive_snap(p) > 0).all()
        assert (expected_value_snap(p) > 0).all()

    def test_ev_snap_more_uniform_than_naive_for_zeros(self):
        # EV snap adds Dirichlet prior so should be more uniform when input has zeros
        p = np.zeros((1, 1, 6))
        p[0, 0, 0] = 1.0
        naive = naive_snap(p, target_sum=200)
        ev = expected_value_snap(p, target_sum=200, epsilon=0.01)
        # EV snap should give slightly more probability to non-zero classes
        assert ev[0, 0, 1:].sum() > naive[0, 0, 1:].sum()

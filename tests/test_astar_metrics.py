"""Tests for tasks/astar-island information-theoretic metric functions.

Covers:
- benchmark: kl_divergence, entropy (per-cell H×W×C array operations)
- cpu_monte_carlo: calc_wkl (weighted KL divergence)

All pure numpy functions — no file I/O, no GPU, no ground-truth data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR)

from benchmark import entropy, kl_divergence
from cpu_monte_carlo import calc_wkl


# ---------------------------------------------------------------------------
# kl_divergence (benchmark.py)
# ---------------------------------------------------------------------------

class TestKlDivergence:
    def _uniform(self, h, w, c):
        return np.ones((h, w, c)) / c

    def test_returns_ndarray(self):
        p = self._uniform(3, 3, 4)
        result = kl_divergence(p, p)
        assert isinstance(result, np.ndarray)

    def test_output_shape_hw(self):
        p = self._uniform(5, 7, 6)
        result = kl_divergence(p, p)
        assert result.shape == (5, 7)

    def test_identical_distributions_zero_kl(self):
        p = self._uniform(2, 2, 4)
        result = kl_divergence(p, p)
        assert np.allclose(result, 0.0, atol=1e-8)

    def test_kl_nonnegative(self):
        p = np.array([[[0.4, 0.3, 0.2, 0.1]]])
        q = np.array([[[0.25, 0.25, 0.25, 0.25]]])
        result = kl_divergence(p, q)
        assert result[0, 0] >= 0.0

    def test_kl_asymmetric(self):
        # KL(p||q) != KL(q||p) in general
        p = np.array([[[0.8, 0.2]]])
        q = np.array([[[0.5, 0.5]]])
        kl_pq = kl_divergence(p, q)
        kl_qp = kl_divergence(q, p)
        assert not np.allclose(kl_pq, kl_qp)

    def test_uniform_vs_peaked_positive_kl(self):
        p = np.array([[[1.0, 0.0, 0.0, 0.0]]])
        q = self._uniform(1, 1, 4)
        result = kl_divergence(p, q)
        assert result[0, 0] > 0.0


# ---------------------------------------------------------------------------
# entropy (benchmark.py)
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_returns_ndarray(self):
        p = np.ones((2, 3, 4)) / 4
        result = entropy(p)
        assert isinstance(result, np.ndarray)

    def test_output_shape_hw(self):
        p = np.ones((4, 5, 6)) / 6
        result = entropy(p)
        assert result.shape == (4, 5)

    def test_uniform_4class_is_2_bits(self):
        p = np.ones((1, 1, 4)) / 4
        result = entropy(p)
        assert result[0, 0] == pytest.approx(2.0, abs=1e-6)

    def test_uniform_8class_is_3_bits(self):
        p = np.ones((1, 1, 8)) / 8
        result = entropy(p)
        assert result[0, 0] == pytest.approx(3.0, abs=1e-6)

    def test_deterministic_zero_entropy(self):
        p = np.zeros((1, 1, 4))
        p[0, 0, 0] = 1.0
        result = entropy(p)
        assert result[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_entropy_nonnegative(self):
        p = np.array([[[0.5, 0.3, 0.2]]])
        result = entropy(p)
        assert result[0, 0] >= 0.0

    def test_more_uniform_higher_entropy(self):
        peaked = np.array([[[0.9, 0.05, 0.05]]])
        uniform = np.ones((1, 1, 3)) / 3
        assert entropy(uniform)[0, 0] > entropy(peaked)[0, 0]


# ---------------------------------------------------------------------------
# calc_wkl (cpu_monte_carlo.py)
# ---------------------------------------------------------------------------

class TestCalcWkl:
    def _make_probs(self, h, w, c, equal=True):
        if equal:
            return np.ones((h, w, c)) / c
        p = np.random.default_rng(42).dirichlet(np.ones(c), size=(h, w))
        return p

    def _make_ig(self, h, w, val=1):
        return np.full((h, w), val, dtype=np.int32)

    def test_returns_float(self):
        h, w, c = 3, 3, 6
        pred = self._make_probs(h, w, c)
        target = self._make_probs(h, w, c)
        ig = self._make_ig(h, w)
        result = calc_wkl(pred, target, ig)
        assert isinstance(result, float)

    def test_identical_pred_target_near_zero(self):
        h, w, c = 4, 4, 6
        p = self._make_probs(h, w, c)
        ig = self._make_ig(h, w)
        result = calc_wkl(p, p, ig)
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_nonnegative(self):
        h, w, c = 3, 3, 6
        pred = self._make_probs(h, w, c, equal=False)
        target = self._make_probs(h, w, c, equal=False)
        ig = self._make_ig(h, w)
        result = calc_wkl(pred, target, ig)
        assert result >= 0.0

    def test_ocean_cells_ignored(self):
        h, w, c = 3, 3, 6
        p = self._make_probs(h, w, c)
        q = np.zeros((h, w, c))
        q[:, :, 0] = 1.0  # very different from p
        # All ocean (ig == 10) → dynamic_mask is all False → result should be 0 or handle division
        ig_ocean = self._make_ig(h, w, val=10)
        # With all dynamic_mask=False, sum(weights)=0, result would be nan or 0
        result = calc_wkl(p, q, ig_ocean)
        assert np.isnan(result) or result == pytest.approx(0.0, abs=1e-6)

    def test_more_different_distributions_higher_wkl(self):
        h, w, c = 4, 4, 6
        target = self._make_probs(h, w, c)
        similar = target + np.random.default_rng(0).normal(0, 0.01, target.shape)
        similar = np.clip(similar, 1e-8, None)
        similar /= similar.sum(axis=-1, keepdims=True)
        very_different = np.zeros_like(target)
        very_different[:, :, 0] = 1.0
        ig = self._make_ig(h, w, val=1)
        wkl_similar = calc_wkl(similar, target, ig)
        wkl_different = calc_wkl(very_different, target, ig)
        assert wkl_different > wkl_similar

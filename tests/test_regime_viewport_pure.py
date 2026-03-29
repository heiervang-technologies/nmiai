"""Tests for astar-island regime_interpolator and value_viewport_selector pure helpers.

Covers:
  - regime_interpolator.interpolate_global_tensor
  - value_viewport_selector.cell_entropy
Pure torch/numpy functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from regime_interpolator import interpolate_global_tensor
from value_viewport_selector import cell_entropy


# ---------------------------------------------------------------------------
# regime_interpolator.interpolate_global_tensor
# ---------------------------------------------------------------------------

class TestInterpolateGlobalTensor:
    def _uniform_tensor(self, n=4, classes=12):
        """Create uniform probability tensor of shape (n, classes)."""
        t = torch.full((n, classes), 1.0 / classes)
        return t

    def test_returns_tensor(self):
        t = self._uniform_tensor()
        result = interpolate_global_tensor(t, 1.0)
        assert isinstance(result, torch.Tensor)

    def test_shape_preserved(self):
        t = self._uniform_tensor(n=5, classes=12)
        result = interpolate_global_tensor(t, 1.0)
        assert result.shape == (5, 12)

    def test_rows_sum_to_one(self):
        t = self._uniform_tensor()
        result = interpolate_global_tensor(t, 1.3)
        sums = result.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_scalar_1_preserves_distribution(self):
        t = self._uniform_tensor()
        result = interpolate_global_tensor(t, 1.0)
        torch.testing.assert_close(result, t, atol=1e-5, rtol=1e-5)

    def test_scalar_greater_than_1_increases_civ_mass(self):
        t = self._uniform_tensor()
        result = interpolate_global_tensor(t, 2.0)
        # civ mass = classes 1 + 2 should be greater than original
        original_civ = t[:, 1] + t[:, 2]
        new_civ = result[:, 1] + result[:, 2]
        assert (new_civ >= original_civ - 1e-5).all()

    def test_scalar_less_than_1_decreases_civ_mass(self):
        t = self._uniform_tensor()
        result = interpolate_global_tensor(t, 0.5)
        original_civ = t[:, 1] + t[:, 2]
        new_civ = result[:, 1] + result[:, 2]
        assert (new_civ <= original_civ + 1e-5).all()

    def test_capped_at_less_than_1(self):
        # Very high scalar shouldn't make rows sum to > 1
        t = self._uniform_tensor()
        result = interpolate_global_tensor(t, 100.0)
        sums = result.sum(dim=1)
        assert (sums <= 1.0 + 1e-4).all()

    def test_zero_civ_mass_rows_not_changed_by_scalar(self):
        # Row with zero civ mass should be unaffected by scaling
        t = torch.zeros(1, 12)
        t[0, 0] = 0.5
        t[0, 3] = 0.5  # no civ (indices 1,2)
        result = interpolate_global_tensor(t, 2.0)
        torch.testing.assert_close(result[0], t[0], atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# value_viewport_selector.cell_entropy
# ---------------------------------------------------------------------------

class TestCellEntropy:
    def test_uniform_6_class_is_positive(self):
        p = np.array([1.0 / 6.0] * 6)
        result = cell_entropy(p)
        assert result > 0.0

    def test_deterministic_is_zero(self):
        p = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = cell_entropy(p)
        assert abs(result) < 1e-5

    def test_returns_float(self):
        p = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
        result = cell_entropy(p)
        assert isinstance(result, float)

    def test_non_negative(self):
        for _ in range(5):
            p = np.random.dirichlet([1.0] * 6)
            result = cell_entropy(p)
            assert result >= 0.0

    def test_uniform_higher_than_peaked(self):
        uniform = np.array([1.0 / 6.0] * 6)
        peaked = np.array([0.9, 0.02, 0.02, 0.02, 0.02, 0.02])
        assert cell_entropy(uniform) > cell_entropy(peaked)

    def test_uses_nats_not_bits(self):
        # Shannon entropy in nats: -sum(p * log(p))
        # For uniform 2-class: entropy = log(2) ≈ 0.693 nats
        p = np.array([0.5, 0.5])
        result = cell_entropy(p)
        assert result == pytest.approx(np.log(2), abs=1e-5)

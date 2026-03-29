"""Tests for miscellaneous astar-island pure helpers.

Covers:
  - build_regime_tables.get_regime
  - dpca_model.to_onehot, compute_kl_divergence
  - round_optimizer.kl_divergence, entropy, compute_wkl
All pure numpy/torch functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from build_regime_tables import get_regime
from dpca_model import compute_kl_divergence, to_onehot
from round_optimizer import compute_wkl
from round_optimizer import entropy as ro_entropy
from round_optimizer import kl_divergence as ro_kl


# ---------------------------------------------------------------------------
# build_regime_tables.get_regime
# ---------------------------------------------------------------------------

class TestGetRegime:
    def _grid_with_civ_count(self, n):
        """Create a grid with n settlement cells."""
        g = np.zeros((20, 20), dtype=np.int32)
        for i in range(n):
            g[i // 20, i % 20] = 1  # settlement
        return g

    def test_low_count_returns_harsh(self):
        g = self._grid_with_civ_count(10)
        assert get_regime(g) == "Harsh"

    def test_zero_count_returns_harsh(self):
        g = np.zeros((10, 10), dtype=np.int32)
        assert get_regime(g) == "Harsh"

    def test_high_count_returns_prosperous(self):
        g = self._grid_with_civ_count(200)
        assert get_regime(g) == "Prosperous"

    def test_medium_count_returns_moderate(self):
        g = self._grid_with_civ_count(80)
        assert get_regime(g) == "Moderate"

    def test_boundary_29_is_harsh(self):
        g = self._grid_with_civ_count(29)
        assert get_regime(g) == "Harsh"

    def test_boundary_30_is_moderate(self):
        g = self._grid_with_civ_count(30)
        assert get_regime(g) == "Moderate"

    def test_boundary_151_is_prosperous(self):
        g = self._grid_with_civ_count(151)
        assert get_regime(g) == "Prosperous"

    def test_port_counts_as_civ(self):
        g = np.zeros((20, 20), dtype=np.int32)
        g[:10, :] = 2  # port cells → 200 ports → Prosperous
        assert get_regime(g) == "Prosperous"


# ---------------------------------------------------------------------------
# dpca_model.to_onehot
# ---------------------------------------------------------------------------

class TestToOnehot:
    def test_returns_float_tensor(self):
        idx = torch.zeros((1, 3, 4), dtype=torch.long)
        result = to_onehot(idx, num_classes=8)
        assert result.dtype == torch.float32

    def test_shape_is_batch_classes_h_w(self):
        idx = torch.zeros((2, 4, 5), dtype=torch.long)
        result = to_onehot(idx, num_classes=8)
        assert result.shape == (2, 8, 4, 5)

    def test_each_position_sums_to_one(self):
        idx = torch.randint(0, 6, (1, 4, 4))
        result = to_onehot(idx, num_classes=6)
        sums = result.sum(dim=1)
        torch.testing.assert_close(sums, torch.ones_like(sums))

    def test_correct_class_is_hot(self):
        idx = torch.tensor([[[2, 3]]])  # shape (1, 1, 2)
        result = to_onehot(idx, num_classes=6)
        assert result[0, 2, 0, 0] == 1.0  # class 2 at position 0
        assert result[0, 3, 0, 1] == 1.0  # class 3 at position 1


# ---------------------------------------------------------------------------
# dpca_model.compute_kl_divergence
# ---------------------------------------------------------------------------

class TestDpcaComputeKlDivergence:
    def _uniform(self, b=1, h=2, w=2, n=6):
        return torch.full((b, n, h, w), 1.0 / n)

    def test_identical_near_zero(self):
        p = self._uniform()
        result = compute_kl_divergence(p, p)
        assert result.item() < 1e-4

    def test_returns_scalar(self):
        p = self._uniform()
        result = compute_kl_divergence(p, p)
        assert result.dim() == 0

    def test_non_negative(self):
        pred = self._uniform()
        gt = torch.softmax(torch.randn(1, 6, 2, 2), dim=1)
        result = compute_kl_divergence(pred, gt)
        assert result.item() >= 0.0


# ---------------------------------------------------------------------------
# round_optimizer.kl_divergence, entropy, compute_wkl
# ---------------------------------------------------------------------------

class TestRoundOptimizerKlDivergence:
    def test_identical_near_zero(self):
        p = np.full((2, 2, 6), 1.0 / 6)
        result = ro_kl(p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_shape_removes_class_axis(self):
        p = np.full((3, 4, 6), 1.0 / 6)
        result = ro_kl(p, p)
        assert result.shape == (3, 4)


class TestRoundOptimizerEntropy:
    def test_uniform_6_class_gives_log2_6(self):
        p = np.full((2, 2, 6), 1.0 / 6)
        result = ro_entropy(p)
        np.testing.assert_allclose(result, np.log2(6), atol=1e-5)

    def test_deterministic_gives_zero(self):
        p = np.zeros((2, 2, 6))
        p[:, :, 0] = 1.0
        result = ro_entropy(p)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)


class TestComputeWkl:
    def test_identical_near_zero(self):
        p = np.full((3, 3, 6), 1.0 / 6)
        result = compute_wkl(p, p)
        assert abs(result) < 1e-4

    def test_returns_float(self):
        p = np.full((3, 3, 6), 1.0 / 6)
        assert isinstance(compute_wkl(p, p), float)

    def test_all_static_returns_zero(self):
        # Static cells have entropy=0 → wkl=0
        p = np.zeros((3, 3, 6))
        p[:, :, 0] = 1.0
        result = compute_wkl(p, p)
        assert result == 0.0

    def test_non_negative(self):
        gt = np.full((3, 3, 6), 1.0 / 6)
        pred = np.random.dirichlet([1.0] * 6, size=(3, 3))
        result = compute_wkl(gt, pred)
        assert result >= 0.0

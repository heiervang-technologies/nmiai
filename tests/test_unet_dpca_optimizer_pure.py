"""Pure-function tests for astar-island unet/dpca/optimizer helpers.

Covers:
  tasks/astar-island/unet_predictor.py — grid_to_features, compute_kl_loss
  tasks/astar-island/dpca_model.py     — to_onehot, compute_kl_divergence
  tasks/astar-island/round_optimizer.py — kl_divergence, entropy, compute_wkl

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

from unet_predictor import grid_to_features, compute_kl_loss
from dpca_model import to_onehot, compute_kl_divergence
from round_optimizer import kl_divergence as opt_kl, entropy as opt_entropy, compute_wkl


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_grid(code: int = 4, size: int = 10) -> np.ndarray:
    """Small grid filled with a single cell code."""
    return np.full((size, size), code, dtype=np.int32)


def _mixed_grid(size: int = 10) -> np.ndarray:
    """Grid mixing ocean(5), civ(4), mountain(10), plains(1)."""
    g = np.ones((size, size), dtype=np.int32)
    g[:2, :] = 5   # ocean strip
    g[-2:, :] = 10  # mountain strip
    g[4, 4] = 4    # single civ cell
    return g


# ---------------------------------------------------------------------------
# unet_predictor.grid_to_features
# ---------------------------------------------------------------------------

UNET_NUM_CHANNELS = 10  # 6 one-hot + dist_civ + dist_ocean + ocean_adj + mtn_adj

class TestGridToFeatures:
    def test_output_shape(self):
        g = _uniform_grid(size=40)
        features = grid_to_features(g)
        assert features.shape == (UNET_NUM_CHANNELS, 40, 40)

    def test_output_shape_small(self):
        g = _mixed_grid(size=10)
        features = grid_to_features(g)
        assert features.shape == (UNET_NUM_CHANNELS, 10, 10)

    def test_one_hot_channels_sum_to_one(self):
        g = _mixed_grid(size=10)
        features = grid_to_features(g)
        # Channels 0-5 are one-hot; each spatial position has exactly 1 channel active
        one_hot = features[:6]  # (6, H, W)
        channel_sum = one_hot.sum(axis=0)
        np.testing.assert_allclose(channel_sum, np.ones((10, 10)), atol=1e-6)

    def test_dist_civ_nonneg(self):
        g = _mixed_grid(size=10)
        features = grid_to_features(g)
        dist_civ = features[6]  # channel 6
        assert (dist_civ >= 0).all()

    def test_dist_ocean_nonneg(self):
        g = _mixed_grid(size=10)
        features = grid_to_features(g)
        dist_ocean = features[7]
        assert (dist_ocean >= 0).all()

    def test_ocean_adj_in_range(self):
        g = _mixed_grid(size=10)
        features = grid_to_features(g)
        ocean_adj = features[8]
        assert (ocean_adj >= 0).all()
        assert (ocean_adj <= 1.0).all()

    def test_mtn_adj_in_range(self):
        g = _mixed_grid(size=10)
        features = grid_to_features(g)
        mtn_adj = features[9]
        assert (mtn_adj >= 0).all()
        assert (mtn_adj <= 1.0).all()

    def test_no_civ_dist_all_ones(self):
        # Pure ocean grid — no civ cells → dist_civ should be all 1.0 (fallback)
        g = np.full((10, 10), 5, dtype=np.int32)  # all ocean
        features = grid_to_features(g)
        dist_civ = features[6]
        np.testing.assert_allclose(dist_civ, np.ones((10, 10)), atol=1e-6)

    def test_no_ocean_dist_all_ones(self):
        # Pure civ grid — no ocean cells → dist_ocean should be all 1.0 (fallback)
        g = np.full((10, 10), 4, dtype=np.int32)
        features = grid_to_features(g)
        dist_ocean = features[7]
        np.testing.assert_allclose(dist_ocean, np.ones((10, 10)), atol=1e-6)

    def test_returns_float32(self):
        g = _mixed_grid(size=10)
        features = grid_to_features(g)
        assert features.dtype == np.float32


# ---------------------------------------------------------------------------
# unet_predictor.compute_kl_loss
# ---------------------------------------------------------------------------

class TestComputeKlLoss:
    def test_identical_preds_gives_zero(self):
        B, C, H, W = 2, 6, 4, 4
        pred = torch.full((B, C, H, W), 1.0 / C)
        gt = torch.full((B, C, H, W), 1.0 / C)
        mask = torch.ones(B, H, W, dtype=torch.bool)
        loss = compute_kl_loss(pred, gt, mask)
        assert float(loss) < 1e-5

    def test_empty_mask_returns_zero(self):
        B, C, H, W = 2, 6, 4, 4
        pred = torch.rand(B, C, H, W)
        gt = torch.rand(B, C, H, W)
        mask = torch.zeros(B, H, W, dtype=torch.bool)
        loss = compute_kl_loss(pred, gt, mask)
        assert float(loss) == 0.0

    def test_nonneg(self):
        B, C, H, W = 1, 6, 5, 5
        pred = torch.softmax(torch.randn(B, C, H, W), dim=1)
        gt = torch.softmax(torch.randn(B, C, H, W), dim=1)
        mask = torch.ones(B, H, W, dtype=torch.bool)
        loss = compute_kl_loss(pred, gt, mask)
        assert float(loss) >= 0

    def test_returns_scalar(self):
        B, C, H, W = 1, 6, 4, 4
        pred = torch.softmax(torch.randn(B, C, H, W), dim=1)
        gt = torch.softmax(torch.randn(B, C, H, W), dim=1)
        mask = torch.ones(B, H, W, dtype=torch.bool)
        loss = compute_kl_loss(pred, gt, mask)
        assert loss.ndim == 0

    def test_partial_mask(self):
        B, C, H, W = 1, 6, 4, 4
        pred = torch.softmax(torch.randn(B, C, H, W), dim=1)
        gt = torch.softmax(torch.randn(B, C, H, W), dim=1)
        mask = torch.zeros(B, H, W, dtype=torch.bool)
        mask[0, :2, :2] = True
        loss = compute_kl_loss(pred, gt, mask)
        assert float(loss) >= 0


# ---------------------------------------------------------------------------
# dpca_model.to_onehot
# ---------------------------------------------------------------------------

class TestToOnehot:
    def test_basic_shape(self):
        x = torch.zeros(1, 10, 10, dtype=torch.long)
        out = to_onehot(x, num_classes=8)
        assert out.shape == (1, 8, 10, 10)

    def test_single_class_encoded(self):
        x = torch.full((1, 4, 4), 3, dtype=torch.long)
        out = to_onehot(x, num_classes=8)
        # Channel 3 should be all 1s, others all 0s
        assert out[0, 3].all()
        assert not out[0, :3].any()
        assert not out[0, 4:].any()

    def test_sum_per_cell_is_one(self):
        x = torch.randint(0, 8, (2, 5, 5))
        out = to_onehot(x, num_classes=8)
        assert (out.sum(dim=1) == 1).all()

    def test_numpy_input_works(self):
        x = np.array([[[0, 1], [2, 3]]])
        out = to_onehot(x, num_classes=4)
        assert out.shape == (1, 4, 2, 2)

    def test_output_float(self):
        x = torch.zeros(1, 3, 3, dtype=torch.long)
        out = to_onehot(x)
        assert out.dtype == torch.float32


# ---------------------------------------------------------------------------
# dpca_model.compute_kl_divergence
# ---------------------------------------------------------------------------

class TestDpcaComputeKlDivergence:
    def _uniform(self, B: int = 1, C: int = 8, H: int = 4, W: int = 4) -> torch.Tensor:
        return torch.full((B, C, H, W), 1.0 / C)

    def test_identical_gives_near_zero(self):
        p = self._uniform()
        loss = compute_kl_divergence(p.clone(), p.clone())
        assert float(loss) < 1e-4

    def test_returns_scalar(self):
        p = self._uniform()
        q = self._uniform()
        loss = compute_kl_divergence(p, q)
        assert loss.ndim == 0

    def test_nonneg(self):
        B, C, H, W = 2, 8, 4, 4
        pred = torch.softmax(torch.randn(B, C, H, W), dim=1)
        gt = torch.softmax(torch.randn(B, C, H, W), dim=1)
        loss = compute_kl_divergence(pred, gt)
        assert float(loss) >= 0

    def test_with_h_gt_all_static_returns_zero(self):
        p = self._uniform()
        q = self._uniform()
        # H_gt all zeros → dynamic mask empty → 0.0
        H_gt = torch.zeros(1, 4, 4)
        loss = compute_kl_divergence(p, q, H_gt)
        assert float(loss) == 0.0

    def test_with_h_gt_dynamic_nonneg(self):
        p = torch.softmax(torch.randn(1, 8, 4, 4), dim=1)
        q = torch.softmax(torch.randn(1, 8, 4, 4), dim=1)
        H_gt = torch.ones(1, 4, 4)  # all dynamic
        loss = compute_kl_divergence(p, q, H_gt)
        assert float(loss) >= 0


# ---------------------------------------------------------------------------
# round_optimizer.kl_divergence, entropy, compute_wkl
# ---------------------------------------------------------------------------

class TestOptKlDivergence:
    def test_identical_gives_zero(self):
        p = np.full((4, 4, 6), 1.0 / 6)
        result = opt_kl(p, p)
        np.testing.assert_allclose(result, np.zeros((4, 4)), atol=1e-6)

    def test_nonneg(self):
        p = np.random.dirichlet(np.ones(6), size=(4, 4))
        p = p[:, :, np.newaxis] * np.ones((1, 1, 6))  # broadcast
        p = np.random.dirichlet(np.ones(6), size=(4, 4, 6))
        p /= p.sum(axis=2, keepdims=True)
        q = np.random.dirichlet(np.ones(6), size=(4, 4, 6))
        q /= q.sum(axis=2, keepdims=True)
        result = opt_kl(p, q)
        assert (result >= -1e-9).all()

    def test_output_shape(self):
        H, W, C = 5, 6, 8
        p = np.full((H, W, C), 1.0 / C)
        q = np.full((H, W, C), 1.0 / C)
        result = opt_kl(p, q)
        assert result.shape == (H, W)


class TestOptEntropy:
    def test_uniform_max_entropy(self):
        C = 8
        p = np.full((4, 4, C), 1.0 / C)
        result = opt_entropy(p)
        expected = np.log2(C)
        np.testing.assert_allclose(result, np.full((4, 4), expected), atol=1e-5)

    def test_deterministic_zero_entropy(self):
        C = 6
        p = np.zeros((4, 4, C))
        p[:, :, 0] = 1.0  # all mass on class 0
        result = opt_entropy(p)
        np.testing.assert_allclose(result, np.zeros((4, 4)), atol=1e-4)

    def test_nonneg(self):
        p = np.random.dirichlet(np.ones(6), size=(5, 5, 6))
        p /= p.sum(axis=2, keepdims=True)
        result = opt_entropy(p)
        assert (result >= -1e-9).all()

    def test_output_shape(self):
        p = np.full((3, 4, 5), 0.2)
        result = opt_entropy(p)
        assert result.shape == (3, 4)


class TestComputeWkl:
    def test_identical_gives_zero(self):
        C = 6
        gt = np.full((4, 4, C), 1.0 / C)
        pred = np.full((4, 4, C), 1.0 / C)
        # Uniform gt → non-zero entropy, KL=0 → wkl=0
        result = compute_wkl(gt, pred)
        assert abs(result) < 1e-5

    def test_all_static_gives_zero(self):
        # Deterministic gt → entropy=0 → dynamic mask empty → returns 0.0
        C = 6
        gt = np.zeros((4, 4, C))
        gt[:, :, 0] = 1.0
        pred = np.full((4, 4, C), 1.0 / C)
        result = compute_wkl(gt, pred)
        assert result == 0.0

    def test_nonneg(self):
        C = 6
        gt = np.random.dirichlet(np.ones(C), size=(4, 4, C))
        gt /= gt.sum(axis=2, keepdims=True)
        pred = np.random.dirichlet(np.ones(C), size=(4, 4, C))
        result = compute_wkl(gt, pred)
        assert result >= 0

    def test_returns_float(self):
        C = 6
        gt = np.full((4, 4, C), 1.0 / C)
        pred = np.full((4, 4, C), 1.0 / C)
        result = compute_wkl(gt, pred)
        assert isinstance(result, float)

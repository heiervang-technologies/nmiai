"""Tests for tasks/astar-island/recursive_model.py — pure helper functions.

Covers: initial_state_from_grid, encode_grid, kl_loss.
Pure numpy/torch functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from recursive_model import N_CLASSES, encode_grid, initial_state_from_grid, kl_loss


# ---------------------------------------------------------------------------
# initial_state_from_grid
# ---------------------------------------------------------------------------

class TestInitialStateFromGrid:
    def _grid(self, codes, shape=(2, 2)):
        """Build a grid with given cell codes."""
        g = np.array(codes, dtype=np.int32).reshape(shape)
        return g

    def test_returns_ndarray(self):
        g = self._grid([11, 11, 11, 11])
        result = initial_state_from_grid(g)
        assert isinstance(result, np.ndarray)

    def test_shape_is_6_h_w(self):
        g = np.full((3, 5), 11, dtype=np.int32)
        result = initial_state_from_grid(g)
        assert result.shape == (6, 3, 5)

    def test_dtype_float32(self):
        g = np.full((2, 2), 11, dtype=np.int32)
        result = initial_state_from_grid(g)
        assert result.dtype == np.float32

    def test_plains_goes_to_class_0(self):
        g = np.full((2, 2), 11, dtype=np.int32)
        result = initial_state_from_grid(g)
        np.testing.assert_array_equal(result[0], np.ones((2, 2)))

    def test_ocean_goes_to_class_0(self):
        g = np.full((2, 2), 10, dtype=np.int32)
        result = initial_state_from_grid(g)
        np.testing.assert_array_equal(result[0], np.ones((2, 2)))

    def test_empty_goes_to_class_0(self):
        g = np.full((2, 2), 0, dtype=np.int32)
        result = initial_state_from_grid(g)
        np.testing.assert_array_equal(result[0], np.ones((2, 2)))

    def test_settlement_goes_to_class_1(self):
        g = np.array([[1, 11], [11, 11]], dtype=np.int32)
        result = initial_state_from_grid(g)
        assert result[1, 0, 0] == 1.0
        assert result[1, 0, 1] == 0.0

    def test_port_goes_to_class_2(self):
        g = np.array([[2, 11], [11, 11]], dtype=np.int32)
        result = initial_state_from_grid(g)
        assert result[2, 0, 0] == 1.0
        assert result[2, 0, 1] == 0.0

    def test_forest_goes_to_class_4(self):
        g = np.array([[4, 11], [11, 11]], dtype=np.int32)
        result = initial_state_from_grid(g)
        assert result[4, 0, 0] == 1.0

    def test_mountain_goes_to_class_5(self):
        g = np.array([[5, 11], [11, 11]], dtype=np.int32)
        result = initial_state_from_grid(g)
        assert result[5, 0, 0] == 1.0

    def test_class_3_ruin_always_zero(self):
        # Ruin (code 3) is not in initial grid → class 3 should be all zeros
        g = np.array([[1, 2], [4, 5]], dtype=np.int32)
        result = initial_state_from_grid(g)
        np.testing.assert_array_equal(result[3], np.zeros((2, 2)))

    def test_each_cell_has_exactly_one_active_class(self):
        # Each cell is assigned to exactly one class
        g = np.array([[0, 1, 2, 4, 5, 10, 11]], dtype=np.int32).reshape(1, 7)
        result = initial_state_from_grid(g)
        sums = result.sum(axis=0)  # sum over class dimension → (1, 7)
        np.testing.assert_array_equal(sums, np.ones((1, 7)))


# ---------------------------------------------------------------------------
# encode_grid
# ---------------------------------------------------------------------------

class TestEncodeGrid:
    def _simple_grid(self, h=5, w=5):
        g = np.full((h, w), 11, dtype=np.int32)  # all plains
        g[2, 2] = 1  # settlement
        return g

    def test_returns_ndarray(self):
        g = self._simple_grid()
        result = encode_grid(g)
        assert isinstance(result, np.ndarray)

    def test_shape_has_c_h_w_form(self):
        g = self._simple_grid(h=4, w=6)
        result = encode_grid(g)
        # Shape is (C, H, W) where C > 1
        assert len(result.shape) == 3
        assert result.shape[1] == 4
        assert result.shape[2] == 6

    def test_dtype_float32(self):
        g = self._simple_grid()
        result = encode_grid(g)
        assert result.dtype == np.float32

    def test_dist_civ_channel_bounded(self):
        g = self._simple_grid()
        result = encode_grid(g)
        # dist_civ is normalized to [0, 1] at channel index 11
        dist_channel = result[11]
        assert dist_channel.min() >= 0.0
        assert dist_channel.max() <= 1.0 + 1e-5

    def test_no_civ_dist_all_ones(self):
        g = np.full((4, 4), 11, dtype=np.int32)  # no civ → dist_civ = ones
        result = encode_grid(g)
        dist_channel = result[11]
        np.testing.assert_allclose(dist_channel, 1.0, atol=1e-5)

    def test_position_channels_in_minus1_to_1(self):
        g = self._simple_grid(h=5, w=5)
        result = encode_grid(g)
        # Channels 9 and 10 are y and x position (linspace -1 to 1)
        for ch in [9, 10]:
            assert result[ch].min() >= -1.0 - 1e-5
            assert result[ch].max() <= 1.0 + 1e-5


# ---------------------------------------------------------------------------
# kl_loss
# ---------------------------------------------------------------------------

class TestKlLoss:
    def _uniform(self, b=1, h=2, w=2):
        t = torch.full((b, N_CLASSES, h, w), 1.0 / N_CLASSES)
        return t

    def test_returns_scalar_tensor(self):
        p = self._uniform()
        q = self._uniform()
        result = kl_loss(q, p)
        assert result.dim() == 0

    def test_identical_distributions_near_zero(self):
        p = self._uniform()
        result = kl_loss(p, p)
        assert result.item() < 1e-4

    def test_non_negative(self):
        p = self._uniform()
        q = torch.rand(1, N_CLASSES, 2, 2)  # same spatial size as p
        q = q / q.sum(dim=1, keepdim=True)
        result = kl_loss(q, p)
        assert result.item() >= 0.0

    def test_larger_for_more_different_distributions(self):
        p = self._uniform()
        # q1 slightly different, q2 very different
        q1 = p.clone()
        q1[0, 0, :, :] += 0.1
        q1 = q1 / q1.sum(dim=1, keepdim=True)

        q2 = torch.zeros_like(p)
        q2[0, 0, :, :] = 1.0  # degenerate

        loss1 = kl_loss(q1, p)
        loss2 = kl_loss(q2, p)
        assert loss2.item() > loss1.item()

    def test_deterministic_pred_nonzero_for_uniform_target(self):
        """KL(uniform || degenerate) should be large and positive."""
        target = torch.full((1, N_CLASSES, 1, 1), 1.0 / N_CLASSES)  # uniform
        pred = torch.zeros(1, N_CLASSES, 1, 1)
        pred[0, 0, 0, 0] = 1.0  # degenerate pred
        result = kl_loss(pred, target)
        assert result.item() > 0.1

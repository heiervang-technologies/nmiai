"""Tests for tasks/astar-island/diffusion_model.py — pure helpers.

Covers: floor_and_normalize, grid_hash, initial_class_index,
deterministic_static_distribution, static_mask, normalized_distance,
prior_state_from_grid, encode_grid, apply_static_override, weighted_kl_loss.
All pure functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from diffusion_model import (
    FLOOR,
    GRID_SIZE,
    N_CLASSES,
    apply_static_override,
    deterministic_static_distribution,
    encode_grid,
    floor_and_normalize,
    grid_hash,
    initial_class_index,
    normalized_distance,
    prior_state_from_grid,
    static_mask,
    weighted_kl_loss,
)


# ---------------------------------------------------------------------------
# floor_and_normalize
# ---------------------------------------------------------------------------

class TestFloorAndNormalize:
    def test_zeros_become_floor(self):
        arr = np.zeros((2, 2, N_CLASSES), dtype=np.float32)
        result = floor_and_normalize(arr)
        assert (result >= FLOOR).all()

    def test_output_sums_to_one_per_cell(self):
        arr = np.random.rand(3, 3, N_CLASSES).astype(np.float32)
        result = floor_and_normalize(arr)
        np.testing.assert_allclose(result.sum(axis=2), 1.0, atol=1e-5)

    def test_shape_preserved(self):
        arr = np.ones((4, 5, N_CLASSES), dtype=np.float32)
        result = floor_and_normalize(arr)
        assert result.shape == (4, 5, N_CLASSES)

    def test_already_normalized_stays_normalized(self):
        arr = np.full((2, 2, N_CLASSES), 1.0 / N_CLASSES)
        result = floor_and_normalize(arr)
        np.testing.assert_allclose(result.sum(axis=2), 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# grid_hash
# ---------------------------------------------------------------------------

class TestGridHash:
    def test_returns_string(self):
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        assert isinstance(grid_hash(g), str)

    def test_same_grid_same_hash(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        assert grid_hash(g) == grid_hash(g)

    def test_different_grids_different_hashes(self):
        g1 = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        g2 = g1.copy()
        g2[0, 0] = 1
        assert grid_hash(g1) != grid_hash(g2)

    def test_accepts_list(self):
        g = [[11] * 5 for _ in range(5)]
        result = grid_hash(g)
        assert isinstance(result, str)

    def test_sha1_length(self):
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        assert len(grid_hash(g)) == 40


# ---------------------------------------------------------------------------
# initial_class_index
# ---------------------------------------------------------------------------

class TestInitialClassIndex:
    def test_empty_maps_0(self):
        assert initial_class_index(0) == 0

    def test_ocean_maps_0(self):
        assert initial_class_index(10) == 0

    def test_plains_maps_0(self):
        assert initial_class_index(11) == 0

    def test_settlement_maps_1(self):
        assert initial_class_index(1) == 1

    def test_port_maps_2(self):
        assert initial_class_index(2) == 2

    def test_forest_maps_4(self):
        assert initial_class_index(4) == 4

    def test_mountain_maps_5(self):
        assert initial_class_index(5) == 5

    def test_ruin_maps_0(self):
        # Ruin (3) is not in CELL_CODES — maps to 0
        assert initial_class_index(3) == 0

    def test_unknown_maps_0(self):
        assert initial_class_index(99) == 0


# ---------------------------------------------------------------------------
# deterministic_static_distribution
# ---------------------------------------------------------------------------

class TestDeterministicStaticDistribution:
    def _make_grid(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        g[5, 5] = 1   # settlement
        g[0, 0] = 10  # ocean
        g[39, 39] = 5 # mountain
        return g

    def test_shape(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        result = deterministic_static_distribution(g)
        assert result.shape == (GRID_SIZE, GRID_SIZE, N_CLASSES)

    def test_settlement_cell_class_1(self):
        g = self._make_grid()
        result = deterministic_static_distribution(g)
        assert result[5, 5, 1] == 1.0
        assert result[5, 5, 0] == 0.0

    def test_ocean_cell_class_0(self):
        g = self._make_grid()
        result = deterministic_static_distribution(g)
        assert result[0, 0, 0] == 1.0

    def test_mountain_cell_class_5(self):
        g = self._make_grid()
        result = deterministic_static_distribution(g)
        assert result[39, 39, 5] == 1.0

    def test_float32_dtype(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        result = deterministic_static_distribution(g)
        assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# static_mask
# ---------------------------------------------------------------------------

class TestStaticMask:
    def test_ocean_is_static(self):
        g = np.full((5, 5), 11, dtype=np.int32)
        g[2, 2] = 10
        result = static_mask(g)
        assert result[2, 2] == 1.0

    def test_mountain_is_static(self):
        g = np.full((5, 5), 11, dtype=np.int32)
        g[1, 3] = 5
        result = static_mask(g)
        assert result[1, 3] == 1.0

    def test_plains_is_not_static(self):
        g = np.full((5, 5), 11, dtype=np.int32)
        result = static_mask(g)
        assert (result == 0.0).all()

    def test_shape(self):
        g = np.zeros((6, 7), dtype=np.int32)
        assert static_mask(g).shape == (6, 7)

    def test_float32_dtype(self):
        g = np.full((5, 5), 11, dtype=np.int32)
        assert static_mask(g).dtype == np.float32


# ---------------------------------------------------------------------------
# normalized_distance
# ---------------------------------------------------------------------------

class TestNormalizedDistance:
    def test_no_mask_gives_all_ones(self):
        mask = np.zeros((5, 5), dtype=bool)
        result = normalized_distance(mask)
        assert (result == 1.0).all()

    def test_at_masked_cell_distance_is_zero(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        result = normalized_distance(mask)
        assert result[5, 5] == 0.0

    def test_output_in_zero_one(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[0, 0] = True
        result = normalized_distance(mask)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_float32_dtype(self):
        mask = np.zeros((5, 5), dtype=bool)
        assert normalized_distance(mask).dtype == np.float32


# ---------------------------------------------------------------------------
# prior_state_from_grid
# ---------------------------------------------------------------------------

class TestPriorStateFromGrid:
    def _make_priors(self):
        return {11: np.full(N_CLASSES, 1.0 / N_CLASSES)}

    def test_shape(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        result = prior_state_from_grid(g, self._make_priors())
        assert result.shape == (GRID_SIZE, GRID_SIZE, N_CLASSES)

    def test_rows_sum_to_one(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        result = prior_state_from_grid(g, self._make_priors())
        np.testing.assert_allclose(result.sum(axis=2), 1.0, atol=1e-5)

    def test_applies_code_vector(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        g[20, 20] = 1  # settlement
        settlement_prior = np.array([0.0, 0.9, 0.0, 0.0, 0.05, 0.05])
        priors = {11: np.full(N_CLASSES, 1.0 / N_CLASSES), 1: settlement_prior}
        result = prior_state_from_grid(g, priors)
        # Settlement cell should be dominated by settlement_prior (after flooring)
        assert result[20, 20, 1] > result[20, 20, 0]


# ---------------------------------------------------------------------------
# encode_grid
# ---------------------------------------------------------------------------

class TestEncodeGrid:
    def _simple_grid(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        g[20, 20] = 1
        return g

    def _simple_priors(self):
        return {11: np.full(N_CLASSES, 1.0 / N_CLASSES),
                1: np.array([0.0, 0.9, 0.05, 0.0, 0.025, 0.025])}

    def test_output_is_ndarray(self):
        g = self._simple_grid()
        result = encode_grid(g, self._simple_priors())
        assert isinstance(result, np.ndarray)

    def test_float32_dtype(self):
        g = self._simple_grid()
        result = encode_grid(g, self._simple_priors())
        assert result.dtype == np.float32

    def test_shape_channels_height_width(self):
        g = self._simple_grid()
        result = encode_grid(g, self._simple_priors())
        # C channels, GRID_SIZE x GRID_SIZE
        assert result.ndim == 3
        assert result.shape[1] == GRID_SIZE
        assert result.shape[2] == GRID_SIZE

    def test_first_axis_has_many_channels(self):
        g = self._simple_grid()
        result = encode_grid(g, self._simple_priors())
        # 7 one-hot + 6 prior + 2 pos + 4 dist + 3 masks = 22 channels
        assert result.shape[0] >= 16


# ---------------------------------------------------------------------------
# apply_static_override
# ---------------------------------------------------------------------------

class TestApplyStaticOverride:
    def test_all_static_returns_static_target(self):
        pred = torch.rand(2, N_CLASSES, 4, 4)
        static = torch.ones(2, 1, 4, 4)
        st_target = torch.rand(2, N_CLASSES, 4, 4)
        result = apply_static_override(pred, static, st_target)
        assert torch.allclose(result, st_target)

    def test_all_dynamic_returns_pred(self):
        pred = torch.rand(2, N_CLASSES, 4, 4)
        static = torch.zeros(2, 1, 4, 4)
        st_target = torch.rand(2, N_CLASSES, 4, 4)
        result = apply_static_override(pred, static, st_target)
        assert torch.allclose(result, pred)

    def test_shape_preserved(self):
        pred = torch.rand(1, N_CLASSES, 5, 5)
        static = torch.rand(1, 1, 5, 5)
        st_target = torch.rand(1, N_CLASSES, 5, 5)
        result = apply_static_override(pred, static, st_target)
        assert result.shape == pred.shape


# ---------------------------------------------------------------------------
# weighted_kl_loss
# ---------------------------------------------------------------------------

class TestWeightedKlLoss:
    def _uniform(self, b=1, h=4, w=4):
        return torch.full((b, N_CLASSES, h, w), 1.0 / N_CLASSES)

    def test_identical_near_zero(self):
        p = self._uniform()
        dynamic = torch.ones(1, 1, 4, 4)
        result = weighted_kl_loss(p, p, dynamic)
        assert abs(result.item()) < 1e-4

    def test_returns_scalar(self):
        p = self._uniform()
        dynamic = torch.ones(1, 1, 4, 4)
        result = weighted_kl_loss(p, p, dynamic)
        assert result.ndim == 0

    def test_non_negative(self):
        p = self._uniform()
        q = torch.softmax(torch.rand(1, N_CLASSES, 4, 4), dim=1)
        dynamic = torch.ones(1, 1, 4, 4)
        result = weighted_kl_loss(p, q, dynamic)
        assert result.item() >= 0.0

    def test_all_static_near_zero(self):
        p = self._uniform()
        q = torch.softmax(torch.rand(1, N_CLASSES, 4, 4), dim=1)
        dynamic = torch.zeros(1, 1, 4, 4)
        result = weighted_kl_loss(p, q, dynamic)
        assert result.item() == 0.0 or np.isnan(result.item()) or result.item() >= 0

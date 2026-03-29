"""Tests for tasks/astar-island/diffusion_ca.py — pure feature computation helpers.

Covers: code_to_class_grid, compute_feature_maps, features_to_tensor.
All use only numpy/scipy — no file system, network, or GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from diffusion_ca import (
    N_CLASSES,
    code_to_class_grid,
    compute_feature_maps,
    features_to_tensor,
)

# Cell codes used in tests
OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4
PLAINS = 11
EMPTY = 0


# ---------------------------------------------------------------------------
# code_to_class_grid
# ---------------------------------------------------------------------------

class TestCodeToClassGrid:
    def _grid(self, *codes):
        return np.array(codes, dtype=np.int32).reshape(1, -1)

    def test_ocean_maps_to_zero(self):
        g = self._grid(10)
        assert code_to_class_grid(g)[0, 0] == 0

    def test_plains_maps_to_zero(self):
        g = self._grid(11)
        assert code_to_class_grid(g)[0, 0] == 0

    def test_empty_maps_to_zero(self):
        g = self._grid(0)
        assert code_to_class_grid(g)[0, 0] == 0

    def test_settlement_maps_to_one(self):
        g = self._grid(1)
        assert code_to_class_grid(g)[0, 0] == 1

    def test_port_maps_to_two(self):
        g = self._grid(2)
        assert code_to_class_grid(g)[0, 0] == 2

    def test_forest_maps_to_four(self):
        g = self._grid(4)
        assert code_to_class_grid(g)[0, 0] == 4

    def test_mountain_maps_to_five(self):
        g = self._grid(5)
        assert code_to_class_grid(g)[0, 0] == 5

    def test_output_shape_matches_input(self):
        g = np.ones((3, 4), dtype=np.int32)
        out = code_to_class_grid(g)
        assert out.shape == (3, 4)

    def test_output_dtype_int64(self):
        g = np.zeros((2, 2), dtype=np.int32)
        out = code_to_class_grid(g)
        assert out.dtype == np.int64

    def test_mixed_grid(self):
        g = np.array([[0, 1, 2], [4, 5, 10]], dtype=np.int32)
        out = code_to_class_grid(g)
        assert out[0, 0] == 0  # empty → 0
        assert out[0, 1] == 1  # settlement → 1
        assert out[0, 2] == 2  # port → 2
        assert out[1, 0] == 4  # forest → 4
        assert out[1, 1] == 5  # mountain → 5
        assert out[1, 2] == 0  # ocean → 0


# ---------------------------------------------------------------------------
# compute_feature_maps
# ---------------------------------------------------------------------------

class TestComputeFeatureMaps:
    def _simple_grid(self, h=5, w=5, civ_code=1):
        """Grid with one civ cell at (2,2) surrounded by plains."""
        g = np.full((h, w), 11, dtype=np.int32)  # plains
        g[2, 2] = civ_code
        return g

    def test_returns_dict(self):
        g = self._simple_grid()
        result = compute_feature_maps(g)
        assert isinstance(result, dict)

    def test_has_expected_keys(self):
        g = self._simple_grid()
        result = compute_feature_maps(g)
        for key in ("ig_classes", "dist_civ", "n_ocean", "n_civ", "coast", "is_forest", "is_plains", "static"):
            assert key in result, f"Missing key: {key}"

    def test_dist_civ_zero_at_civ_cell(self):
        g = self._simple_grid(civ_code=1)
        result = compute_feature_maps(g)
        assert result["dist_civ"][2, 2] == pytest.approx(0.0)

    def test_dist_civ_positive_away_from_civ(self):
        g = self._simple_grid(civ_code=1)
        result = compute_feature_maps(g)
        assert result["dist_civ"][0, 0] > 0

    def test_no_civ_sets_dist_to_99(self):
        g = np.full((4, 4), 11, dtype=np.int32)  # all plains
        result = compute_feature_maps(g)
        assert result["dist_civ"][0, 0] == pytest.approx(99.0)

    def test_ocean_cells_mark_static(self):
        g = np.full((4, 4), 11, dtype=np.int32)
        g[0, 0] = 10  # ocean
        result = compute_feature_maps(g)
        assert result["static"][0, 0]

    def test_mountain_cells_mark_static(self):
        g = np.full((4, 4), 11, dtype=np.int32)
        g[1, 1] = 5  # mountain
        result = compute_feature_maps(g)
        assert result["static"][1, 1]

    def test_plains_not_static(self):
        g = np.full((4, 4), 11, dtype=np.int32)
        result = compute_feature_maps(g)
        assert not result["static"][2, 2]

    def test_n_ocean_zero_no_ocean(self):
        g = np.full((4, 4), 11, dtype=np.int32)
        result = compute_feature_maps(g)
        assert result["n_ocean"][1, 1] == pytest.approx(0.0)

    def test_n_ocean_positive_near_ocean(self):
        g = np.full((4, 4), 11, dtype=np.int32)
        g[0, 0] = 10  # ocean at corner
        result = compute_feature_maps(g)
        # Cell (0,1) is adjacent to ocean at (0,0) — should have n_ocean >= 1
        assert result["n_ocean"][0, 1] >= 1

    def test_is_forest_marks_forest_cells(self):
        g = np.full((4, 4), 11, dtype=np.int32)
        g[1, 2] = 4  # forest
        result = compute_feature_maps(g)
        assert result["is_forest"][1, 2] == pytest.approx(1.0)
        assert result["is_forest"][0, 0] == pytest.approx(0.0)

    def test_is_plains_marks_plains_cells(self):
        g = np.full((4, 4), 11, dtype=np.int32)  # all plains
        result = compute_feature_maps(g)
        assert result["is_plains"][0, 0] == pytest.approx(1.0)

    def test_ig_classes_shape(self):
        g = self._simple_grid(h=4, w=6)
        result = compute_feature_maps(g)
        assert result["ig_classes"].shape == (4, 6)


# ---------------------------------------------------------------------------
# features_to_tensor
# ---------------------------------------------------------------------------

class TestFeaturesToTensor:
    def _simple_feats(self, h=4, w=4):
        ig = np.full((h, w), 11, dtype=np.int32)
        return compute_feature_maps(ig)

    def test_returns_ndarray(self):
        feats = self._simple_feats()
        result = features_to_tensor(feats)
        assert isinstance(result, np.ndarray)

    def test_output_shape_12_channels(self):
        h, w = 5, 6
        feats = self._simple_feats(h, w)
        result = features_to_tensor(feats)
        assert result.shape == (12, h, w)

    def test_output_dtype_float32(self):
        feats = self._simple_feats()
        result = features_to_tensor(feats)
        assert result.dtype == np.float32

    def test_dist_civ_channel_normalized(self):
        # Channel 6 is clipped dist_civ / 20 → should be in [0, 1]
        feats = self._simple_feats()
        result = features_to_tensor(feats)
        assert result[6].min() >= 0.0
        assert result[6].max() <= 1.0

    def test_one_hot_channels_sum_to_one(self):
        # Channels 0-5 are one-hot → should sum to 1 per cell
        feats = self._simple_feats()
        result = features_to_tensor(feats)
        onehot = result[:N_CLASSES]  # first 6 channels
        cell_sums = onehot.sum(axis=0)
        np.testing.assert_allclose(cell_sums, 1.0, atol=1e-5)

    def test_ocean_n_channel_in_range(self):
        # Channel 7 is n_ocean / 8 → should be in [0, 1]
        feats = self._simple_feats()
        result = features_to_tensor(feats)
        assert result[7].min() >= 0.0
        assert result[7].max() <= 1.0

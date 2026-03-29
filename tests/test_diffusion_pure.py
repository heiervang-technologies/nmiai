"""Pure-function tests for diffusion_model.py and diffusion_ca.py.

Covers:
  diffusion_model: floor_and_normalize, grid_hash, initial_class_index,
                   deterministic_static_distribution, static_mask,
                   normalized_distance, prior_state_from_grid, encode_grid
  diffusion_ca: code_to_class_grid, compute_feature_maps, features_to_tensor

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock torch before importing diffusion modules
_torch_mock = MagicMock()
_torch_mock.cuda.is_available.return_value = False
_torch_mock.device.return_value = "cpu"
for _mod in ("torch", "torch.nn", "torch.nn.functional"):
    sys.modules.setdefault(_mod, MagicMock())
sys.modules["torch"] = _torch_mock
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

import diffusion_model as dm
import diffusion_ca as dc


GRID_SIZE = 40
N_CLASSES = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_grid(size: int = GRID_SIZE, code: int = 11) -> np.ndarray:
    g = np.full((size, size), code, dtype=np.int32)
    g[5, 5] = 1    # settlement
    g[5, 6] = 2    # port
    g[10, 10] = 4  # forest
    g[0, 0] = 10   # ocean
    g[-1, -1] = 5  # mountain
    return g


def _simple_priors() -> dict[int, np.ndarray]:
    """Minimal code_priors dict with the correct keys."""
    priors = {}
    for code in dm.CELL_CODES:
        p = np.ones(N_CLASSES, dtype=np.float32) / N_CLASSES
        priors[code] = p
    # Override ocean/mountain
    priors[10] = np.array([1, 0, 0, 0, 0, 0], dtype=np.float32)
    priors[5] = np.array([0, 0, 0, 0, 0, 1], dtype=np.float32)
    return priors


# ---------------------------------------------------------------------------
# diffusion_model.floor_and_normalize
# ---------------------------------------------------------------------------

class TestFloorAndNormalize:
    def test_sums_to_one(self):
        arr = np.ones((3, 3, N_CLASSES), dtype=np.float32)
        result = dm.floor_and_normalize(arr)
        np.testing.assert_allclose(result.sum(axis=2), 1.0, atol=1e-6)

    def test_floor_applied(self):
        arr = np.zeros((2, 2, N_CLASSES), dtype=np.float32)
        arr[:, :, 0] = 1.0
        result = dm.floor_and_normalize(arr)
        assert (result > 0).all()

    def test_shape_preserved(self):
        arr = np.random.rand(4, 4, N_CLASSES).astype(np.float32)
        assert dm.floor_and_normalize(arr).shape == (4, 4, N_CLASSES)


# ---------------------------------------------------------------------------
# diffusion_model.grid_hash
# ---------------------------------------------------------------------------

class TestGridHash:
    def test_returns_string(self):
        g = np.zeros((5, 5), dtype=np.int16)
        assert isinstance(dm.grid_hash(g), str)

    def test_same_grid_same_hash(self):
        g = _uniform_grid()
        assert dm.grid_hash(g) == dm.grid_hash(g.copy())

    def test_different_grid_different_hash(self):
        g1 = np.zeros((5, 5), dtype=np.int16)
        g2 = np.ones((5, 5), dtype=np.int16)
        assert dm.grid_hash(g1) != dm.grid_hash(g2)

    def test_accepts_list_input(self):
        g = [[0, 1], [10, 5]]
        assert isinstance(dm.grid_hash(g), str)


# ---------------------------------------------------------------------------
# diffusion_model.initial_class_index
# ---------------------------------------------------------------------------

class TestInitialClassIndex:
    @pytest.mark.parametrize("code,expected", [
        (0, 0), (10, 0), (11, 0),
        (1, 1), (2, 2), (4, 4), (5, 5),
    ])
    def test_known_codes(self, code, expected):
        assert dm.initial_class_index(code) == expected

    def test_unknown_returns_zero(self):
        assert dm.initial_class_index(99) == 0


# ---------------------------------------------------------------------------
# diffusion_model.deterministic_static_distribution
# ---------------------------------------------------------------------------

class TestDeterministicStaticDistribution:
    def test_shape(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        dist = dm.deterministic_static_distribution(g)
        assert dist.shape == (GRID_SIZE, GRID_SIZE, N_CLASSES)

    def test_ocean_cell_class0(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 10, dtype=np.int32)
        dist = dm.deterministic_static_distribution(g)
        assert (dist[:, :, 0] == 1.0).all()
        assert (dist[:, :, 1:] == 0.0).all()

    def test_mountain_cell_class5(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 5, dtype=np.int32)
        dist = dm.deterministic_static_distribution(g)
        assert (dist[:, :, 5] == 1.0).all()

    def test_settlement_cell_class1(self):
        g = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
        g[20, 20] = 1
        dist = dm.deterministic_static_distribution(g)
        assert dist[20, 20, 1] == 1.0

    def test_each_cell_one_hot(self):
        g = _uniform_grid()
        dist = dm.deterministic_static_distribution(g)
        sums = dist.sum(axis=2)
        assert (sums == 1.0).all()


# ---------------------------------------------------------------------------
# diffusion_model.static_mask
# ---------------------------------------------------------------------------

class TestStaticMask:
    def test_ocean_is_one(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        g[5, 5] = 10
        mask = dm.static_mask(g)
        assert mask[5, 5] == pytest.approx(1.0)

    def test_mountain_is_one(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        g[3, 3] = 5
        mask = dm.static_mask(g)
        assert mask[3, 3] == pytest.approx(1.0)

    def test_plains_is_zero(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        mask = dm.static_mask(g)
        assert (mask == 0.0).all()

    def test_shape(self):
        g = _uniform_grid()
        assert dm.static_mask(g).shape == (GRID_SIZE, GRID_SIZE)


# ---------------------------------------------------------------------------
# diffusion_model.normalized_distance
# ---------------------------------------------------------------------------

class TestNormalizedDistance:
    def test_returns_zero_to_one(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        result = dm.normalized_distance(mask)
        assert result.min() >= 0.0
        assert result.max() <= 1.0 + 1e-6

    def test_all_false_returns_ones(self):
        mask = np.zeros((5, 5), dtype=bool)
        result = dm.normalized_distance(mask)
        assert (result == 1.0).all()

    def test_seed_cell_distance_zero(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[5, 5] = True
        result = dm.normalized_distance(mask)
        assert result[5, 5] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# diffusion_model.prior_state_from_grid
# ---------------------------------------------------------------------------

class TestPriorStateFromGrid:
    def test_shape(self):
        g = _uniform_grid()
        priors = _simple_priors()
        result = dm.prior_state_from_grid(g, priors)
        assert result.shape == (GRID_SIZE, GRID_SIZE, N_CLASSES)

    def test_sums_to_one(self):
        g = _uniform_grid()
        priors = _simple_priors()
        result = dm.prior_state_from_grid(g, priors)
        np.testing.assert_allclose(result.sum(axis=2), 1.0, atol=1e-5)

    def test_ocean_gets_ocean_prior(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 10, dtype=np.int32)
        priors = _simple_priors()
        result = dm.prior_state_from_grid(g, priors)
        # ocean prior is [1,0,0,0,0,0]
        assert result[0, 0, 0] > 0.9


# ---------------------------------------------------------------------------
# diffusion_model.encode_grid
# ---------------------------------------------------------------------------

class TestEncodeGrid:
    def test_shape(self):
        g = _uniform_grid()
        priors = _simple_priors()
        enc = dm.encode_grid(g, priors)
        # CELL_CODES has 7 codes, prior has 6, then pos + dist features
        assert enc.ndim == 3
        assert enc.shape[1] == GRID_SIZE
        assert enc.shape[2] == GRID_SIZE

    def test_dtype_float32(self):
        g = _uniform_grid()
        priors = _simple_priors()
        enc = dm.encode_grid(g, priors)
        assert enc.dtype == np.float32

    def test_finite_values(self):
        g = _uniform_grid()
        priors = _simple_priors()
        enc = dm.encode_grid(g, priors)
        assert np.isfinite(enc).all()


# ---------------------------------------------------------------------------
# diffusion_ca.code_to_class_grid
# ---------------------------------------------------------------------------

class TestCodeToClassGrid:
    def test_known_codes(self):
        g = np.array([[0, 1, 2, 3, 4, 5, 10, 11]], dtype=np.int32)
        out = dc.code_to_class_grid(g)
        np.testing.assert_array_equal(out, [[0, 1, 2, 3, 4, 5, 0, 0]])

    def test_shape_preserved(self):
        g = _uniform_grid()
        assert dc.code_to_class_grid(g).shape == g.shape

    def test_dtype_int64(self):
        g = np.zeros((3, 3), dtype=np.int32)
        assert dc.code_to_class_grid(g).dtype == np.int64


# ---------------------------------------------------------------------------
# diffusion_ca.compute_feature_maps
# ---------------------------------------------------------------------------

class TestComputeFeatureMaps:
    def test_returns_dict_with_required_keys(self):
        g = _uniform_grid()
        feats = dc.compute_feature_maps(g)
        for key in ("ig_classes", "dist_civ", "n_ocean", "n_civ", "coast",
                    "is_forest", "is_plains", "static"):
            assert key in feats

    def test_shapes_match_grid(self):
        g = _uniform_grid()
        feats = dc.compute_feature_maps(g)
        for key, val in feats.items():
            assert val.shape == (GRID_SIZE, GRID_SIZE), f"Wrong shape for {key}"

    def test_static_mask_ocean_and_mountain(self):
        g = _uniform_grid()
        feats = dc.compute_feature_maps(g)
        assert feats["static"][0, 0]    # ocean
        assert feats["static"][-1, -1]  # mountain

    def test_is_forest_marks_forest_cell(self):
        g = _uniform_grid()
        feats = dc.compute_feature_maps(g)
        assert feats["is_forest"][10, 10] == pytest.approx(1.0)

    def test_coast_nonzero_near_ocean(self):
        g = _uniform_grid()
        feats = dc.compute_feature_maps(g)
        # At least some coast cells near ocean (0,0)
        assert feats["coast"].any()


# ---------------------------------------------------------------------------
# diffusion_ca.features_to_tensor
# ---------------------------------------------------------------------------

class TestFeaturesToTensor:
    def test_shape_C_H_W(self):
        g = _uniform_grid()
        feats = dc.compute_feature_maps(g)
        tensor = dc.features_to_tensor(feats)
        C, H, W = tensor.shape
        assert H == GRID_SIZE
        assert W == GRID_SIZE
        assert C == dc.N_FEAT_CH  # 12 channels

    def test_dtype_float32(self):
        g = _uniform_grid()
        tensor = dc.features_to_tensor(dc.compute_feature_maps(g))
        assert tensor.dtype == np.float32

    def test_finite_values(self):
        g = _uniform_grid()
        tensor = dc.features_to_tensor(dc.compute_feature_maps(g))
        assert np.isfinite(tensor).all()

    def test_dist_channel_normalized(self):
        g = _uniform_grid()
        feats = dc.compute_feature_maps(g)
        tensor = dc.features_to_tensor(feats)
        # Channel 6 is dist_civ / 20 clipped to [0,1]
        assert tensor[6].min() >= 0.0
        assert tensor[6].max() <= 1.0 + 1e-6

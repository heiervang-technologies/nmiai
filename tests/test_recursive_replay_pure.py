"""Pure-function tests for recursive_model.py and replay_boosted_predictor.py.

Covers:
  recursive_model: encode_grid, initial_state_from_grid
  replay_boosted_predictor: replay_to_onehot, lookup_residual

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock torch before importing recursive_model
_torch_mock = MagicMock()
_torch_mock.cuda.is_available.return_value = False
_torch_mock.device.return_value = "cpu"
sys.modules["torch"] = _torch_mock
sys.modules["torch.nn"] = MagicMock()
sys.modules["torch.nn.functional"] = MagicMock()

# Mock requests for replay_boosted_predictor
sys.modules.setdefault("requests", MagicMock())

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

import recursive_model as rm
import replay_boosted_predictor as rbp


GRID_SIZE = 40
N_CLASSES = 6
RM_N_CHANNELS = 13   # 7 one-hot + 2 static masks + 2 pos + 2 dist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_grid(size: int = GRID_SIZE, code: int = 11) -> np.ndarray:
    g = np.full((size, size), code, dtype=np.int32)
    g[5, 5] = 1     # settlement
    g[5, 6] = 2     # port
    g[10, 10] = 4   # forest
    g[0, 0] = 10    # ocean
    g[-1, -1] = 5   # mountain
    return g


def _make_replay(size: int = 5) -> dict:
    """Minimal replay dict with 2 frames."""
    frame = {"grid": [[11] * size for _ in range(size)]}
    frame["grid"][0][0] = 1   # settlement
    frame["grid"][1][1] = 10  # ocean
    return {
        "frames": [
            {"grid": [[11] * size for _ in range(size)]},
            {"grid": [list(row) for row in frame["grid"]]},
        ]
    }


# ---------------------------------------------------------------------------
# recursive_model.encode_grid
# ---------------------------------------------------------------------------

class TestEncodeGrid:
    def test_shape(self):
        g = _uniform_grid()
        enc = rm.encode_grid(g)
        assert enc.shape == (RM_N_CHANNELS, GRID_SIZE, GRID_SIZE)

    def test_dtype_float32(self):
        g = _uniform_grid()
        assert rm.encode_grid(g).dtype == np.float32

    def test_finite_values(self):
        g = _uniform_grid()
        assert np.isfinite(rm.encode_grid(g)).all()

    def test_one_hot_sums_at_each_cell(self):
        g = _uniform_grid()
        enc = rm.encode_grid(g)
        # First 7 channels are one-hot cell types; each cell should be assigned to exactly one type
        one_hot_sum = enc[:rm.N_CELL_TYPES].sum(axis=0)
        np.testing.assert_allclose(one_hot_sum, 1.0, atol=1e-6)

    def test_dist_channels_normalized(self):
        g = _uniform_grid()
        enc = rm.encode_grid(g)
        # Channels 11 and 12 are normalized distances (0–1)
        assert enc[11].min() >= 0.0
        assert enc[11].max() <= 1.0 + 1e-6
        assert enc[12].min() >= 0.0
        assert enc[12].max() <= 1.0 + 1e-6

    def test_ocean_mask_marks_ocean_cells(self):
        g = _uniform_grid()
        enc = rm.encode_grid(g)
        # Channel 7 is ocean_mask; (0,0) is ocean
        assert enc[7, 0, 0] == pytest.approx(1.0)
        assert enc[7, 5, 5] == pytest.approx(0.0)

    def test_mountain_mask_marks_mountain_cells(self):
        g = _uniform_grid()
        enc = rm.encode_grid(g)
        # Channel 8 is mountain_mask; (-1,-1) is mountain
        assert enc[8, -1, -1] == pytest.approx(1.0)
        assert enc[8, 5, 5] == pytest.approx(0.0)

    def test_no_civ_grid_all_dist_ones(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)  # no civ
        enc = rm.encode_grid(g)
        # With no civ, dist_civ = ones_like normalized → all 1
        np.testing.assert_allclose(enc[11], 1.0)


# ---------------------------------------------------------------------------
# recursive_model.initial_state_from_grid
# ---------------------------------------------------------------------------

class TestInitialStateFromGrid:
    def test_shape(self):
        g = _uniform_grid()
        state = rm.initial_state_from_grid(g)
        assert state.shape == (N_CLASSES, GRID_SIZE, GRID_SIZE)

    def test_dtype_float32(self):
        g = _uniform_grid()
        assert rm.initial_state_from_grid(g).dtype == np.float32

    def test_each_cell_one_hot(self):
        g = _uniform_grid()
        state = rm.initial_state_from_grid(g)
        sums = state.sum(axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_settlement_class1(self):
        g = _uniform_grid()
        state = rm.initial_state_from_grid(g)
        assert state[1, 5, 5] == pytest.approx(1.0)

    def test_port_class2(self):
        g = _uniform_grid()
        state = rm.initial_state_from_grid(g)
        assert state[2, 5, 6] == pytest.approx(1.0)

    def test_forest_class4(self):
        g = _uniform_grid()
        state = rm.initial_state_from_grid(g)
        assert state[4, 10, 10] == pytest.approx(1.0)

    def test_mountain_class5(self):
        g = _uniform_grid()
        state = rm.initial_state_from_grid(g)
        assert state[5, -1, -1] == pytest.approx(1.0)

    def test_ocean_class0(self):
        g = _uniform_grid()
        state = rm.initial_state_from_grid(g)
        assert state[0, 0, 0] == pytest.approx(1.0)

    def test_plains_class0(self):
        g = np.full((GRID_SIZE, GRID_SIZE), 11, dtype=np.int32)
        state = rm.initial_state_from_grid(g)
        assert (state[0] == 1.0).all()


# ---------------------------------------------------------------------------
# replay_boosted_predictor.replay_to_onehot
# ---------------------------------------------------------------------------

class TestReplayToOnehot:
    def test_shape(self):
        size = 5
        replay = _make_replay(size)
        out = rbp.replay_to_onehot(replay)
        assert out.shape == (size, size, N_CLASSES)

    def test_one_hot_per_cell(self):
        replay = _make_replay(4)
        out = rbp.replay_to_onehot(replay)
        np.testing.assert_allclose(out.sum(axis=2), 1.0)

    def test_settlement_class1(self):
        replay = _make_replay(4)
        replay["frames"][-1]["grid"][2][2] = 1
        out = rbp.replay_to_onehot(replay)
        assert out[2, 2, 1] == pytest.approx(1.0)

    def test_ocean_class0(self):
        replay = _make_replay(4)
        replay["frames"][-1]["grid"][0][0] = 10
        out = rbp.replay_to_onehot(replay)
        assert out[0, 0, 0] == pytest.approx(1.0)

    def test_mountain_class5(self):
        replay = _make_replay(4)
        replay["frames"][-1]["grid"][3][3] = 5
        out = rbp.replay_to_onehot(replay)
        assert out[3, 3, 5] == pytest.approx(1.0)

    def test_uses_last_frame(self):
        # Final frame differs from initial; should reflect final state
        replay = {
            "frames": [
                {"grid": [[11, 11], [11, 11]]},  # all plains
                {"grid": [[1, 11], [11, 11]]},    # settlement in final
            ]
        }
        out = rbp.replay_to_onehot(replay)
        assert out[0, 0, 1] == pytest.approx(1.0)  # settlement → class 1


# ---------------------------------------------------------------------------
# replay_boosted_predictor.lookup_residual
# ---------------------------------------------------------------------------

class TestLookupResidual:
    def _make_residuals(self):
        vec = np.ones(N_CLASSES, dtype=np.float64) / N_CLASSES
        residuals = [
            {("S", 2, 0, 1, 1): vec},    # fine
            {("S", 2, 0, 1): vec},        # mid
            {("S", 2, 1): vec},            # coarse
            {("S", 2): vec},               # broad
        ]
        residual_counts = [
            {("S", 2, 0, 1, 1): 25},
            {("S", 2, 0, 1): 35},
            {("S", 2, 1): 60},
            {("S", 2): 15},
        ]
        return residuals, residual_counts

    def test_returns_fine_when_sufficient(self):
        res, counts = self._make_residuals()
        keys = (("S", 2, 0, 1, 1), ("S", 2, 0, 1), ("S", 2, 1), ("S", 2))
        _, c, level = rbp.lookup_residual(res, counts, keys)
        assert level == 0

    def test_skips_insufficient_support(self):
        res, counts = self._make_residuals()
        counts[0][("S", 2, 0, 1, 1)] = 5   # below default min_count=20
        counts[1][("S", 2, 0, 1)] = 35     # above mid min_count=30
        keys = (("S", 2, 0, 1, 1), ("S", 2, 0, 1), ("S", 2, 1), ("S", 2))
        _, _, level = rbp.lookup_residual(res, counts, keys)
        assert level == 1

    def test_no_match_returns_none(self):
        res = [{}, {}, {}, {}]
        counts = [{}, {}, {}, {}]
        keys = (("X",), ("X",), ("X",), ("X",))
        val, _, _ = rbp.lookup_residual(res, counts, keys)
        assert val is None

    def test_returns_array_on_match(self):
        res, counts = self._make_residuals()
        keys = (("S", 2, 0, 1, 1), ("S", 2, 0, 1), ("S", 2, 1), ("S", 2))
        val, _, _ = rbp.lookup_residual(res, counts, keys)
        assert isinstance(val, np.ndarray)
        assert val.shape == (N_CLASSES,)

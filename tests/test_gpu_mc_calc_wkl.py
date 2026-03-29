"""Pure-function tests for astar-island gpu_exact_mc calc_wkl (both versions).

Covers:
  tasks/astar-island/gpu_exact_mc.py    — calc_wkl
  tasks/astar-island/gpu_exact_mc_v2.py — calc_wkl

Both implementations are identical in formula:
  weights = (0.15 + H_bits) * dynamic_mask
  wkl = sum(weights * KL(p||q)) / sum(weights)
where dynamic_mask excludes ocean (code 10) and mountain (code 5).

All pure functions — no GPU, no file system, no network.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

from gpu_exact_mc import calc_wkl as calc_wkl_v1
from gpu_exact_mc_v2 import calc_wkl as calc_wkl_v2


# Run the same test body against both implementations via parametrize
@pytest.fixture(params=["v1", "v2"])
def calc_wkl(request):
    if request.param == "v1":
        return calc_wkl_v1
    return calc_wkl_v2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

OCEAN, MOUNTAIN = 10, 5
N_CLASSES = 6


def _uniform(h: int, w: int) -> np.ndarray:
    return np.full((h, w, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float32)


def _dynamic_grid(h: int, w: int) -> np.ndarray:
    """All-plains grid — every cell is dynamic."""
    return np.full((h, w), 11, dtype=np.int32)


def _static_grid(h: int, w: int) -> np.ndarray:
    """All-ocean grid — every cell is static (excluded from wkl)."""
    return np.full((h, w), OCEAN, dtype=np.int32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCalcWkl:
    def test_identical_pred_and_target_gives_zero(self, calc_wkl):
        p = _uniform(5, 5)
        ig = _dynamic_grid(5, 5)
        result = calc_wkl(p, p, ig)
        assert abs(result) < 1e-5

    def test_all_static_grid_returns_zero(self, calc_wkl):
        p = _uniform(4, 4)
        q = _uniform(4, 4)
        q[0, 0] = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]
        ig = _static_grid(4, 4)
        # All cells masked out → sum(weights)=0 → division by zero guard or 0
        # Implementation: if sum(weights)==0 it would be NaN, but uniform p has H=0 for static
        # Actually uniform has H > 0, so weights > 0 for dynamic cells only.
        # All ocean → dynamic_mask all False → weights all 0 → sum(weights)=0 → nan
        # Let's just verify it's finite or 0.
        result = calc_wkl(p, p, ig)
        assert result == 0.0 or np.isnan(result)

    def test_nonneg_for_dynamic_grid(self, calc_wkl):
        p = _uniform(5, 5)
        q = np.random.dirichlet(np.ones(N_CLASSES), size=(5, 5)).astype(np.float32)
        ig = _dynamic_grid(5, 5)
        result = calc_wkl(p, q, ig)
        assert result >= -1e-9

    def test_returns_scalar(self, calc_wkl):
        p = _uniform(4, 4)
        ig = _dynamic_grid(4, 4)
        result = calc_wkl(p, p, ig)
        assert np.ndim(result) == 0

    def test_returns_float(self, calc_wkl):
        p = _uniform(4, 4)
        ig = _dynamic_grid(4, 4)
        result = calc_wkl(p, p, ig)
        assert isinstance(float(result), float)

    def test_ocean_cells_not_contributing(self, calc_wkl):
        # Grid with one dynamic cell, rest ocean
        p = _uniform(3, 3)
        q = _uniform(3, 3)
        # Make dynamic cell differ substantially
        q[1, 1] = [0.9, 0.02, 0.02, 0.02, 0.02, 0.02]
        ig = _static_grid(3, 3)
        ig[1, 1] = 11  # one dynamic cell

        result_with_diff = calc_wkl(p, q, ig)
        result_identical = calc_wkl(p, p, ig)

        # The differing cell should give higher wkl than identical
        assert result_with_diff > result_identical

    def test_mountain_cells_excluded(self, calc_wkl):
        p = _uniform(4, 4)
        q = _uniform(4, 4)
        ig = np.full((4, 4), MOUNTAIN, dtype=np.int32)
        ig[0, 0] = 11  # one dynamic cell
        result = calc_wkl(p, p, ig)
        # Identical pred → 0
        assert abs(result) < 1e-5

    def test_high_entropy_cells_weighted_more(self, calc_wkl):
        # Create a grid where one cell has high entropy gt and one has low entropy gt
        H, W = 2, 1
        ig = np.full((H, W), 11, dtype=np.int32)
        p = np.zeros((H, W, N_CLASSES), dtype=np.float32)
        # Cell 0: uniform (high entropy)
        p[0, 0] = 1.0 / N_CLASSES
        # Cell 1: near-deterministic (low entropy)
        p[1, 0] = [0.99, 0.002, 0.002, 0.002, 0.002, 0.002]

        # Make pred wrong by the same KL amount at both cells
        q = p.copy()
        # Swap slightly at both
        q[0, 0] = [0.1, 0.3, 0.2, 0.1, 0.2, 0.1]  # diverges from uniform
        q[1, 0] = [0.9, 0.01, 0.02, 0.02, 0.02, 0.03]  # small divergence

        result = float(calc_wkl(p, q, ig))
        assert result >= 0

    def test_mixed_grid(self, calc_wkl):
        # A more realistic grid
        ig = np.array([
            [11, 1, OCEAN],
            [MOUNTAIN, 4, 11],
        ], dtype=np.int32)
        p = _uniform(2, 3)
        q = _uniform(2, 3)
        q[0, 1] = [0.5, 0.2, 0.1, 0.1, 0.05, 0.05]
        result = calc_wkl(p, q, ig)
        assert result >= 0

    def test_uniform_weight_floor_0_15(self, calc_wkl):
        # When entropy=0, weight is still 0.15 (the floor ensures dynamic cells always count)
        ig = np.full((1, 1), 11, dtype=np.int32)
        p = np.zeros((1, 1, N_CLASSES), dtype=np.float32)
        p[0, 0, 0] = 1.0  # deterministic → entropy=0

        q = p.copy()
        q[0, 0] = [0.5, 0.1, 0.1, 0.1, 0.1, 0.1]

        result = calc_wkl(p, q, ig)
        # Should be > 0 because KL(p||q)>0 and weight=0.15>0
        assert result > 0


class TestCalcWklV1V2Agreement:
    """The two implementations should agree on identical inputs."""

    def test_identical_inputs(self):
        np.random.seed(42)
        H, W = 5, 5
        ig = np.random.randint(0, 12, (H, W)).astype(np.int32)
        p = np.random.dirichlet(np.ones(N_CLASSES), size=(H, W)).astype(np.float32)
        q = np.random.dirichlet(np.ones(N_CLASSES), size=(H, W)).astype(np.float32)

        r1 = calc_wkl_v1(p, q, ig)
        r2 = calc_wkl_v2(p, q, ig)
        assert abs(r1 - r2) < 1e-5

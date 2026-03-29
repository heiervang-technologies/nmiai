"""Tests for tasks/astar-island pure helper functions.

Covers:
- adaptive_query: initial_class_grid, static_mask, viewport_cells, overlap_ratio
- calibrated_predictor: entropy_bits, temperature_scale, fit_temperature

All pure functions requiring only numpy — no file I/O or GPU needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Add astar-island to path for local imports (predictor, neighborhood_predictor, etc.)
_ASTAR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR)

from adaptive_query import (
    initial_class_grid,
    overlap_ratio,
    static_mask,
    viewport_cells,
)
from calibrated_predictor import entropy_bits, fit_temperature, temperature_scale


# ---------------------------------------------------------------------------
# viewport_cells
# ---------------------------------------------------------------------------

class TestViewportCells:
    def test_returns_set(self):
        result = viewport_cells((0, 0, 2, 2))
        assert isinstance(result, set)

    def test_width_height_correct(self):
        cells = viewport_cells((0, 0, 3, 2))
        # w=3, h=2 → 6 cells
        assert len(cells) == 6

    def test_offset_x_y(self):
        cells = viewport_cells((5, 3, 2, 2))
        # y in [3,4], x in [5,6]
        assert (3, 5) in cells
        assert (4, 6) in cells

    def test_unit_viewport(self):
        cells = viewport_cells((2, 3, 1, 1))
        assert cells == {(3, 2)}

    def test_cells_contain_tuples(self):
        cells = viewport_cells((0, 0, 2, 2))
        assert all(isinstance(c, tuple) and len(c) == 2 for c in cells)


# ---------------------------------------------------------------------------
# overlap_ratio
# ---------------------------------------------------------------------------

class TestOverlapRatio:
    def test_identical_viewports_full_overlap(self):
        vp = (0, 0, 5, 5)
        assert overlap_ratio(vp, vp) == pytest.approx(1.0)

    def test_no_overlap_returns_zero(self):
        a = (0, 0, 5, 5)
        b = (10, 10, 5, 5)
        assert overlap_ratio(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = (0, 0, 4, 4)  # cells (0-3, 0-3)
        b = (2, 0, 4, 4)  # cells (0-3, 2-5) — overlaps in x=2,3
        ratio = overlap_ratio(a, b)
        assert 0.0 < ratio < 1.0

    def test_returns_float(self):
        result = overlap_ratio((0, 0, 3, 3), (0, 0, 3, 3))
        assert isinstance(result, float)

    def test_ratio_between_zero_and_one(self):
        a = (0, 0, 5, 5)
        b = (3, 0, 5, 5)
        assert 0.0 <= overlap_ratio(a, b) <= 1.0


# ---------------------------------------------------------------------------
# initial_class_grid
# ---------------------------------------------------------------------------

class TestInitialClassGrid:
    def _grid(self, values):
        return np.array(values, dtype=np.int32)

    def test_returns_ndarray(self):
        result = initial_class_grid(self._grid([[0, 1], [2, 4]]))
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        grid = self._grid([[0, 1, 2], [4, 5, 10]])
        result = initial_class_grid(grid)
        assert result.shape == (2, 3)

    def test_cell_1_maps_to_1(self):
        grid = self._grid([[1]])
        result = initial_class_grid(grid)
        assert result[0, 0] == 1

    def test_cell_2_maps_to_2(self):
        grid = self._grid([[2]])
        result = initial_class_grid(grid)
        assert result[0, 0] == 2

    def test_cell_4_maps_to_4(self):
        grid = self._grid([[4]])
        result = initial_class_grid(grid)
        assert result[0, 0] == 4

    def test_cell_5_maps_to_5(self):
        grid = self._grid([[5]])
        result = initial_class_grid(grid)
        assert result[0, 0] == 5

    def test_ocean_0_maps_to_0(self):
        grid = self._grid([[0]])
        result = initial_class_grid(grid)
        assert result[0, 0] == 0

    def test_ocean_10_maps_to_0(self):
        grid = self._grid([[10]])
        result = initial_class_grid(grid)
        assert result[0, 0] == 0

    def test_accepts_list_input(self):
        result = initial_class_grid([[1, 2], [4, 5]])
        assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# static_mask
# ---------------------------------------------------------------------------

class TestStaticMask:
    def test_returns_boolean_array(self):
        grid = np.array([[5, 1], [10, 2]])
        result = static_mask(grid)
        assert result.dtype == bool

    def test_cell_5_is_static(self):
        grid = np.array([[5]])
        assert static_mask(grid)[0, 0] is np.bool_(True)

    def test_cell_10_is_static(self):
        grid = np.array([[10]])
        assert static_mask(grid)[0, 0] is np.bool_(True)

    def test_cell_1_not_static(self):
        grid = np.array([[1]])
        assert static_mask(grid)[0, 0] is np.bool_(False)

    def test_cell_0_not_static(self):
        grid = np.array([[0]])
        assert static_mask(grid)[0, 0] is np.bool_(False)

    def test_shape_preserved(self):
        grid = np.array([[5, 1, 10], [2, 0, 4]])
        result = static_mask(grid)
        assert result.shape == (2, 3)


# ---------------------------------------------------------------------------
# entropy_bits
# ---------------------------------------------------------------------------

class TestEntropyBits:
    def test_uniform_distribution_max_entropy(self):
        prob = np.array([0.25, 0.25, 0.25, 0.25])
        result = entropy_bits(prob)
        assert result == pytest.approx(2.0)  # log2(4) = 2 bits

    def test_deterministic_distribution_zero_entropy(self):
        prob = np.array([1.0, 0.0, 0.0, 0.0])
        result = entropy_bits(prob)
        assert result == pytest.approx(0.0, abs=1e-8)

    def test_returns_float(self):
        prob = np.array([0.5, 0.5])
        assert isinstance(entropy_bits(prob), float)

    def test_entropy_nonnegative(self):
        prob = np.array([0.3, 0.4, 0.3])
        assert entropy_bits(prob) >= 0.0

    def test_binary_50_50_is_one_bit(self):
        prob = np.array([0.5, 0.5])
        assert entropy_bits(prob) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# temperature_scale
# ---------------------------------------------------------------------------

class TestTemperatureScale:
    def test_returns_ndarray(self):
        prob = np.array([0.4, 0.3, 0.3])
        result = temperature_scale(prob, 1.0)
        assert isinstance(result, np.ndarray)

    def test_sums_to_one(self):
        prob = np.array([0.5, 0.3, 0.2])
        result = temperature_scale(prob, 1.5)
        assert result.sum() == pytest.approx(1.0)

    def test_temperature_1_preserves_relative_order(self):
        prob = np.array([0.6, 0.3, 0.1])
        result = temperature_scale(prob, 1.0)
        assert result[0] > result[1] > result[2]

    def test_high_temperature_flattens_distribution(self):
        prob = np.array([0.8, 0.1, 0.1])
        low_t = temperature_scale(prob, 0.1)
        high_t = temperature_scale(prob, 5.0)
        # High temperature makes distribution more uniform
        assert entropy_bits(high_t) > entropy_bits(low_t)

    def test_output_shape_matches_input(self):
        prob = np.array([0.2, 0.3, 0.5])
        result = temperature_scale(prob, 2.0)
        assert result.shape == prob.shape


# ---------------------------------------------------------------------------
# fit_temperature
# ---------------------------------------------------------------------------

class TestFitTemperature:
    def test_returns_float(self):
        prob = np.array([0.6, 0.2, 0.1, 0.1])
        result = fit_temperature(prob, 1.5)
        assert isinstance(result, float)

    def test_temperature_positive(self):
        prob = np.array([0.5, 0.3, 0.2])
        t = fit_temperature(prob, 1.0)
        assert t > 0.0

    def test_already_matching_entropy_returns_one(self):
        # If target matches current entropy, temperature ≈ 1.0
        prob = np.array([0.4, 0.35, 0.25])
        current_entropy = entropy_bits(prob)
        t = fit_temperature(prob, current_entropy)
        assert t == pytest.approx(1.0, abs=0.01)

    def test_target_entropy_achieved(self):
        prob = np.array([0.7, 0.2, 0.05, 0.05])
        target = entropy_bits(np.array([0.4, 0.3, 0.2, 0.1]))
        t = fit_temperature(prob, target)
        scaled = temperature_scale(prob, t)
        achieved = entropy_bits(scaled)
        assert achieved == pytest.approx(target, abs=0.05)

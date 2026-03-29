"""Tests for project_progress.py and combined_predictor.py pure helpers.

Covers:
  - project_progress: to_float, fit_line, predict
  - combined_predictor: cell_code_to_class, detect_survival
All pure functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from project_progress import fit_line, predict, to_float
from combined_predictor import cell_code_to_class, detect_survival


# ---------------------------------------------------------------------------
# project_progress.to_float
# ---------------------------------------------------------------------------

class TestToFloat:
    def test_valid_integer_string(self):
        assert to_float("42") == 42.0

    def test_valid_float_string(self):
        assert abs(to_float("3.14") - 3.14) < 1e-9

    def test_none_gives_none(self):
        assert to_float(None) is None

    def test_empty_string_gives_none(self):
        assert to_float("") is None

    def test_dash_gives_none(self):
        assert to_float("-") is None

    def test_pending_gives_none(self):
        assert to_float("pending") is None

    def test_non_numeric_gives_none(self):
        assert to_float("abc") is None

    def test_negative_number(self):
        result = to_float("-1.5")
        assert result is not None
        assert abs(result - (-1.5)) < 1e-9

    def test_whitespace_stripped(self):
        result = to_float("  2.5  ")
        assert result is not None
        assert abs(result - 2.5) < 1e-9

    def test_zero(self):
        result = to_float("0")
        assert result == 0.0


# ---------------------------------------------------------------------------
# project_progress.fit_line
# ---------------------------------------------------------------------------

class TestFitLine:
    def test_returns_3_tuple(self):
        result = fit_line([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert len(result) == 3

    def test_perfect_linear_slope(self):
        slope, intercept, rmse = fit_line([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert abs(slope - 2.0) < 1e-6

    def test_perfect_linear_intercept(self):
        slope, intercept, rmse = fit_line([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert abs(intercept - 0.0) < 1e-6

    def test_perfect_fit_zero_rmse(self):
        slope, intercept, rmse = fit_line([0.0, 1.0, 2.0], [1.0, 3.0, 5.0])
        assert abs(rmse) < 1e-6

    def test_noisy_data_positive_rmse(self):
        slope, intercept, rmse = fit_line([0.0, 1.0, 2.0], [1.0, 3.1, 4.9])
        assert rmse > 0.0

    def test_flat_data_zero_slope(self):
        slope, intercept, rmse = fit_line([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
        assert abs(slope) < 1e-6

    def test_flat_data_intercept_equals_y(self):
        slope, intercept, rmse = fit_line([1.0, 2.0, 3.0], [5.0, 5.0, 5.0])
        assert abs(intercept - 5.0) < 1e-6

    def test_negative_slope(self):
        slope, intercept, rmse = fit_line([1.0, 2.0, 3.0], [6.0, 4.0, 2.0])
        assert slope < 0.0


# ---------------------------------------------------------------------------
# project_progress.predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_basic_linear_prediction(self):
        # predict(intercept, slope, x) = intercept + slope * x
        assert abs(predict(1.0, 2.0, 3.0) - 7.0) < 1e-9

    def test_zero_slope_returns_intercept(self):
        assert abs(predict(5.0, 0.0, 100.0) - 5.0) < 1e-9

    def test_zero_x_returns_intercept(self):
        assert abs(predict(3.0, 7.0, 0.0) - 3.0) < 1e-9

    def test_negative_intercept(self):
        assert abs(predict(-2.0, 1.0, 2.0) - 0.0) < 1e-9


# ---------------------------------------------------------------------------
# combined_predictor.cell_code_to_class
# ---------------------------------------------------------------------------

class TestCombinedCellCodeToClass:
    def test_empty_maps_to_0(self):
        assert cell_code_to_class(0) == 0

    def test_ocean_maps_to_0(self):
        assert cell_code_to_class(10) == 0

    def test_plains_maps_to_0(self):
        assert cell_code_to_class(11) == 0

    def test_settlement_maps_to_1(self):
        assert cell_code_to_class(1) == 1

    def test_port_maps_to_2(self):
        assert cell_code_to_class(2) == 2

    def test_ruin_maps_to_3(self):
        assert cell_code_to_class(3) == 3

    def test_forest_maps_to_4(self):
        assert cell_code_to_class(4) == 4

    def test_mountain_maps_to_5(self):
        assert cell_code_to_class(5) == 5

    def test_unknown_maps_to_0(self):
        assert cell_code_to_class(99) == 0


# ---------------------------------------------------------------------------
# combined_predictor.detect_survival
# ---------------------------------------------------------------------------

class TestDetectSurvival:
    def _grid_with_settlement(self, sy=5, sx=5):
        """10×10 initial grid with one settlement at (sy, sx)."""
        g = np.full((10, 10), 11, dtype=np.int32)
        g[sy, sx] = 1  # settlement
        return g

    def _obs_with_cell(self, cell_code, y, x, vx=0, vy=0):
        """Single observation viewport containing one cell."""
        return {
            "viewport_x": vx,
            "viewport_y": vy,
            "grid": [[cell_code if (dy == y - vy and dx == x - vx) else 11
                      for dx in range(15)]
                     for dy in range(15)],
        }

    def test_no_observations_returns_default(self):
        g = self._grid_with_settlement()
        result = detect_survival(g.tolist(), [])
        assert result == 0.3

    def test_settlement_seen_alive_gives_one(self):
        g = self._grid_with_settlement(sy=2, sx=3)
        obs = self._obs_with_cell(cell_code=1, y=2, x=3)
        result = detect_survival(g.tolist(), [obs])
        assert abs(result - 1.0) < 1e-6

    def test_settlement_seen_dead_gives_zero(self):
        g = self._grid_with_settlement(sy=2, sx=3)
        # Cell code 11 (plains) where settlement was → dead
        obs = self._obs_with_cell(cell_code=11, y=2, x=3)
        result = detect_survival(g.tolist(), [obs])
        assert abs(result - 0.0) < 1e-6

    def test_returns_float(self):
        g = self._grid_with_settlement()
        result = detect_survival(g.tolist(), [])
        assert isinstance(result, float)

    def test_no_settlement_in_viewport_returns_default(self):
        # Settlement at (9, 9) but viewport covers (0, 0) to (14, 14)
        g = np.full((40, 40), 11, dtype=np.int32)
        g[39, 39] = 1  # far corner
        obs = {
            "viewport_x": 0, "viewport_y": 0,
            "grid": [[11] * 15 for _ in range(15)],
        }
        result = detect_survival(g.tolist(), [obs])
        assert result == 0.3

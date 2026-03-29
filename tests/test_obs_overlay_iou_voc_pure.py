"""Pure-function tests for astar-island benchmark_tau and object-detection eval_honest.

Covers:
  tasks/astar-island/benchmark_tau.py        — apply_obs_overlay
  tasks/object-detection/eval_honest.py      — iou, voc_ap

All pure functions — no file system, GPU, or network access.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
_OD_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "object-detection")
sys.path.insert(0, _ASTAR_DIR)

from benchmark_tau import apply_obs_overlay

# Load iou and voc_ap from eval_honest by explicit path to avoid cv2/onnx imports at module level
_EVAL_HONEST_PATH = str(Path(_OD_DIR) / "eval_honest.py")

def _load_func(module_name, path, func_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)

# eval_honest imports cv2 at module level — mock it first
import types
if "cv2" not in sys.modules:
    cv2_mock = types.ModuleType("cv2")
    cv2_mock.resize = lambda *a, **k: None
    cv2_mock.INTER_LINEAR = 1
    cv2_mock.BORDER_CONSTANT = 0
    cv2_mock.copyMakeBorder = lambda *a, **k: None
    sys.modules["cv2"] = cv2_mock
    cv2_mock.dnn = types.SimpleNamespace(NMSBoxesBatched=lambda *a, **k: [])

iou = _load_func("eval_honest", _EVAL_HONEST_PATH, "iou")
voc_ap = _load_func("eval_honest", _EVAL_HONEST_PATH, "voc_ap")


N_CLASSES = 6


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_pred(h: int = 10, w: int = 10) -> np.ndarray:
    return np.full((h, w, N_CLASSES), 1.0 / N_CLASSES, dtype=np.float64)


def _obs(vx: int, vy: int, grid_code: int = 1, h: int = 3, w: int = 3) -> dict:
    return {
        "viewport_x": vx,
        "viewport_y": vy,
        "grid": [[grid_code] * w for _ in range(h)],
    }


# ---------------------------------------------------------------------------
# benchmark_tau.apply_obs_overlay
# ---------------------------------------------------------------------------

class TestApplyObsOverlay:
    def test_no_observations_unchanged(self):
        pred = _uniform_pred()
        result = apply_obs_overlay(pred.copy(), None, [], tau=5)
        np.testing.assert_allclose(result, pred, atol=1e-6)

    def test_output_shape(self):
        pred = _uniform_pred()
        result = apply_obs_overlay(pred.copy(), None, [_obs(0, 0)], tau=5)
        assert result.shape == pred.shape

    def test_sums_to_one(self):
        pred = _uniform_pred()
        result = apply_obs_overlay(pred.copy(), None, [_obs(0, 0, 1)], tau=5)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((10, 10)), atol=1e-5)

    def test_nonneg(self):
        pred = _uniform_pred()
        result = apply_obs_overlay(pred.copy(), None, [_obs(0, 0, 1)], tau=5)
        assert (result >= 0).all()

    def test_strong_obs_shifts_probability(self):
        # With tau=1, a single settlement obs should increase class-1 probability
        pred = _uniform_pred()
        result = apply_obs_overlay(pred.copy(), None, [_obs(0, 0, 1, 1, 1)], tau=1)
        assert result[0, 0, 1] > 1.0 / N_CLASSES

    def test_unobserved_cells_unchanged(self):
        pred = _uniform_pred()
        result = apply_obs_overlay(pred.copy(), None, [_obs(0, 0, 1, 2, 2)], tau=10)
        # Cells far from viewport should be essentially unchanged
        np.testing.assert_allclose(result[8:, 8:], pred[8:, 8:], atol=1e-6)

    def test_floor_applied(self):
        # Result should have no zeros (FLOOR=0.005)
        pred = _uniform_pred()
        result = apply_obs_overlay(pred.copy(), None, [_obs(0, 0, 1, 5, 5)], tau=100)
        assert (result > 0).all()

    def test_higher_tau_less_update(self):
        # Higher tau → less influence from observations → prob closer to prior
        pred = _uniform_pred()
        obs = [_obs(3, 3, 1, 1, 1)]

        result_low_tau = apply_obs_overlay(pred.copy(), None, obs, tau=1)
        result_high_tau = apply_obs_overlay(pred.copy(), None, obs, tau=100)

        # At the observed cell, low tau should give higher class-1 prob
        assert result_low_tau[3, 3, 1] > result_high_tau[3, 3, 1]

    def test_multiple_observations(self):
        pred = _uniform_pred()
        obs = [_obs(0, 0, 1, 2, 2), _obs(5, 5, 2, 2, 2)]
        result = apply_obs_overlay(pred.copy(), None, obs, tau=5)
        np.testing.assert_allclose(result.sum(axis=-1), np.ones((10, 10)), atol=1e-5)


# ---------------------------------------------------------------------------
# eval_honest.iou
# ---------------------------------------------------------------------------

class TestIou:
    def test_identical_boxes(self):
        box = [10, 10, 50, 50]
        assert abs(iou(box, box) - 1.0) < 1e-9

    def test_non_overlapping(self):
        a = [0, 0, 10, 10]
        b = [20, 20, 30, 30]
        assert iou(a, b) == 0.0

    def test_partial_overlap(self):
        a = [0, 0, 10, 10]
        b = [5, 5, 15, 15]
        # Intersection: 5×5=25, union: 100+100-25=175
        expected = 25 / 175
        assert abs(iou(a, b) - expected) < 1e-9

    def test_contained_box(self):
        outer = [0, 0, 10, 10]
        inner = [2, 2, 8, 8]
        # Inner area = 36, outer area = 100, intersection = 36, union = 100
        expected = 36 / 100
        assert abs(iou(outer, inner) - expected) < 1e-9

    def test_zero_area_box(self):
        a = [5, 5, 5, 5]  # zero area
        b = [0, 0, 10, 10]
        assert iou(a, b) == 0.0

    def test_range_0_to_1(self):
        a = [0, 0, 100, 100]
        b = [50, 50, 150, 150]
        result = iou(a, b)
        assert 0 <= result <= 1

    def test_symmetric(self):
        a = [0, 0, 20, 30]
        b = [10, 15, 40, 50]
        assert abs(iou(a, b) - iou(b, a)) < 1e-9


# ---------------------------------------------------------------------------
# eval_honest.voc_ap
# ---------------------------------------------------------------------------

class TestVocAp:
    def test_perfect_ap(self):
        # Recall goes from 0 to 1 monotonically with precision=1 throughout
        recalls = [0.1, 0.2, 0.5, 0.8, 1.0]
        precisions = [1.0, 1.0, 1.0, 1.0, 1.0]
        result = voc_ap(recalls, precisions)
        assert abs(result - 1.0) < 1e-9

    def test_zero_ap(self):
        recalls = [0.5, 1.0]
        precisions = [0.0, 0.0]
        result = voc_ap(recalls, precisions)
        assert abs(result) < 1e-9

    def test_range_0_to_1(self):
        recalls = [0.1, 0.3, 0.6, 0.9]
        precisions = [0.8, 0.6, 0.4, 0.2]
        result = voc_ap(recalls, precisions)
        assert 0 <= result <= 1

    def test_empty_lists(self):
        result = voc_ap([], [])
        assert result == 0.0

    def test_single_point(self):
        result = voc_ap([0.5], [0.8])
        assert 0 <= result <= 1

    def test_non_monotonic_recall_handled(self):
        # voc_ap uses monotonic envelope so non-monotonic precisions are ok
        recalls = [0.2, 0.5, 0.4, 0.8]
        precisions = [0.9, 0.7, 0.85, 0.5]
        result = voc_ap(recalls, precisions)
        assert 0 <= result <= 1

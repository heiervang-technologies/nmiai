"""Pure-function tests for grid mapping and COCO→YOLO conversion helpers.

Covers:
  tasks/astar-island/gpu_simulator.py    — map_to_internal, map_to_output
  tasks/object-detection/build_merged_dataset.py      — coco_to_yolo_bbox
  tasks/object-detection/convert_synthetic_to_yolo.py — coco_to_yolo_bbox

All pure functions — no file system, GPU, or network access.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
_OD_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "object-detection")
sys.path.insert(0, _ASTAR_DIR)

# Load coco_to_yolo_bbox from both OD modules by explicit path
def _load_func(module_name: str, path: str, func_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, func_name)

_OD_MERGED_PATH = str(Path(_OD_DIR) / "build_merged_dataset.py")
_OD_SYNTHETIC_PATH = str(Path(_OD_DIR) / "convert_synthetic_to_yolo.py")

coco_to_yolo_merged = _load_func("build_merged_dataset", _OD_MERGED_PATH, "coco_to_yolo_bbox")
coco_to_yolo_synthetic = _load_func("convert_synthetic_to_yolo", _OD_SYNTHETIC_PATH, "coco_to_yolo_bbox")

from gpu_simulator import map_to_internal, map_to_output, INTERNAL_MAP, INV_INTERNAL_MAP


# ---------------------------------------------------------------------------
# gpu_simulator.map_to_internal
# ---------------------------------------------------------------------------

class TestMapToInternal:
    def test_settlement_maps_to_1(self):
        grid = np.array([[1, 0]], dtype=np.int32)
        result = map_to_internal(grid)
        assert result[0, 0] == 1

    def test_ocean_maps_to_6(self):
        grid = np.array([[10]], dtype=np.int32)
        result = map_to_internal(grid)
        assert result[0, 0] == 6

    def test_plains_maps_to_7(self):
        grid = np.array([[11]], dtype=np.int32)
        result = map_to_internal(grid)
        assert result[0, 0] == 7

    def test_unknown_code_maps_to_0(self):
        grid = np.array([[99]], dtype=np.int32)
        result = map_to_internal(grid)
        assert result[0, 0] == 0

    def test_all_standard_codes(self):
        # Verify every code in INTERNAL_MAP maps correctly
        for code, expected in INTERNAL_MAP.items():
            grid = np.array([[code]], dtype=np.int32)
            result = map_to_internal(grid)
            assert result[0, 0] == expected, f"code {code} → expected {expected}, got {result[0,0]}"

    def test_preserves_shape(self):
        grid = np.arange(12, dtype=np.int32).reshape(3, 4)
        result = map_to_internal(grid)
        assert result.shape == (3, 4)


# ---------------------------------------------------------------------------
# gpu_simulator.map_to_output
# ---------------------------------------------------------------------------

class TestMapToOutput:
    def _tensor(self, *values) -> torch.Tensor:
        return torch.tensor([[values]], dtype=torch.long)

    def test_empty_0_stays_0(self):
        t = self._tensor(0)
        result = map_to_output(t)
        assert result[0, 0, 0].item() == 0

    def test_settlement_1_stays_1(self):
        t = self._tensor(1)
        assert map_to_output(t)[0, 0, 0].item() == 1

    def test_port_2_stays_2(self):
        t = self._tensor(2)
        assert map_to_output(t)[0, 0, 0].item() == 2

    def test_ruin_3_stays_3(self):
        t = self._tensor(3)
        assert map_to_output(t)[0, 0, 0].item() == 3

    def test_forest_4_stays_4(self):
        t = self._tensor(4)
        assert map_to_output(t)[0, 0, 0].item() == 4

    def test_mountain_5_stays_5(self):
        t = self._tensor(5)
        assert map_to_output(t)[0, 0, 0].item() == 5

    def test_ocean_internal_6_maps_to_0(self):
        t = self._tensor(6)
        assert map_to_output(t)[0, 0, 0].item() == 0

    def test_plains_internal_7_maps_to_0(self):
        t = self._tensor(7)
        assert map_to_output(t)[0, 0, 0].item() == 0

    def test_preserves_shape(self):
        t = torch.zeros(2, 5, 5, dtype=torch.long)
        result = map_to_output(t)
        assert result.shape == (2, 5, 5)


# ---------------------------------------------------------------------------
# coco_to_yolo_bbox — shared behavior for both modules
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [coco_to_yolo_merged, coco_to_yolo_synthetic],
                         ids=["merged", "synthetic"])
class TestCocoToYoloBbox:
    def test_centered_box(self, fn):
        # Box at (0,0) with size (img_w, img_h) → center at (0.5, 0.5)
        cx, cy, nw, nh = fn([0, 0, 100, 80], 100, 80)
        assert abs(cx - 0.5) < 1e-9
        assert abs(cy - 0.5) < 1e-9
        assert abs(nw - 1.0) < 1e-9
        assert abs(nh - 1.0) < 1e-9

    def test_small_box(self, fn):
        cx, cy, nw, nh = fn([10, 10, 20, 20], 100, 100)
        assert abs(cx - 0.2) < 1e-9
        assert abs(cy - 0.2) < 1e-9
        assert abs(nw - 0.2) < 1e-9
        assert abs(nh - 0.2) < 1e-9

    def test_output_clamped_to_0_1(self, fn):
        # Box that extends beyond image boundaries
        cx, cy, nw, nh = fn([-10, -10, 200, 200], 100, 100)
        assert 0 <= cx <= 1
        assert 0 <= cy <= 1
        assert 0 <= nw <= 1
        assert 0 <= nh <= 1

    def test_returns_four_values(self, fn):
        result = fn([0, 0, 50, 50], 100, 100)
        assert len(result) == 4

    def test_top_left_corner_box(self, fn):
        # 10×10 box at top-left corner of 100×100 image
        cx, cy, nw, nh = fn([0, 0, 10, 10], 100, 100)
        assert abs(cx - 0.05) < 1e-9
        assert abs(cy - 0.05) < 1e-9

    def test_normalized_width_height(self, fn):
        cx, cy, nw, nh = fn([20, 30, 40, 50], 200, 150)
        # cx = (20 + 20) / 200 = 0.2
        # cy = (30 + 25) / 150 = 55/150
        assert abs(cx - 0.2) < 1e-9
        assert abs(nw - 0.2) < 1e-9

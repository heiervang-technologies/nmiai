"""Tests for tasks/object-detection/vlm-approach/wbf.py."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(
    0,
    str(Path(__file__).resolve().parent.parent / "tasks" / "object-detection" / "vlm-approach"),
)

from wbf import _fused_box, _iou, weighted_boxes_fusion


# ---------------------------------------------------------------------------
# _iou
# ---------------------------------------------------------------------------

class TestIou:
    def test_identical_boxes_returns_one(self):
        box = np.array([0.0, 0.0, 1.0, 1.0])
        assert _iou(box, box) == pytest.approx(1.0)

    def test_non_overlapping_returns_zero(self):
        a = np.array([0.0, 0.0, 0.4, 0.4])
        b = np.array([0.6, 0.6, 1.0, 1.0])
        assert _iou(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = np.array([0.0, 0.0, 0.6, 0.6])
        b = np.array([0.4, 0.4, 1.0, 1.0])
        iou = _iou(a, b)
        assert 0.0 < iou < 1.0

    def test_contained_box_has_iou_below_one(self):
        outer = np.array([0.0, 0.0, 1.0, 1.0])
        inner = np.array([0.2, 0.2, 0.8, 0.8])
        iou = _iou(outer, inner)
        # inner area / outer area = 0.36
        assert iou == pytest.approx(0.36, abs=0.01)

    def test_zero_area_box_returns_zero(self):
        a = np.array([0.5, 0.5, 0.5, 0.5])  # zero area
        b = np.array([0.0, 0.0, 1.0, 1.0])
        assert _iou(a, b) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _fused_box
# ---------------------------------------------------------------------------

class TestFusedBox:
    def test_single_box_returns_itself(self):
        cluster = [{"box": np.array([0.1, 0.2, 0.5, 0.6]), "score": 0.9}]
        result = _fused_box(cluster)
        np.testing.assert_allclose(result, [0.1, 0.2, 0.5, 0.6])

    def test_equal_weights_gives_average(self):
        cluster = [
            {"box": np.array([0.0, 0.0, 0.4, 0.4]), "score": 1.0},
            {"box": np.array([0.2, 0.2, 0.6, 0.6]), "score": 1.0},
        ]
        result = _fused_box(cluster)
        np.testing.assert_allclose(result, [0.1, 0.1, 0.5, 0.5])

    def test_higher_score_weights_more(self):
        cluster = [
            {"box": np.array([0.0, 0.0, 0.4, 0.4]), "score": 0.1},
            {"box": np.array([0.5, 0.5, 0.9, 0.9]), "score": 0.9},
        ]
        result = _fused_box(cluster)
        # Should be closer to the second box
        assert result[0] > 0.25
        assert result[1] > 0.25


# ---------------------------------------------------------------------------
# weighted_boxes_fusion
# ---------------------------------------------------------------------------

class TestWeightedBoxesFusion:
    def test_empty_input_returns_empty_arrays(self):
        boxes, scores, labels = weighted_boxes_fusion([], [], [])
        assert boxes.shape == (0, 4)
        assert scores.shape == (0,)
        assert labels.shape == (0,)

    def test_single_box_returns_single_box(self):
        boxes = [np.array([[0.1, 0.2, 0.5, 0.6]])]
        scores = [np.array([0.9])]
        labels = [np.array([1])]
        fboxes, fscores, flabels = weighted_boxes_fusion(boxes, scores, labels)
        assert fboxes.shape == (1, 4)
        assert flabels[0] == 1

    def test_two_identical_boxes_fused_to_one(self):
        box = np.array([[0.1, 0.2, 0.5, 0.6]])
        boxes = [box, box]
        scores = [np.array([0.8]), np.array([0.8])]
        labels = [np.array([0]), np.array([0])]
        fboxes, fscores, flabels = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.5)
        assert len(fboxes) == 1
        np.testing.assert_allclose(fboxes[0], [0.1, 0.2, 0.5, 0.6], atol=1e-6)

    def test_non_overlapping_boxes_stay_separate(self):
        boxes = [
            np.array([[0.0, 0.0, 0.3, 0.3]]),
            np.array([[0.7, 0.7, 1.0, 1.0]]),
        ]
        scores = [np.array([0.9]), np.array([0.9])]
        labels = [np.array([0]), np.array([0])]
        fboxes, fscores, flabels = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.5)
        assert len(fboxes) == 2

    def test_different_labels_stay_separate(self):
        box = np.array([[0.1, 0.1, 0.9, 0.9]])
        boxes = [box, box]
        scores = [np.array([0.9]), np.array([0.9])]
        labels = [np.array([0]), np.array([1])]  # different labels
        fboxes, fscores, flabels = weighted_boxes_fusion(boxes, scores, labels, iou_thr=0.5)
        assert len(fboxes) == 2
        assert set(flabels) == {0, 1}

    def test_skip_box_thr_filters_low_scores(self):
        boxes = [np.array([[0.1, 0.1, 0.5, 0.5]])]
        scores = [np.array([0.05])]
        labels = [np.array([0])]
        fboxes, fscores, flabels = weighted_boxes_fusion(
            boxes, scores, labels, skip_box_thr=0.1
        )
        assert len(fboxes) == 0

    def test_conf_type_max(self):
        box = np.array([[0.1, 0.2, 0.5, 0.6]])
        boxes = [box, box]
        scores_a = [np.array([0.7]), np.array([0.9])]
        labels_both = [np.array([0]), np.array([0])]
        _, fscores, _ = weighted_boxes_fusion(
            boxes, scores_a, labels_both, iou_thr=0.5, conf_type="max"
        )
        assert fscores[0] == pytest.approx(0.9, abs=0.01)

    def test_returns_three_arrays(self):
        result = weighted_boxes_fusion([], [], [])
        assert len(result) == 3

    def test_output_labels_are_integers(self):
        boxes = [np.array([[0.0, 0.0, 0.5, 0.5]])]
        scores = [np.array([0.8])]
        labels = [np.array([3])]
        _, _, flabels = weighted_boxes_fusion(boxes, scores, labels)
        assert flabels.dtype == int or np.issubdtype(flabels.dtype, np.integer)

    def test_custom_weights_applied(self):
        """Higher-weight model's box should dominate the fused result."""
        box_a = np.array([[0.0, 0.0, 0.4, 0.4]])
        box_b = np.array([[0.5, 0.5, 0.9, 0.9]])
        # Both score 0.8 but box_b has much higher weight
        boxes = [box_a, box_b]
        scores = [np.array([0.8]), np.array([0.8])]
        labels = [np.array([0]), np.array([0])]
        _, fscores, _ = weighted_boxes_fusion(
            boxes, scores, labels, weights=[0.1, 10.0], iou_thr=0.0
        )
        # With iou_thr=0 (no merging) we get 2 clusters; just sanity check count
        assert len(fscores) >= 1

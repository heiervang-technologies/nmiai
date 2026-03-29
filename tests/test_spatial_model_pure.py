"""Tests for tasks/astar-island/spatial_model.py — pure helper functions.

Covers: predict_logodds (multinomial logistic regression inference).
Pure numpy matrix operation — no file I/O or GPU required.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR)

from spatial_model import predict_logodds


# ---------------------------------------------------------------------------
# predict_logodds
# ---------------------------------------------------------------------------

class TestPredictLogodds:
    def test_returns_ndarray(self):
        X = np.ones((3, 4))
        W = np.zeros((4, 5))
        result = predict_logodds(X, W)
        assert isinstance(result, np.ndarray)

    def test_output_shape_n_by_6(self):
        n, n_features = 5, 4
        X = np.random.default_rng(0).normal(size=(n, n_features))
        W = np.random.default_rng(1).normal(size=(n_features, 5))
        result = predict_logodds(X, W)
        assert result.shape == (n, 6)

    def test_rows_sum_to_one(self):
        X = np.random.default_rng(2).normal(size=(4, 3))
        W = np.random.default_rng(3).normal(size=(3, 5))
        result = predict_logodds(X, W)
        assert np.allclose(result.sum(axis=1), 1.0)

    def test_all_probabilities_positive(self):
        X = np.random.default_rng(4).normal(size=(3, 4))
        W = np.random.default_rng(5).normal(size=(4, 5))
        result = predict_logodds(X, W)
        assert np.all(result > 0.0)

    def test_zero_weights_uniform_over_six_classes(self):
        X = np.ones((2, 3))
        W = np.zeros((3, 5))
        result = predict_logodds(X, W)
        # With zero weights, logits are all zero → uniform distribution
        assert np.allclose(result, 1 / 6, atol=1e-6)

    def test_single_row(self):
        X = np.array([[1.0, 0.0, -1.0]])
        W = np.eye(3, 5)  # 3 features, 5 non-reference classes
        result = predict_logodds(X, W)
        assert result.shape == (1, 6)
        assert result.sum() == pytest.approx(1.0)

    def test_large_positive_weight_biases_towards_class(self):
        X = np.ones((1, 1))
        # Very large weight for class 1 (second column of W, which is class index 1 in logits)
        W = np.zeros((1, 5))
        W[0, 0] = 100.0  # class index 1 (after prepending reference class 0)
        result = predict_logodds(X, W)
        # Class 1 (logits column 1) should dominate
        assert result[0, 1] > 0.99

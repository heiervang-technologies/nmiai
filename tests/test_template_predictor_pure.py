"""Pure-function tests for astar-island/template_predictor.py.

Covers: cell_to_type, cell_code_to_class, quantized_distance, support_blend,
        blend_probs, finalize_tables, lookup_tables, template_strength_from_weights

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock
from collections import defaultdict

import numpy as np
import pytest

# neighborhood_predictor and requests needed for imports
sys.modules.setdefault("neighborhood_predictor", MagicMock())
sys.modules.setdefault("requests", MagicMock())

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

import template_predictor as tp


N_CLASSES = 6


# ---------------------------------------------------------------------------
# cell_to_type
# ---------------------------------------------------------------------------

class TestCellToType:
    @pytest.mark.parametrize("code,expected", [
        (1, "settlement"),
        (2, "port"),
        (4, "forest"),
        (11, "plains"),
        (0, "empty"),
    ])
    def test_known_codes(self, code, expected):
        assert tp.cell_to_type(code) == expected

    def test_ocean_returns_none(self):
        assert tp.cell_to_type(10) is None

    def test_mountain_returns_none(self):
        assert tp.cell_to_type(5) is None

    def test_unknown_returns_none(self):
        assert tp.cell_to_type(99) is None


# ---------------------------------------------------------------------------
# cell_code_to_class
# ---------------------------------------------------------------------------

class TestCellCodeToClass:
    @pytest.mark.parametrize("code,expected", [
        (0, 0), (10, 0), (11, 0),
        (1, 1), (2, 2), (3, 3), (4, 4), (5, 5),
    ])
    def test_known_codes(self, code, expected):
        assert tp.cell_code_to_class(code) == expected

    def test_unknown_returns_zero(self):
        assert tp.cell_code_to_class(99) == 0


# ---------------------------------------------------------------------------
# quantized_distance
# ---------------------------------------------------------------------------

class TestQuantizedDistance:
    def test_zero_distance(self):
        assert tp.quantized_distance(0.0) == 0

    def test_rounds_down(self):
        assert tp.quantized_distance(3.9) == 3

    def test_clamps_at_12(self):
        assert tp.quantized_distance(100.0) == 12

    def test_negative_clamps_at_zero(self):
        assert tp.quantized_distance(-5.0) == 0

    def test_exact_integer(self):
        assert tp.quantized_distance(7.0) == 7

    def test_returns_int(self):
        assert isinstance(tp.quantized_distance(3.5), int)


# ---------------------------------------------------------------------------
# support_blend
# ---------------------------------------------------------------------------

class TestSupportBlend:
    def test_zero_count_returns_zero(self):
        assert tp.support_blend(0.0, 10.0) == pytest.approx(0.0)

    def test_large_count_approaches_one(self):
        result = tp.support_blend(1e9, 1.0)
        assert result == pytest.approx(1.0, rel=1e-5)

    def test_equal_count_and_shrink_is_half(self):
        assert tp.support_blend(10.0, 10.0) == pytest.approx(0.5)

    def test_returns_float(self):
        assert isinstance(tp.support_blend(5.0, 5.0), float)

    def test_nonneg(self):
        assert tp.support_blend(3.0, 7.0) >= 0.0


# ---------------------------------------------------------------------------
# blend_probs
# ---------------------------------------------------------------------------

class TestBlendProbs:
    def test_alpha_one_returns_primary(self):
        primary = np.array([0.5, 0.3, 0.1, 0.05, 0.03, 0.02])
        fallback = np.ones(N_CLASSES) / N_CLASSES
        result = tp.blend_probs(primary, fallback, 1.0)
        np.testing.assert_allclose(result, primary / primary.sum(), atol=1e-6)

    def test_alpha_zero_returns_fallback(self):
        primary = np.array([0.9, 0.05, 0.01, 0.01, 0.01, 0.02])
        fallback = np.ones(N_CLASSES) / N_CLASSES
        result = tp.blend_probs(primary, fallback, 0.0)
        np.testing.assert_allclose(result, fallback / fallback.sum(), atol=1e-6)

    def test_sums_to_one(self):
        primary = np.array([0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
        fallback = np.ones(N_CLASSES) / N_CLASSES
        result = tp.blend_probs(primary, fallback, 0.6)
        assert abs(result.sum() - 1.0) < 1e-9

    def test_all_values_positive(self):
        primary = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        fallback = np.ones(N_CLASSES) / N_CLASSES
        result = tp.blend_probs(primary, fallback, 0.5)
        assert (result > 0).all()


# ---------------------------------------------------------------------------
# finalize_tables
# ---------------------------------------------------------------------------

class TestFinalizeTables:
    def test_returns_list_of_4_dicts(self):
        sums = [defaultdict(lambda: np.zeros(N_CLASSES)) for _ in range(4)]
        counts = [defaultdict(float) for _ in range(4)]
        tables = tp.finalize_tables(sums, counts)
        assert isinstance(tables, list)
        assert len(tables) == 4

    def test_averages_correctly(self):
        sums = [defaultdict(lambda: np.zeros(N_CLASSES)) for _ in range(4)]
        counts = [defaultdict(float) for _ in range(4)]
        vec = np.array([2.0, 1.0, 0.5, 0.25, 0.125, 0.125])
        sums[0][("S", 2)] = vec.copy()
        counts[0][("S", 2)] = 2.0
        tables = tp.finalize_tables(sums, counts)
        np.testing.assert_allclose(tables[0][("S", 2)], vec / 2.0)

    def test_zero_count_does_not_crash(self):
        sums = [defaultdict(lambda: np.zeros(N_CLASSES)) for _ in range(4)]
        counts = [defaultdict(float) for _ in range(4)]
        sums[0][("L", 5)] = np.ones(N_CLASSES)
        counts[0][("L", 5)] = 0.0
        tables = tp.finalize_tables(sums, counts)
        # count=0 → divides by max(0,1)=1 → same as sum
        np.testing.assert_allclose(tables[0][("L", 5)], np.ones(N_CLASSES))


# ---------------------------------------------------------------------------
# lookup_tables
# ---------------------------------------------------------------------------

class TestLookupTables:
    def _make_tables_counts(self):
        p = np.ones(N_CLASSES) / N_CLASSES
        tables = [{("S", 2): p}, {("S", 1): p}, {("S",): p}, {("S",): p}]
        counts = [{("S", 2): 5.0}, {("S", 1): 8.0}, {("S",): 10.0}, {("S",): 2.0}]
        return tables, counts

    def test_returns_fine_when_sufficient(self):
        tables, counts = self._make_tables_counts()
        keys = (("S", 2), ("S", 1), ("S",), ("S",))
        _, c, level = tp.lookup_tables(tables, counts, keys)
        assert level == 0

    def test_falls_back_when_count_low(self):
        tables, counts = self._make_tables_counts()
        counts[0][("S", 2)] = 1.0  # below threshold=3
        keys = (("S", 2), ("S", 1), ("S",), ("S",))
        _, c, level = tp.lookup_tables(tables, counts, keys)
        assert level == 1

    def test_no_match_returns_none(self):
        tables = [{}, {}, {}, {}]
        counts = [{}, {}, {}, {}]
        keys = (("X",), ("X",), ("X",), ("X",))
        prob, c, level = tp.lookup_tables(tables, counts, keys)
        assert prob is None


# ---------------------------------------------------------------------------
# template_strength_from_weights
# ---------------------------------------------------------------------------

class TestTemplateStrengthFromWeights:
    def test_no_observations_returns_zero(self):
        weights = np.ones(5) / 5
        assert tp.template_strength_from_weights(weights, has_observations=False) == pytest.approx(0.0)

    def test_returns_float(self):
        weights = np.ones(5) / 5
        result = tp.template_strength_from_weights(weights, has_observations=True)
        assert isinstance(result, float)

    def test_nonneg(self):
        weights = np.ones(5) / 5
        assert tp.template_strength_from_weights(weights, has_observations=True) >= 0.0

    def test_sharp_posterior_gives_higher_strength(self):
        # Very concentrated posterior
        sharp = np.array([0.98, 0.005, 0.005, 0.005, 0.005])
        # Diffuse posterior (uniform)
        diffuse = np.ones(5) / 5
        s_sharp = tp.template_strength_from_weights(sharp, has_observations=True)
        s_diffuse = tp.template_strength_from_weights(diffuse, has_observations=True)
        assert s_sharp > s_diffuse

    def test_bounded_by_max(self):
        weights = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        result = tp.template_strength_from_weights(weights, has_observations=True)
        assert result <= tp.TEMPLATE_SHIFT_MAX + 1e-9

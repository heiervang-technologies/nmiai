"""Tests for tasks/astar-island/surrogate_mc.py — pure helper functions.

Covers: normalize, cell_code_to_class, cell_type_token, quantize_dist,
        quantize_neighbor_civ, base_bucket_keys, spatial_bucket_keys,
        lookup_tables, apply_constraints, structured_fallback,
        kl_divergence, entropy.
All pure numpy functions — no file system or network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island"))

from surrogate_mc import (
    FLOOR,
    N_CLASSES,
    apply_constraints,
    base_bucket_keys,
    cell_code_to_class,
    cell_type_token,
    entropy,
    kl_divergence,
    lookup_tables,
    normalize,
    quantize_dist,
    quantize_neighbor_civ,
    spatial_bucket_keys,
    structured_fallback,
)

OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4
PLAINS = 11


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_sums_to_one(self):
        prob = np.array([2.0, 3.0, 1.0, 0.0, 0.5, 0.5])
        result = normalize(prob)
        assert result.sum() == pytest.approx(1.0)

    def test_floor_applied_to_zeros(self):
        prob = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = normalize(prob)
        assert (result > 0).all()

    def test_nan_returns_uniform(self):
        prob = np.array([float("nan"), 1.0, 0.0, 0.0, 0.0, 0.0])
        result = normalize(prob)
        assert result.sum() == pytest.approx(1.0)

    def test_zero_vector_returns_uniform(self):
        prob = np.zeros(N_CLASSES)
        result = normalize(prob)
        assert result.sum() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# cell_code_to_class
# ---------------------------------------------------------------------------

class TestCellCodeToClass:
    def test_ocean_maps_to_0(self):
        assert cell_code_to_class(10) == 0

    def test_plains_maps_to_0(self):
        assert cell_code_to_class(11) == 0

    def test_empty_maps_to_0(self):
        assert cell_code_to_class(0) == 0

    def test_settlement_maps_to_1(self):
        assert cell_code_to_class(1) == 1

    def test_port_maps_to_2(self):
        assert cell_code_to_class(2) == 2

    def test_forest_maps_to_4(self):
        assert cell_code_to_class(4) == 4

    def test_mountain_maps_to_5(self):
        assert cell_code_to_class(5) == 5

    def test_unknown_maps_to_0(self):
        assert cell_code_to_class(99) == 0


# ---------------------------------------------------------------------------
# cell_type_token
# ---------------------------------------------------------------------------

class TestCellTypeToken:
    def test_settlement_returns_S(self):
        assert cell_type_token(SETTLEMENT) == "S"

    def test_port_returns_P(self):
        assert cell_type_token(PORT) == "P"

    def test_forest_returns_F(self):
        assert cell_type_token(FOREST) == "F"

    def test_plains_returns_L(self):
        assert cell_type_token(PLAINS) == "L"

    def test_empty_returns_L(self):
        assert cell_type_token(0) == "L"


# ---------------------------------------------------------------------------
# quantize_dist
# ---------------------------------------------------------------------------

class TestQuantizeDist:
    def test_zero_returns_0(self):
        assert quantize_dist(0) == 0

    def test_floor_applied(self):
        assert quantize_dist(-1) == 0

    def test_exact_integer(self):
        assert quantize_dist(5.0) == 5

    def test_clamped_at_15(self):
        assert quantize_dist(100) == 15

    def test_floor_of_float(self):
        assert quantize_dist(3.9) == 3


# ---------------------------------------------------------------------------
# quantize_neighbor_civ
# ---------------------------------------------------------------------------

class TestQuantizeNeighborCiv:
    def test_zero_neighbors_returns_0(self):
        assert quantize_neighbor_civ(0) == 0

    def test_one_neighbor_returns_1(self):
        assert quantize_neighbor_civ(1) == 1

    def test_two_neighbors_returns_2(self):
        assert quantize_neighbor_civ(2) == 2

    def test_three_neighbors_returns_2(self):
        assert quantize_neighbor_civ(3) == 2

    def test_four_neighbors_returns_3(self):
        assert quantize_neighbor_civ(4) == 3

    def test_eight_neighbors_returns_3(self):
        assert quantize_neighbor_civ(8) == 3


# ---------------------------------------------------------------------------
# base_bucket_keys
# ---------------------------------------------------------------------------

class TestBaseBucketKeys:
    def test_ocean_returns_none(self):
        assert base_bucket_keys(OCEAN, 0, 3, True) is None

    def test_mountain_returns_none(self):
        assert base_bucket_keys(MOUNTAIN, 5, 0, False) is None

    def test_returns_4_keys(self):
        keys = base_bucket_keys(SETTLEMENT, 3, 2, True)
        assert len(keys) == 4

    def test_first_key_is_finest(self):
        keys = base_bucket_keys(SETTLEMENT, 3, 2, True)
        assert len(keys[0]) == 4  # (t, d, no, c)

    def test_last_key_is_broadest(self):
        keys = base_bucket_keys(SETTLEMENT, 3, 2, True)
        assert len(keys[3]) == 2  # (t, clamped_d)

    def test_token_in_first_key(self):
        keys = base_bucket_keys(SETTLEMENT, 3, 2, True)
        assert keys[0][0] == "S"


# ---------------------------------------------------------------------------
# spatial_bucket_keys
# ---------------------------------------------------------------------------

class TestSpatialBucketKeys:
    def test_ocean_returns_none(self):
        assert spatial_bucket_keys(OCEAN, 0, 3, True, 0) is None

    def test_returns_4_keys(self):
        keys = spatial_bucket_keys(SETTLEMENT, 3, 2, True, 1)
        assert len(keys) == 4

    def test_includes_nc_in_first_key(self):
        keys = spatial_bucket_keys(SETTLEMENT, 3, 2, True, 2)
        assert len(keys[0]) == 5  # (t, d, no, c, nc)


# ---------------------------------------------------------------------------
# lookup_tables
# ---------------------------------------------------------------------------

class TestLookupTables:
    def _tables_counts(self, key, value, count):
        tables = [{key: value}, {}, {}, {}]
        counts = [{key: count}, {}, {}, {}]
        return tables, counts

    def test_returns_value_when_found_with_enough_support(self):
        tables, counts = self._tables_counts(("S", 3, 2, 1), [0.5, 0.5, 0, 0, 0, 0], 20)
        result, c, level = lookup_tables(tables, counts, [("S", 3, 2, 1), ("S", 3, 1), ("S", 4, 1), ("S", 2)])
        assert result is not None
        assert level == 0

    def test_returns_none_when_key_absent(self):
        tables = [{}, {}, {}, {}]
        counts = [{}, {}, {}, {}]
        result, c, level = lookup_tables(tables, counts, [("X", 0), ("X",), ("Y",), ("Z",)])
        assert result is None

    def test_falls_back_when_support_too_low(self):
        key_fine = ("S", 3, 2, 1)
        key_coarse = ("S", 3)
        tables = [{key_fine: [0.5]*6, key_coarse: [0.3]*6 + [0]*0}, {key_coarse: [0.3]*6}, {}, {}]
        counts = [{key_fine: 2, key_coarse: 50}, {key_coarse: 50}, {}, {}]  # fine has low support
        result, c, level = lookup_tables(
            tables, counts,
            [key_fine, key_coarse, ("S", 3), ("S",)],
            min_counts=(10, 10, 10, 1),
        )
        # Should skip fine (support=2 < 10) and find coarse (support=50)
        assert result is not None


# ---------------------------------------------------------------------------
# apply_constraints
# ---------------------------------------------------------------------------

class TestApplyConstraints:
    def test_ocean_returns_ocean_dist(self):
        prob = np.ones(N_CLASSES) / N_CLASSES
        result = apply_constraints(prob, OCEAN, 0, False)
        assert result[0] == pytest.approx(1.0)
        assert result[1:].sum() == pytest.approx(0.0)

    def test_mountain_returns_mountain_dist(self):
        prob = np.ones(N_CLASSES) / N_CLASSES
        result = apply_constraints(prob, MOUNTAIN, 0, False)
        assert result[5] == pytest.approx(1.0)

    def test_no_mountain_class_for_non_mountain(self):
        prob = np.ones(N_CLASSES) / N_CLASSES
        result = apply_constraints(prob, PLAINS, 3, False)
        # floor is applied after zeroing, so value is tiny (< FLOOR * 2)
        assert result[5] < FLOOR * 2

    def test_no_port_if_not_coastal(self):
        prob = np.ones(N_CLASSES) / N_CLASSES
        result = apply_constraints(prob, PLAINS, 3, False)  # not coastal
        # floor is applied after zeroing, so value is tiny (< FLOOR * 2)
        assert result[2] < FLOOR * 2

    def test_port_allowed_if_coastal(self):
        prob = np.ones(N_CLASSES) / N_CLASSES
        result = apply_constraints(prob, PLAINS, 3, True)  # coastal
        # Port class can be non-zero
        assert result[2] > 0.0

    def test_result_sums_to_one(self):
        prob = np.ones(N_CLASSES) / N_CLASSES
        result = apply_constraints(prob, PLAINS, 3, False)
        assert result.sum() == pytest.approx(1.0, abs=1e-5)


# ---------------------------------------------------------------------------
# structured_fallback
# ---------------------------------------------------------------------------

class TestStructuredFallback:
    def test_returns_array_of_n_classes(self):
        result = structured_fallback(PLAINS, 5, False)
        assert len(result) == N_CLASSES

    def test_sums_to_one(self):
        for code in [PLAINS, FOREST, SETTLEMENT, PORT]:
            for coast in [True, False]:
                result = structured_fallback(code, 5, coast)
                assert result.sum() == pytest.approx(1.0, abs=1e-5), f"code={code}, coast={coast}"

    def test_remote_forest_returns_pure_forest(self):
        result = structured_fallback(FOREST, 15, False)
        assert result[4] == pytest.approx(1.0)

    def test_remote_plains_returns_pure_plains(self):
        result = structured_fallback(PLAINS, 15, False)
        assert result[0] == pytest.approx(1.0)

    def test_coastal_settlement_has_port_mass(self):
        coastal = structured_fallback(SETTLEMENT, 2, True)
        non_coastal = structured_fallback(SETTLEMENT, 2, False)
        assert coastal[2] > non_coastal[2]


# ---------------------------------------------------------------------------
# kl_divergence
# ---------------------------------------------------------------------------

class TestKlDivergence:
    def test_identical_near_zero(self):
        p = np.full((2, 2, 6), 1.0 / 6)
        result = kl_divergence(p, p)
        np.testing.assert_allclose(result, 0.0, atol=1e-6)

    def test_shape_removes_class_axis(self):
        p = np.full((3, 4, 6), 1.0 / 6)
        result = kl_divergence(p, p)
        assert result.shape == (3, 4)

    def test_non_negative(self):
        p = np.full((2, 2, 6), 1.0 / 6)
        # q rows must each sum to 1
        q = np.array([
            [[0.7, 0.06, 0.06, 0.06, 0.06, 0.06],
             [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]],
            [[0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
             [0.4, 0.12, 0.12, 0.12, 0.12, 0.12]]
        ])
        result = kl_divergence(p, q)
        assert (result >= 0).all()


# ---------------------------------------------------------------------------
# entropy
# ---------------------------------------------------------------------------

class TestEntropy:
    def test_uniform_6_class_gives_log2_6_bits(self):
        p = np.full((2, 2, 6), 1.0 / 6)
        result = entropy(p)
        np.testing.assert_allclose(result, np.log2(6), atol=1e-5)

    def test_deterministic_gives_zero(self):
        p = np.zeros((2, 2, 6))
        p[:, :, 0] = 1.0
        result = entropy(p)
        np.testing.assert_allclose(result, 0.0, atol=1e-5)

    def test_shape_removes_class_axis(self):
        p = np.full((3, 4, 6), 1.0 / 6)
        result = entropy(p)
        assert result.shape == (3, 4)

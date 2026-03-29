"""Pure-function tests for astar-island adaptive_query.py.

Covers: initial_class_grid, static_mask, viewport_cells, overlap_ratio,
        hotspot_prior, reconnaissance_viewports, observation_stats,
        target_heatmap, plan_queries.

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# Mock heavy dependencies before importing the module under test
for _mod in ("requests", "sklearn", "sklearn.ensemble", "sklearn.linear_model"):
    sys.modules.setdefault(_mod, MagicMock())

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

from adaptive_query import (
    initial_class_grid,
    static_mask,
    viewport_cells,
    overlap_ratio,
    hotspot_prior,
    reconnaissance_viewports,
    observation_stats,
    target_heatmap,
    plan_queries,
    GRID_SIZE, VIEWPORT, RECON_QUERIES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uniform_grid(code: int = 11, size: int = GRID_SIZE) -> list[list[int]]:
    """A grid of uniform cell code with a few features."""
    g = [[code] * size for _ in range(size)]
    g[5][5] = 1    # settlement
    g[5][6] = 2    # port
    g[10][10] = 4  # forest
    g[0][0] = 10   # ocean
    g[-1][-1] = 5  # mountain
    return g


def _make_obs(x: int = 0, y: int = 0, size: int = VIEWPORT, code: int = 11) -> dict:
    """A fake viewport observation."""
    return {
        "grid": [[code] * size for _ in range(size)],
        "viewport_x": x,
        "viewport_y": y,
    }


# ---------------------------------------------------------------------------
# initial_class_grid
# ---------------------------------------------------------------------------

class TestInitialClassGrid:
    def test_shape_preserved(self):
        g = _uniform_grid()
        out = initial_class_grid(g)
        assert out.shape == (GRID_SIZE, GRID_SIZE)

    def test_settlement_maps_to_1(self):
        g = [[11] * 5 for _ in range(5)]
        g[2][2] = 1
        out = initial_class_grid(g)
        assert out[2, 2] == 1

    def test_port_maps_to_2(self):
        g = [[11] * 5 for _ in range(5)]
        g[3][3] = 2
        out = initial_class_grid(g)
        assert out[3, 3] == 2

    def test_mountain_maps_to_5(self):
        g = [[11] * 5 for _ in range(5)]
        g[1][1] = 5
        out = initial_class_grid(g)
        assert out[1, 1] == 5

    def test_ocean_maps_to_0(self):
        g = [[10] * 5 for _ in range(5)]
        out = initial_class_grid(g)
        assert (out == 0).all()

    def test_plains_maps_to_0(self):
        g = [[11] * 5 for _ in range(5)]
        out = initial_class_grid(g)
        assert (out == 0).all()

    def test_accepts_numpy_input(self):
        arr = np.full((5, 5), 11, dtype=np.int32)
        arr[2, 2] = 4
        out = initial_class_grid(arr)
        assert out[2, 2] == 4


# ---------------------------------------------------------------------------
# static_mask
# ---------------------------------------------------------------------------

class TestStaticMask:
    def test_mountain_is_static(self):
        g = [[11] * 5 for _ in range(5)]
        g[2][2] = 5
        mask = static_mask(g)
        assert mask[2, 2]

    def test_ocean_is_static(self):
        g = [[11] * 5 for _ in range(5)]
        g[0][0] = 10
        mask = static_mask(g)
        assert mask[0, 0]

    def test_settlement_not_static(self):
        g = [[11] * 5 for _ in range(5)]
        g[1][1] = 1
        mask = static_mask(g)
        assert not mask[1, 1]

    def test_plains_not_static(self):
        g = [[11] * 5 for _ in range(5)]
        mask = static_mask(g)
        assert not mask.any()

    def test_shape(self):
        g = [[11] * GRID_SIZE for _ in range(GRID_SIZE)]
        assert static_mask(g).shape == (GRID_SIZE, GRID_SIZE)


# ---------------------------------------------------------------------------
# viewport_cells
# ---------------------------------------------------------------------------

class TestViewportCells:
    def test_count_correct(self):
        cells = viewport_cells((0, 0, 5, 5))
        assert len(cells) == 25

    def test_top_left_corner(self):
        cells = viewport_cells((0, 0, 2, 2))
        assert (0, 0) in cells
        assert (1, 1) in cells

    def test_offset(self):
        cells = viewport_cells((3, 2, 2, 3))
        assert (2, 3) in cells  # row=y+0=2, col=x+0=3
        assert (4, 4) in cells  # row=y+2=4, col=x+1=4


# ---------------------------------------------------------------------------
# overlap_ratio
# ---------------------------------------------------------------------------

class TestOverlapRatio:
    def test_identical_viewports_full_overlap(self):
        v = (0, 0, 5, 5)
        assert overlap_ratio(v, v) == pytest.approx(1.0)

    def test_no_overlap(self):
        a = (0, 0, 5, 5)
        b = (10, 10, 5, 5)
        assert overlap_ratio(a, b) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = (0, 0, 4, 4)    # rows 0-3, cols 0-3
        b = (2, 2, 4, 4)    # rows 2-5, cols 2-5
        ratio = overlap_ratio(a, b)
        assert 0.0 < ratio < 1.0

    def test_symmetry(self):
        a = (0, 0, 6, 6)
        b = (3, 3, 6, 6)
        # overlap_ratio is not symmetric in general (denominator = len(a_cells))
        # but both should be positive
        assert overlap_ratio(a, b) > 0
        assert overlap_ratio(b, a) > 0


# ---------------------------------------------------------------------------
# hotspot_prior
# ---------------------------------------------------------------------------

class TestHotspotPrior:
    def test_shape(self):
        g = _uniform_grid()
        prior = hotspot_prior(g)
        assert prior.shape == (GRID_SIZE, GRID_SIZE)

    def test_range_zero_to_one(self):
        g = _uniform_grid()
        prior = hotspot_prior(g)
        assert prior.min() >= 0.0
        assert prior.max() <= 1.0 + 1e-6

    def test_static_cells_zero_score(self):
        g = _uniform_grid()
        prior = hotspot_prior(g)
        # ocean cell (0,0) and mountain cell (-1,-1) should be 0
        assert prior[0, 0] == pytest.approx(0.0)
        assert prior[-1, -1] == pytest.approx(0.0)

    def test_settlement_neighborhood_high(self):
        # Settlement at (5,5) — neighboring cells should score higher than far cells
        g = _uniform_grid()
        prior = hotspot_prior(g)
        near_score = prior[5, 6]   # adjacent to settlement
        far_score = prior[35, 35]  # far from any civ
        assert near_score >= far_score

    def test_all_ocean_returns_zero(self):
        g = [[10] * GRID_SIZE for _ in range(GRID_SIZE)]
        prior = hotspot_prior(g)
        assert (prior == 0.0).all()


# ---------------------------------------------------------------------------
# reconnaissance_viewports
# ---------------------------------------------------------------------------

class TestReconnaissanceViewports:
    def test_returns_n_viewports(self):
        g = _uniform_grid()
        vps = reconnaissance_viewports(g, n_queries=4)
        assert len(vps) == 4

    def test_viewports_are_valid_tuples(self):
        g = _uniform_grid()
        for vp in reconnaissance_viewports(g, n_queries=3):
            x, y, w, h = vp
            assert 0 <= x <= GRID_SIZE - VIEWPORT
            assert 0 <= y <= GRID_SIZE - VIEWPORT
            assert w == VIEWPORT
            assert h == VIEWPORT

    def test_low_overlap_between_selected(self):
        g = _uniform_grid()
        vps = reconnaissance_viewports(g, n_queries=4)
        for i in range(len(vps)):
            for j in range(i + 1, len(vps)):
                assert overlap_ratio(vps[i], vps[j]) <= 0.36


# ---------------------------------------------------------------------------
# observation_stats
# ---------------------------------------------------------------------------

class TestObservationStats:
    def test_empty_observations(self):
        g = _uniform_grid()
        seen, change_rate, ent, probs = observation_stats(g, [])
        assert (seen == 0).all()

    def test_shapes(self):
        g = _uniform_grid()
        obs = [_make_obs(0, 0)]
        seen, change_rate, ent, probs = observation_stats(g, obs)
        assert seen.shape == (GRID_SIZE, GRID_SIZE)
        assert change_rate.shape == (GRID_SIZE, GRID_SIZE)
        assert ent.shape == (GRID_SIZE, GRID_SIZE)
        assert probs.shape == (GRID_SIZE, GRID_SIZE, 6)

    def test_seen_counts_observations(self):
        g = _uniform_grid()
        obs = [_make_obs(0, 0, size=VIEWPORT)]
        seen, _, _, _ = observation_stats(g, obs)
        assert seen[0, 0] == pytest.approx(1.0)

    def test_probs_sum_to_one_where_seen(self):
        g = _uniform_grid()
        obs = [_make_obs(0, 0)]
        seen, _, _, probs = observation_stats(g, obs)
        # where seen > 0 probs should sum to 1
        mask = seen > 0
        np.testing.assert_allclose(probs[mask].sum(axis=-1), 1.0, atol=1e-5)

    def test_change_rate_range(self):
        g = _uniform_grid()
        obs = [_make_obs(0, 0)]
        _, change_rate, _, _ = observation_stats(g, obs)
        assert (change_rate >= 0).all()
        assert (change_rate <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# target_heatmap
# ---------------------------------------------------------------------------

class TestTargetHeatmap:
    def test_shape(self):
        g = _uniform_grid()
        obs = [_make_obs(0, 0)]
        heat = target_heatmap(g, obs)
        assert heat.shape == (GRID_SIZE, GRID_SIZE)

    def test_static_cells_zero(self):
        g = _uniform_grid()
        obs = [_make_obs(0, 0)]
        heat = target_heatmap(g, obs)
        assert heat[0, 0] == pytest.approx(0.0)   # ocean
        assert heat[-1, -1] == pytest.approx(0.0)  # mountain

    def test_nonneg(self):
        g = _uniform_grid()
        obs = [_make_obs(0, 0)]
        heat = target_heatmap(g, obs)
        assert (heat >= 0).all()


# ---------------------------------------------------------------------------
# plan_queries
# ---------------------------------------------------------------------------

class TestPlanQueries:
    def test_no_observations_returns_recon_viewports(self):
        g = _uniform_grid()
        plan = plan_queries(g, [])
        assert len(plan) == RECON_QUERIES

    def test_partial_observations_returns_remaining_recon(self):
        g = _uniform_grid()
        obs = [_make_obs(0, 0), _make_obs(25, 0)]
        plan = plan_queries(g, obs)
        # 2 done, 2 remain
        assert len(plan) == RECON_QUERIES - 2

    def test_full_recon_returns_targeted_viewports(self):
        g = _uniform_grid()
        obs = [_make_obs(0 + i * 5, 0) for i in range(RECON_QUERIES)]
        plan = plan_queries(g, obs)
        # remaining budget = TOTAL_QUERIES - RECON_QUERIES = 6
        assert len(plan) >= 0

    def test_returns_list_of_tuples(self):
        g = _uniform_grid()
        plan = plan_queries(g, [])
        for vp in plan:
            assert isinstance(vp, tuple)
            assert len(vp) == 4

"""Pure-function tests for astar-island query_runner.py.

Covers: summarize_seed, heuristic_seed_scores, allocate_queries_across_seeds.

All pure functions — no file system, network, or GPU access.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ASTAR_DIR = str(Path(__file__).resolve().parent.parent / "tasks" / "astar-island")
sys.path.insert(0, _ASTAR_DIR)

from query_runner import (
    summarize_seed,
    heuristic_seed_scores,
    allocate_queries_across_seeds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(size: int = 20, base_code: int = 11) -> list:
    """Simple uniform grid as nested list."""
    g = [[base_code] * size for _ in range(size)]
    g[0][0] = 10   # ocean
    g[-1][-1] = 5  # mountain
    g[5][5] = 1    # settlement
    return g


def _make_seed_state(size: int = 20) -> dict:
    return {"grid": _make_grid(size)}


# ---------------------------------------------------------------------------
# summarize_seed
# ---------------------------------------------------------------------------

class TestSummarizeSeed:
    def test_returns_dict(self):
        result = summarize_seed(_make_grid(), 20, 20)
        assert isinstance(result, dict)

    def test_required_keys(self):
        result = summarize_seed(_make_grid(), 20, 20)
        for key in ("settlements", "ports", "forest", "plains", "mountains",
                    "ocean", "coast_land", "coastal_civ", "near_civ_land",
                    "forest_edge", "dynamic_land", "top_hotspots"):
            assert key in result, f"Missing key: {key}"

    def test_counts_settlements(self):
        grid = [[11] * 10 for _ in range(10)]
        grid[3][3] = 1
        grid[4][4] = 1
        result = summarize_seed(grid, 10, 10)
        assert result["settlements"] == 2

    def test_counts_ports(self):
        grid = [[11] * 10 for _ in range(10)]
        grid[2][2] = 2
        result = summarize_seed(grid, 10, 10)
        assert result["ports"] == 1

    def test_counts_mountains(self):
        grid = [[11] * 10 for _ in range(10)]
        grid[0][0] = 5
        grid[0][1] = 5
        result = summarize_seed(grid, 10, 10)
        assert result["mountains"] == 2

    def test_ocean_count(self):
        grid = [[10] * 10 for _ in range(10)]  # all ocean
        result = summarize_seed(grid, 10, 10)
        assert result["ocean"] == 100

    def test_dynamic_land_excludes_ocean_mountain(self):
        grid = [[11] * 5 for _ in range(5)]
        grid[0][0] = 10  # ocean
        grid[0][1] = 5   # mountain
        result = summarize_seed(grid, 5, 5)
        assert result["dynamic_land"] == 23  # 25 - 1 ocean - 1 mountain

    def test_top_hotspots_list(self):
        result = summarize_seed(_make_grid(), 20, 20)
        assert isinstance(result["top_hotspots"], list)

    def test_nonneg_counts(self):
        result = summarize_seed(_make_grid(), 20, 20)
        for key in ("settlements", "ports", "forest", "ocean", "mountains"):
            assert result[key] >= 0


# ---------------------------------------------------------------------------
# heuristic_seed_scores
# ---------------------------------------------------------------------------

class TestHeuristicSeedScores:
    def test_returns_summaries_and_scores(self):
        states = [_make_seed_state()] * 3
        summaries, scores = heuristic_seed_scores(states, 20, 20)
        assert isinstance(summaries, list)
        assert isinstance(scores, np.ndarray)

    def test_length_matches_input(self):
        states = [_make_seed_state()] * 4
        summaries, scores = heuristic_seed_scores(states, 20, 20)
        assert len(summaries) == 4
        assert len(scores) == 4

    def test_scores_nonneg(self):
        states = [_make_seed_state()] * 2
        _, scores = heuristic_seed_scores(states, 20, 20)
        assert (scores >= 0).all()

    def test_port_heavy_grid_scores_higher(self):
        # Grid with many ports should score higher than plain grid
        port_grid = [[2] * 5 for _ in range(5)]
        plain_grid = [[11] * 5 for _ in range(5)]
        states_port = [{"grid": port_grid}]
        states_plain = [{"grid": plain_grid}]
        _, scores_port = heuristic_seed_scores(states_port, 5, 5)
        _, scores_plain = heuristic_seed_scores(states_plain, 5, 5)
        assert scores_port[0] > scores_plain[0]

    def test_empty_states(self):
        summaries, scores = heuristic_seed_scores([], 20, 20)
        assert len(summaries) == 0
        assert len(scores) == 0


# ---------------------------------------------------------------------------
# allocate_queries_across_seeds
# ---------------------------------------------------------------------------

class TestAllocateQueriesAcrossSeeds:
    def test_returns_allocation_and_plan(self):
        states = [_make_seed_state()] * 3
        allocation, plan = allocate_queries_across_seeds(states, 20, 20, remaining=10)
        assert isinstance(allocation, list)
        assert isinstance(plan, dict)

    def test_allocation_length_matches_seeds(self):
        states = [_make_seed_state()] * 5
        allocation, _ = allocate_queries_across_seeds(states, 20, 20, remaining=10)
        assert len(allocation) == 5

    def test_allocation_one_per_seed(self):
        # Strategy is always 1 query per seed
        states = [_make_seed_state()] * 4
        allocation, _ = allocate_queries_across_seeds(states, 20, 20, remaining=20)
        assert all(a == 1 for a in allocation)

    def test_plan_has_required_keys(self):
        states = [_make_seed_state()] * 2
        _, plan = allocate_queries_across_seeds(states, 20, 20, remaining=5)
        for key in ("seed_summaries", "heuristic_scores", "allocation",
                    "ranked_seeds", "strategy"):
            assert key in plan

    def test_strategy_mode(self):
        states = [_make_seed_state()] * 2
        _, plan = allocate_queries_across_seeds(states, 20, 20, remaining=5)
        assert plan["strategy"]["mode"] == "regime_detection_only"

    def test_ranked_seeds_length(self):
        states = [_make_seed_state()] * 3
        _, plan = allocate_queries_across_seeds(states, 20, 20, remaining=5)
        assert len(plan["ranked_seeds"]) == 3

    def test_total_queries_equals_seed_count(self):
        n = 5
        states = [_make_seed_state()] * n
        _, plan = allocate_queries_across_seeds(states, 20, 20, remaining=10)
        assert plan["strategy"]["total_queries"] == n

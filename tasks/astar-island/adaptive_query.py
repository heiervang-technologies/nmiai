#!/usr/bin/env python3
"""Adaptive query planning for Astar Island.

Core idea:
- Phase 1: spend 4 reconnaissance queries per seed to identify where sampled
  outcomes differ from the initial state.
- Phase 2: spend the remaining budget re-sampling windows with the strongest
  evidence of change / uncertainty, because those cells dominate KL.

The module exposes:
    plan_queries(initial_grid, seed_observations) -> list[(x, y, w, h)]

It also includes an offline evaluator that approximates simulator calls by
sampling from the released per-cell ground-truth marginals.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, convolve, distance_transform_cdt, uniform_filter

import predictor
import query_runner

log = logging.getLogger(__name__)

GRID_SIZE = 40
VIEWPORT = 15
TOTAL_QUERIES_PER_SEED = 10
RECON_QUERIES = 4
NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float32)


def initial_class_grid(initial_grid: list[list[int]] | np.ndarray) -> np.ndarray:
    grid = np.asarray(initial_grid, dtype=np.int32)
    out = np.zeros_like(grid)
    out[np.isin(grid, [0, 10, 11])] = 0
    out[grid == 1] = 1
    out[grid == 2] = 2
    out[grid == 4] = 4
    out[grid == 5] = 5
    return out


def static_mask(initial_grid: list[list[int]] | np.ndarray) -> np.ndarray:
    grid = np.asarray(initial_grid, dtype=np.int32)
    return np.isin(grid, [5, 10])


def viewport_cells(viewport: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    x, y, w, h = viewport
    return {(y + dy, x + dx) for dy in range(h) for dx in range(w)}


def overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    a_cells = viewport_cells(a)
    b_cells = viewport_cells(b)
    if not a_cells:
        return 0.0
    return len(a_cells & b_cells) / len(a_cells)


def hotspot_prior(initial_grid: list[list[int]] | np.ndarray) -> np.ndarray:
    grid = np.asarray(initial_grid, dtype=np.int32)
    civ_mask = (grid == 1) | (grid == 2)
    ocean_mask = grid == 10
    mountain_mask = grid == 5
    forest_mask = grid == 4

    civ_dist = distance_transform_cdt(~civ_mask, metric="taxicab") if civ_mask.any() else np.full(grid.shape, 99)
    coast = binary_dilation(ocean_mask, np.ones((3, 3), dtype=bool)) & ~ocean_mask
    forest_edge = binary_dilation(forest_mask, np.ones((3, 3), dtype=bool)) & ~forest_mask & ~ocean_mask & ~mountain_mask
    civ_neighbors = convolve(civ_mask.astype(np.float32), NEIGHBOR_KERNEL, mode="constant", cval=0.0)

    score = (
        2.2 * civ_mask.astype(np.float32)
        + 1.2 * np.clip(3.0 - civ_dist, 0.0, 3.0)
        + 0.7 * coast.astype(np.float32)
        + 0.5 * forest_edge.astype(np.float32)
        + 0.35 * civ_neighbors
    )
    score[static_mask(grid)] = 0.0
    max_score = float(score.max())
    if max_score > 0:
        score /= max_score
    return score


def reconnaissance_viewports(initial_grid: list[list[int]] | np.ndarray, n_queries: int = RECON_QUERIES) -> list[tuple[int, int, int, int]]:
    prior = hotspot_prior(initial_grid)
    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for y in range(GRID_SIZE - VIEWPORT + 1):
        for x in range(GRID_SIZE - VIEWPORT + 1):
            window = prior[y : y + VIEWPORT, x : x + VIEWPORT]
            score = float(window.sum())
            candidates.append((score, (x, y, VIEWPORT, VIEWPORT)))

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[int, int, int, int]] = []
    for _, viewport in candidates:
        if len(selected) >= n_queries:
            break
        if any(overlap_ratio(viewport, other) > 0.35 for other in selected):
            continue
        selected.append(viewport)

    if len(selected) < n_queries:
        fallback = [(0, 0, VIEWPORT, VIEWPORT), (25, 0, VIEWPORT, VIEWPORT), (0, 25, VIEWPORT, VIEWPORT), (25, 25, VIEWPORT, VIEWPORT)]
        for viewport in fallback:
            if viewport not in selected:
                selected.append(viewport)
            if len(selected) >= n_queries:
                break

    return selected[:n_queries]


def observation_stats(initial_grid: list[list[int]] | np.ndarray, seed_observations: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    init_cls = initial_class_grid(initial_grid)
    counts = np.zeros((GRID_SIZE, GRID_SIZE, 6), dtype=np.float32)
    seen = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

    for obs in seed_observations:
        grid = np.asarray(obs["grid"], dtype=np.int32)
        x = int(obs["viewport_x"])
        y = int(obs["viewport_y"])
        h, w = grid.shape
        for dy in range(h):
            for dx in range(w):
                cls = predictor.cell_code_to_class(int(grid[dy, dx]))
                yy = y + dy
                xx = x + dx
                counts[yy, xx, cls] += 1.0
                seen[yy, xx] += 1.0

    init_mass = np.take_along_axis(counts, init_cls[..., None], axis=2)[..., 0]
    change_rate = np.divide(seen - init_mass, seen, out=np.zeros_like(seen), where=seen > 0)
    probs = np.divide(counts, seen[..., None], out=np.zeros_like(counts), where=seen[..., None] > 0)
    probs_safe = np.clip(probs, 1e-12, 1.0)
    entropy = -np.sum(np.where(probs > 0, probs * np.log(probs_safe), 0.0), axis=2) / np.log(6.0)
    return seen, change_rate, entropy, probs


def target_heatmap(initial_grid: list[list[int]] | np.ndarray, seed_observations: list[dict]) -> np.ndarray:
    prior = hotspot_prior(initial_grid)
    seen, change_rate, entropy, _ = observation_stats(initial_grid, seed_observations)

    change_support = seen * change_rate
    frontier = uniform_filter(change_support, size=5, mode="constant")
    volatility = 4.0 * change_rate * (1.0 - change_rate)
    resample_need = np.divide(1.0, np.sqrt(seen + 1.0), out=np.ones_like(seen), where=True)

    heat = (
        2.8 * change_rate
        + 1.8 * entropy
        + 1.4 * volatility
        + 1.2 * frontier
        + 0.35 * prior * resample_need
    )

    mask_static = static_mask(initial_grid)
    heat[mask_static] = 0.0
    return heat


def targeted_viewports(initial_grid: list[list[int]] | np.ndarray, seed_observations: list[dict], n_queries: int) -> list[tuple[int, int, int, int]]:
    if n_queries <= 0:
        return []

    heat = target_heatmap(initial_grid, seed_observations)
    used = [
        (
            int(obs["viewport_x"]),
            int(obs["viewport_y"]),
            int(obs.get("viewport_w", VIEWPORT)),
            int(obs.get("viewport_h", VIEWPORT)),
        )
        for obs in seed_observations
    ]

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    for y in range(GRID_SIZE - VIEWPORT + 1):
        for x in range(GRID_SIZE - VIEWPORT + 1):
            viewport = (x, y, VIEWPORT, VIEWPORT)
            window = heat[y : y + VIEWPORT, x : x + VIEWPORT]
            score = float(window.sum())
            if used:
                max_prev_overlap = max(overlap_ratio(viewport, prev) for prev in used)
                score += 25.0 * max_prev_overlap
            candidates.append((score, viewport))

    candidates.sort(key=lambda item: item[0], reverse=True)
    top_windows: list[tuple[float, tuple[int, int, int, int]]] = []
    for score, viewport in candidates:
        if len(top_windows) >= 3:
            break
        if any(overlap_ratio(viewport, other) > 0.70 for _, other in top_windows):
            continue
        top_windows.append((max(score, 1e-6), viewport))

    if not top_windows:
        return reconnaissance_viewports(initial_grid, n_queries=n_queries)

    weights = np.array([score for score, _ in top_windows], dtype=np.float64)
    weights /= weights.sum()
    raw = weights * n_queries
    counts = np.floor(raw).astype(int)
    for idx in np.argsort(-(raw - counts))[: n_queries - int(counts.sum())]:
        counts[idx] += 1
    if counts.sum() == 0:
        counts[0] = n_queries

    plan: list[tuple[int, int, int, int]] = []
    for repeats, (_, viewport) in zip(counts.tolist(), top_windows):
        plan.extend([viewport] * repeats)

    if len(plan) < n_queries:
        plan.extend([top_windows[0][1]] * (n_queries - len(plan)))
    return plan[:n_queries]


def plan_queries(initial_grid: list[list[int]] | np.ndarray, seed_observations: list[dict]) -> list[tuple[int, int, int, int]]:
    """Plan the next queries for one seed.

    If fewer than 4 observations exist, returns the remaining reconnaissance
    windows. Otherwise returns the remaining targeted re-sampling windows up to
    a 10-query total budget for the seed.
    """
    used = {
        (
            int(obs["viewport_x"]),
            int(obs["viewport_y"]),
            int(obs.get("viewport_w", VIEWPORT)),
            int(obs.get("viewport_h", VIEWPORT)),
        )
        for obs in seed_observations
    }
    if len(seed_observations) < RECON_QUERIES:
        recon = reconnaissance_viewports(initial_grid, RECON_QUERIES)
        return [viewport for viewport in recon if viewport not in used][: RECON_QUERIES - len(seed_observations)]

    remaining = max(0, TOTAL_QUERIES_PER_SEED - len(seed_observations))
    return targeted_viewports(initial_grid, seed_observations, remaining)


def sample_query(
    ground_truth: np.ndarray,
    viewport: tuple[int, int, int, int],
    rng: np.random.Generator,
) -> dict:
    x, y, w, h = viewport
    patch = ground_truth[y : y + h, x : x + w]
    flat = patch.reshape(-1, patch.shape[-1])
    sampled = np.array([rng.choice(6, p=row) for row in flat], dtype=np.int32).reshape(h, w)
    return {
        "grid": sampled.tolist(),
        "viewport_x": x,
        "viewport_y": y,
        "viewport_w": w,
        "viewport_h": h,
    }


def run_planner(initial_grid: list[list[int]] | np.ndarray, ground_truth: np.ndarray, rng: np.random.Generator) -> list[dict]:
    observations: list[dict] = []
    while len(observations) < TOTAL_QUERIES_PER_SEED:
        planned = plan_queries(initial_grid, observations)
        if not planned:
            break
        next_viewport = planned[0]
        observations.append(sample_query(ground_truth, next_viewport, rng))
    return observations


def run_hotspot_baseline(initial_grid: list[list[int]] | np.ndarray, ground_truth: np.ndarray, rng: np.random.Generator) -> list[dict]:
    viewports = query_runner.select_viewports_adaptive(initial_grid, GRID_SIZE, GRID_SIZE, TOTAL_QUERIES_PER_SEED)
    return [sample_query(ground_truth, viewport, rng) for viewport in viewports]


def train_eval_model(model_config: str = "phase"):
    X_basic, X_phase, X_dsl, Y, _, rule_names = predictor.load_training_data()
    try:
        model = predictor.build_model(model_config).fit(X_basic, X_phase, X_dsl, Y, rule_names)
        return model, model_config
    except Exception as exc:
        fallback = "base"
        log.warning("model_config=%s failed (%s); falling back to %s", model_config, exc, fallback)
        model = predictor.build_model(fallback).fit(X_basic, X_phase, X_dsl, Y, rule_names)
        return model, fallback


def evaluate_strategies(n_trials: int = 4, model_config: str = "phase") -> dict:
    samples = []
    gt_dir = Path(__file__).parent / "ground_truth"
    for path in sorted(gt_dir.glob("round*_seed*.json")):
        with open(path) as fh:
            data = json.load(fh)
        if "initial_grid" not in data or "ground_truth" not in data:
            continue
        samples.append(
            {
                "name": path.stem,
                "initial_grid": data["initial_grid"],
                "ground_truth": np.asarray(data["ground_truth"], dtype=np.float64),
            }
        )

    model, resolved_model = train_eval_model(model_config=model_config)
    totals = {"no_queries": [], "hotspot": [], "adaptive_change": []}

    for trial in range(n_trials):
        rng = np.random.default_rng(1000 + trial)
        for sample in samples:
            gt = sample["ground_truth"]
            initial_grid = sample["initial_grid"]

            pred_no_queries = predictor.build_prediction(initial_grid, model, observations=[])
            hotspot_obs = run_hotspot_baseline(initial_grid, gt, rng)
            adaptive_obs = run_planner(initial_grid, gt, rng)
            pred_hotspot = predictor.build_prediction(initial_grid, model, hotspot_obs)
            pred_adaptive = predictor.build_prediction(initial_grid, model, adaptive_obs)

            totals["no_queries"].append(predictor.weighted_kl_divergence(gt, pred_no_queries))
            totals["hotspot"].append(predictor.weighted_kl_divergence(gt, pred_hotspot))
            totals["adaptive_change"].append(predictor.weighted_kl_divergence(gt, pred_adaptive))

    summary = {
        "model_config_requested": model_config,
        "model_config_used": resolved_model,
        "n_trials": n_trials,
        "num_seed_files": len(samples),
        "mean_weighted_kl": {key: float(np.mean(values)) for key, values in totals.items()},
    }
    summary["delta_vs_hotspot"] = {
        "adaptive_change": summary["mean_weighted_kl"]["adaptive_change"] - summary["mean_weighted_kl"]["hotspot"],
        "no_queries": summary["mean_weighted_kl"]["no_queries"] - summary["mean_weighted_kl"]["hotspot"],
    }
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    summary = evaluate_strategies()
    print(json.dumps(summary, indent=2))

    out_path = Path(__file__).parent / "benchmark_results" / "adaptive_query.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()

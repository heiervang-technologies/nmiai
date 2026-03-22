#!/usr/bin/env python3
"""Replay-boosted Bayesian template predictor for Astar Island.

Uses 51-frame replay simulations to compute additive residual corrections
on the base predictor's bucket-level predictions. Corrections are applied
to the PRIOR prediction, then observations are overlaid on the corrected
prior using the same tau=20 Bayesian update.

CV results (LOOCV, 19 rounds, 3 obs viewports):
  Base:  no_obs=0.114067  with_obs=0.047913
  Boost: no_obs=0.111459  with_obs=0.047309
  Improvement: +2.3% no_obs, +1.3% with_obs

Also extracts growth timing patterns from replay intermediate frames:
  - Distance 0-1 cells: change early (steps 0-16), ~55% / ~30%
  - Distance 3-5 cells: change late (steps 34-50), ~53-74%
  - Distance 6+ cells: very late (steps 40+), ~74-85%
"""

import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from bayesian_template_predictor import (
    BASE_DIR,
    FLOOR,
    FOREST,
    MOUNTAIN,
    N_CLASSES,
    OCEAN,
    PORT,
    SETTLEMENT,
    build_round_bucket_tables,
    cell_bucket,
    cell_code_to_class,
    compute_features,
    compute_template_log_likelihoods,
    entropy,
    interpolate_lookup,
    kl_divergence,
    load_ground_truth,
    posterior_weights,
    predict_single_template,
    simulate_observations,
)

REPLAY_DIR = BASE_DIR / "replays"


# ---------------------------------------------------------------------------
# Replay loading
# ---------------------------------------------------------------------------

def load_replays():
    replays = {}
    for f in sorted(list(REPLAY_DIR.glob("round*_seed*.json")) + list((REPLAY_DIR / "dense_training").glob("round*_seed*.json")) if (REPLAY_DIR / "dense_training").exists() else sorted(REPLAY_DIR.glob("round*_seed*.json"))):
        parts = f.stem.split("_")
        rn = int(parts[0].replace("round", ""))
        sn = int(parts[1].replace("seed", ""))
        replays[(rn, sn)] = json.loads(f.read_text())
    return replays


def replay_to_onehot(replay):
    final_grid = replay["frames"][-1]["grid"]
    h, w = len(final_grid), len(final_grid[0])
    onehot = np.zeros((h, w, N_CLASSES), dtype=np.float64)
    for y in range(h):
        for x in range(w):
            onehot[y, x, cell_code_to_class(final_grid[y][x])] = 1.0
    return onehot


# ---------------------------------------------------------------------------
# Growth timing analysis
# ---------------------------------------------------------------------------

def analyze_growth_timing(replays):
    """Track at which step each cell first changes, grouped by distance."""
    changes_by_dist = defaultdict(list)

    for (rn, sn), replay in replays.items():
        ig = np.array(replay["frames"][0]["grid"], dtype=np.int32)
        h, w = ig.shape
        dist_civ, _, _, _ = compute_features(ig)

        first_change = np.full((h, w), -1, dtype=np.int32)
        for step_idx in range(1, len(replay["frames"])):
            frame_grid = replay["frames"][step_idx]["grid"]
            for y in range(h):
                for x in range(w):
                    if first_change[y, x] == -1:
                        code = int(ig[y, x])
                        if code in (OCEAN, MOUNTAIN):
                            continue
                        if frame_grid[y][x] != ig[y, x]:
                            first_change[y, x] = step_idx

        for y in range(h):
            for x in range(w):
                if first_change[y, x] > 0:
                    d = int(min(max(math.floor(dist_civ[y, x]), 0), 15))
                    changes_by_dist[d].append(first_change[y, x])

    return {d: np.array(steps) for d, steps in changes_by_dist.items()}


# ---------------------------------------------------------------------------
# Residual computation
# ---------------------------------------------------------------------------

def compute_pooled_residuals(replays, rounds_data, round_nums):
    """Per-bucket residuals: empirical_replay - base_prediction."""
    _, _, pooled_tables, pooled_counts = \
        build_round_bucket_tables(rounds_data, round_nums)

    residual_sums = [defaultdict(lambda: np.zeros(N_CLASSES, dtype=np.float64)) for _ in range(4)]
    residual_counts = [defaultdict(float) for _ in range(4)]

    for (rn, sn), replay in replays.items():
        if rn not in round_nums:
            continue

        ig = np.array(replay["frames"][0]["grid"], dtype=np.int32)
        h, w = ig.shape
        dist_civ, n_ocean, n_civ, coast = compute_features(ig)
        empirical = replay_to_onehot(replay)

        for y in range(h):
            for x in range(w):
                code = int(ig[y, x])
                if code in (OCEAN, MOUNTAIN):
                    continue

                keys = cell_bucket(code, dist_civ[y, x], n_ocean[y, x],
                                   n_civ[y, x], coast[y, x])
                if keys is None:
                    continue

                q_pred, _ = interpolate_lookup(
                    pooled_tables, pooled_counts,
                    ig, dist_civ, n_ocean, n_civ, coast, y, x
                )
                if q_pred is None:
                    continue

                residual = empirical[y, x] - q_pred
                for level, key in enumerate(keys):
                    residual_sums[level][key] = residual_sums[level][key] + residual
                    residual_counts[level][key] += 1.0

    residuals = [{} for _ in range(4)]
    counts_final = [{} for _ in range(4)]
    for level in range(4):
        for key, vec in residual_sums[level].items():
            c = residual_counts[level][key]
            if c > 0:
                residuals[level][key] = vec / c
            counts_final[level][key] = c

    return residuals, counts_final


def lookup_residual(residuals, residual_counts, keys, min_counts=(20, 30, 50, 10)):
    """Hierarchical residual lookup with support thresholds."""
    for level, key in enumerate(keys):
        if key in residuals[level]:
            c = residual_counts[level].get(key, 0)
            if c >= min_counts[level]:
                return residuals[level][key], c, level
    for level, key in enumerate(keys):
        if key in residuals[level]:
            c = residual_counts[level].get(key, 0)
            if c >= 5:
                return residuals[level][key], c, level
    return None, 0, 3


# ---------------------------------------------------------------------------
# Core prediction with residual correction + obs overlay
# ---------------------------------------------------------------------------

def predict_with_residuals(ig, per_round_tables, per_round_counts,
                           pooled_tables, pooled_counts,
                           train_rounds,
                           residuals, residual_counts,
                           residual_weight=0.5,
                           observations=None):
    """Full prediction pipeline:
    1. Compute template weights (using observations if available)
    2. Build weighted mixture prior
    3. Apply residual correction to prior
    4. Apply observation overlay on corrected prior
    5. Structural constraints + renormalize
    """
    h, w = ig.shape
    dist_civ, n_ocean, n_civ, coast = compute_features(ig)

    # --- Step 1: Template weights ---
    if observations:
        log_liks = compute_template_log_likelihoods(
            ig, observations, per_round_tables, per_round_counts, train_rounds
        )
        weights = posterior_weights(log_liks, train_rounds)
    else:
        weights = {rn: 1.0 / len(train_rounds) for rn in train_rounds}

    # --- Step 2: Weighted mixture prior ---
    template_preds = {}
    for rn in train_rounds:
        template_preds[rn] = predict_single_template(
            ig, dist_civ, n_ocean, n_civ, coast,
            per_round_tables[rn], per_round_counts[rn],
            pooled_tables, pooled_counts
        )

    pred = np.zeros((h, w, N_CLASSES), dtype=np.float64)
    for rn in train_rounds:
        pred += weights[rn] * template_preds[rn]

    # --- Step 3: Apply residual correction to prior ---
    if residual_weight > 0:
        for y in range(h):
            for x in range(w):
                code = int(ig[y, x])
                if code in (OCEAN, MOUNTAIN):
                    continue

                keys = cell_bucket(code, dist_civ[y, x], n_ocean[y, x],
                                   n_civ[y, x], coast[y, x])
                if keys is None:
                    continue

                res, c, _ = lookup_residual(residuals, residual_counts, keys)
                if res is not None:
                    confidence = min(c / 100.0, 1.0)
                    pred[y, x] += residual_weight * confidence * res

    # --- Step 4: Apply observation overlay (tau=20) ---
    if observations:
        pred = np.maximum(pred, FLOOR)
        pred /= pred.sum(axis=2, keepdims=True)

        obs_counts = np.zeros((h, w, N_CLASSES), dtype=np.float64)
        obs_total = np.zeros((h, w), dtype=np.float64)
        for obs in observations:
            grid = obs.get("grid", [])
            vx = int(obs.get("viewport_x", 0))
            vy = int(obs.get("viewport_y", 0))
            for dy, row in enumerate(grid):
                for dx, cell_val in enumerate(row):
                    y2 = vy + dy
                    x2 = vx + dx
                    if 0 <= y2 < h and 0 <= x2 < w:
                        cls = cell_code_to_class(int(cell_val))
                        obs_counts[y2, x2, cls] += 1.0
                        obs_total[y2, x2] += 1.0

        tau = 20.0
        mask = obs_total >= 2.0
        if mask.any():
            posterior_num = obs_counts[mask] + tau * pred[mask]
            posterior_denom = obs_total[mask, None] + tau
            pred[mask] = posterior_num / posterior_denom

    # --- Step 5: Structural constraints + renormalize ---
    for y in range(h):
        for x in range(w):
            code = int(ig[y, x])
            if code in (OCEAN, MOUNTAIN):
                continue
            if not coast[y, x]:
                pred[y, x, 2] = 0.0
            pred[y, x, 5] = 0.0
            if dist_civ[y, x] > 10:
                pred[y, x, 3] = 0.0

    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred, weights


# ---------------------------------------------------------------------------
# Model caching and public API
# ---------------------------------------------------------------------------

_CACHED_MODEL = None


def get_or_build_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    rounds = load_ground_truth()
    all_rounds = sorted(rounds.keys())
    replays = load_replays()

    per_round_tables, per_round_counts, pooled_tables, pooled_counts = \
        build_round_bucket_tables(rounds, all_rounds)

    residuals, residual_counts = \
        compute_pooled_residuals(replays, rounds, all_rounds)

    _CACHED_MODEL = {
        "per_round_tables": per_round_tables,
        "per_round_counts": per_round_counts,
        "pooled_tables": pooled_tables,
        "pooled_counts": pooled_counts,
        "train_rounds": all_rounds,
        "residuals": residuals,
        "residual_counts": residual_counts,
    }
    return _CACHED_MODEL


def predict(initial_grid, observations=None):
    """Public API: predict using replay-boosted Bayesian template mixture.

    Args:
        initial_grid: 2D array of cell codes (40x40)
        observations: list of viewport observations (optional)

    Returns:
        pred: (h, w, 6) probability distribution per cell
    """
    model = get_or_build_model()
    ig = np.array(initial_grid, dtype=np.int32)
    pred, _ = predict_with_residuals(
        ig,
        model["per_round_tables"],
        model["per_round_counts"],
        model["pooled_tables"],
        model["pooled_counts"],
        model["train_rounds"],
        model["residuals"],
        model["residual_counts"],
        residual_weight=0.5,
        observations=observations,
    )
    return pred


# ---------------------------------------------------------------------------
# CV and diagnostics
# ---------------------------------------------------------------------------

def run_cv(n_obs_viewports=3, residual_weight=0.5):
    rng = np.random.RandomState(42)
    rounds = load_ground_truth()
    all_round_nums = sorted(rounds.keys())
    replays = load_replays()

    print(f"Replay-boosted LOOCV")
    print(f"Rounds: {len(all_round_nums)}, Replays: {len(replays)}, weight: {residual_weight}")
    print()

    base_no = []
    base_obs = []
    corr_no = []
    corr_obs = []

    for held_out in all_round_nums:
        train_rounds = [rn for rn in all_round_nums if rn != held_out]
        train_replays = {k: v for k, v in replays.items() if k[0] != held_out}

        t0 = time.time()
        prt, prc, pt, pc = build_round_bucket_tables(rounds, train_rounds)
        res, res_c = compute_pooled_residuals(train_replays, rounds, train_rounds)
        build_time = time.time() - t0

        wkls = {"bn": [], "bo": [], "cn": [], "co": []}

        for sn in sorted(rounds[held_out].keys()):
            sd = rounds[held_out][sn]
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]
            H = entropy(gt)
            dynamic = H > 0.01
            if not dynamic.any():
                continue

            obs = simulate_observations(gt, ig, n_obs_viewports, rng)

            # Base no-obs
            p, _ = predict_with_residuals(
                ig, prt, prc, pt, pc, train_rounds,
                res, res_c, residual_weight=0.0, observations=None)
            wkls["bn"].append(float((H * kl_divergence(gt, p))[dynamic].mean()))

            # Base with-obs
            p, _ = predict_with_residuals(
                ig, prt, prc, pt, pc, train_rounds,
                res, res_c, residual_weight=0.0, observations=obs)
            wkls["bo"].append(float((H * kl_divergence(gt, p))[dynamic].mean()))

            # Corrected no-obs
            p, _ = predict_with_residuals(
                ig, prt, prc, pt, pc, train_rounds,
                res, res_c, residual_weight=residual_weight, observations=None)
            wkls["cn"].append(float((H * kl_divergence(gt, p))[dynamic].mean()))

            # Corrected with-obs
            p, _ = predict_with_residuals(
                ig, prt, prc, pt, pc, train_rounds,
                res, res_c, residual_weight=residual_weight, observations=obs)
            wkls["co"].append(float((H * kl_divergence(gt, p))[dynamic].mean()))

        bn = np.mean(wkls["bn"])
        bo = np.mean(wkls["bo"])
        cn = np.mean(wkls["cn"])
        co = np.mean(wkls["co"])

        base_no.append(bn)
        base_obs.append(bo)
        corr_no.append(cn)
        corr_obs.append(co)

        print(f"  R{held_out:2d}: no={bn:.6f}->{cn:.6f} d={cn-bn:+.6f}  "
              f"obs={bo:.6f}->{co:.6f} d={co-bo:+.6f}  t={build_time:.1f}s")

    mbn = np.mean(base_no)
    mbo = np.mean(base_obs)
    mcn = np.mean(corr_no)
    mco = np.mean(corr_obs)

    print(f"\n{'='*70}")
    print(f"BASE  no_obs: {mbn:.6f}   with_obs: {mbo:.6f}")
    print(f"BOOST no_obs: {mcn:.6f}   with_obs: {mco:.6f}")
    print(f"DELTA no_obs: {mcn-mbn:+.6f}   with_obs: {mco-mbo:+.6f}")
    imp_n = (mbn - mcn) / mbn * 100
    imp_o = (mbo - mco) / mbo * 100
    print(f"Improvement no_obs: {imp_n:+.1f}%  with_obs: {imp_o:+.1f}%")


def print_timing_report():
    """Print growth timing analysis from replays."""
    replays = load_replays()
    timing = analyze_growth_timing(replays)

    print("Growth Timing Analysis (from replays)")
    print(f"{'Dist':>5} {'Early%':>7} {'Mid%':>7} {'Late%':>7} {'MeanStep':>9} {'Count':>6}")
    print("-" * 50)
    for d in sorted(timing.keys()):
        steps = timing[d]
        early = np.mean(steps <= 16)
        mid = np.mean((steps > 16) & (steps <= 33))
        late = np.mean(steps > 33)
        print(f"{d:5d} {early:7.1%} {mid:7.1%} {late:7.1%} "
              f"{steps.mean():9.1f} {len(steps):6d}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--timing", action="store_true", help="Growth timing report")
    p.add_argument("--weight", type=float, default=0.5)
    a = p.parse_args()
    if a.timing:
        print_timing_report()
    else:
        run_cv(residual_weight=a.weight)

#!/usr/bin/env python3
"""Query allocation optimizer for Astar Island - FAST version.

Pre-builds all LOOCV models, then tests strategies efficiently.
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.ndimage import gaussian_filter, convolve, distance_transform_cdt

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from bayesian_template_predictor import (
    FLOOR, FOREST, MOUNTAIN, N_CLASSES, OCEAN, PORT, SETTLEMENT,
    VIEWPORT_SIZE, build_round_bucket_tables, cell_bucket, cell_code_to_class,
    compute_features, compute_template_log_likelihoods, entropy, interpolate_lookup,
    kl_divergence, load_ground_truth, posterior_weights, predict_single_template,
)
from replay_boosted_predictor import (
    compute_pooled_residuals, load_replays, lookup_residual,
    predict_with_residuals,
)

GRID_SIZE = 40
VP = VIEWPORT_SIZE  # 15

def wkl_score(pred, gt, ig):
    H = entropy(gt)
    dynamic = H > 0.01
    if not dynamic.any():
        return 0.0
    kl = kl_divergence(gt, pred)
    return float((H * kl)[dynamic].mean())

def sample_viewport(gt, ig, vy, vx, rng):
    h, w, _ = gt.shape
    grid_sample = []
    for dy in range(VP):
        row = []
        for dx in range(VP):
            y2, x2 = vy + dy, vx + dx
            if 0 <= y2 < h and 0 <= x2 < w:
                cls = rng.choice(N_CLASSES, p=gt[y2, x2])
                row.append(cls)
            else:
                row.append(0)
        grid_sample.append(row)
    return {"grid": grid_sample, "viewport_x": int(vx), "viewport_y": int(vy)}

def get_hotspot_scores(ig):
    civ = (ig == SETTLEMENT) | (ig == PORT)
    if not civ.any():
        return np.ones((GRID_SIZE - VP + 1, GRID_SIZE - VP + 1))
    dist = distance_transform_cdt(~civ, metric="taxicab")
    h, w = ig.shape
    scores = np.zeros((h - VP + 1, w - VP + 1))
    for vy in range(h - VP + 1):
        for vx in range(w - VP + 1):
            ig_patch = ig[vy:vy+VP, vx:vx+VP]
            dynamic = (ig_patch != OCEAN) & (ig_patch != MOUNTAIN)
            if dynamic.any():
                d_patch = dist[vy:vy+VP, vx:vx+VP]
                scores[vy, vx] = np.sum(1.0 / (1.0 + d_patch[dynamic]))
    return scores

def get_pred_entropy_scores(pred, ig):
    h, w, _ = pred.shape
    scores = np.zeros((h - VP + 1, w - VP + 1))
    for vy in range(h - VP + 1):
        for vx in range(w - VP + 1):
            patch = pred[vy:vy+VP, vx:vx+VP]
            ig_patch = ig[vy:vy+VP, vx:vx+VP]
            dynamic = (ig_patch != OCEAN) & (ig_patch != MOUNTAIN)
            if dynamic.any():
                ent = -np.sum(np.maximum(patch, 1e-10) * np.log2(np.maximum(patch, 1e-10)), axis=2)
                scores[vy, vx] = ent[dynamic].sum()
    return scores

def pick_viewports(scores, n, min_dist=8):
    positions = []
    flat = scores.ravel()
    sorted_idx = np.argsort(-flat)
    for idx in sorted_idx:
        vy = idx // scores.shape[1]
        vx = idx % scores.shape[1]
        too_close = any(abs(vy - py) < min_dist and abs(vx - px) < min_dist for py, px in positions)
        if not too_close:
            positions.append((vy, vx))
            if len(positions) >= n:
                break
    # Relax distance if needed
    if len(positions) < n:
        for md in [6, 4, 2, 0]:
            positions = []
            for idx in sorted_idx:
                vy = idx // scores.shape[1]
                vx = idx % scores.shape[1]
                too_close = any(abs(vy - py) < md and abs(vx - px) < md for py, px in positions)
                if not too_close:
                    positions.append((vy, vx))
                    if len(positions) >= n:
                        break
            if len(positions) >= n:
                break
    return positions

def apply_tau_overlay(pred, observations, ig, tau=100.0):
    h, w, _ = pred.shape
    pred = pred.copy()
    obs_counts = np.zeros((h, w, N_CLASSES), dtype=np.float64)
    obs_total = np.zeros((h, w), dtype=np.float64)
    for obs in observations:
        grid = obs["grid"]
        vx, vy = obs["viewport_x"], obs["viewport_y"]
        for dy, row in enumerate(grid):
            for dx, cell_val in enumerate(row):
                y2, x2 = vy + dy, vx + dx
                if 0 <= y2 < h and 0 <= x2 < w:
                    obs_counts[y2, x2, int(cell_val)] += 1.0
                    obs_total[y2, x2] += 1.0
    mask = obs_total >= 1.0
    if mask.any():
        pred[mask] = (obs_counts[mask] + tau * pred[mask]) / (obs_total[mask, None] + tau)
    return pred

def apply_constraints(pred, ig):
    h, w, _ = pred.shape
    kernel = np.array([[1,1,1],[1,0,1],[1,1,1]], dtype=np.int32)
    n_ocean = convolve((ig == OCEAN).astype(np.int32), kernel, mode="constant")
    coast = (n_ocean > 0) & (ig != OCEAN)
    dist_civ, _, _, _ = compute_features(ig)
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
    return pred

def predict_with_obs(ig, observations, model, tau=100.0, sigma=0.3):
    """Get base prediction (no obs), then apply our own tau overlay + smoothing."""
    pred_base, _ = predict_with_residuals(
        ig, model["prt"], model["prc"], model["pt"], model["pc"],
        model["train_rounds"], model["res"], model["res_c"],
        residual_weight=0.5, observations=None)
    if observations:
        pred = apply_tau_overlay(pred_base, observations, ig, tau=tau)
    else:
        pred = pred_base.copy()
    pred = apply_constraints(pred, ig)
    if sigma > 0:
        dynamic = (ig != OCEAN) & (ig != MOUNTAIN)
        for c in range(N_CLASSES):
            ch = pred[:, :, c].copy()
            ch[~dynamic] = 0
            sm = gaussian_filter(ch, sigma=sigma)
            pred[:, :, c] = np.where(dynamic, sm, pred[:, :, c])
        pred = np.maximum(pred, FLOOR)
        pred /= pred.sum(axis=2, keepdims=True)
    return pred


# ---------------------------------------------------------------------------
# Strategy functions: each returns list of observations for a seed
# ---------------------------------------------------------------------------

def make_obs_uniform(gt, ig, n_vp, q_per_vp, scores, rng):
    positions = pick_viewports(scores, n_vp)
    obs = []
    for vy, vx in positions:
        for _ in range(q_per_vp):
            obs.append(sample_viewport(gt, ig, vy, vx, rng))
    return obs

def make_obs_adaptive_3phase(gt, ig, model, rng):
    """Phase1: 1q hotspot. Phase2: 2vp x 2q at entropy. Phase3: 1vp x 5q."""
    hs = get_hotspot_scores(ig)
    pos1 = pick_viewports(hs, 1)
    obs = [sample_viewport(gt, ig, pos1[0][0], pos1[0][1], rng)]

    pred1 = predict_with_obs(ig, obs, model, tau=100.0, sigma=0.3)
    es = get_pred_entropy_scores(pred1, ig)
    for vy, vx in pos1:
        r0 = max(0, vy - VP + 1); r1 = min(es.shape[0], vy + 1)
        c0 = max(0, vx - VP + 1); c1 = min(es.shape[1], vx + 1)
        es[r0:r1, c0:c1] *= 0.3
    pos2 = pick_viewports(es, 2, min_dist=8)
    for vy, vx in pos2:
        for _ in range(2):
            obs.append(sample_viewport(gt, ig, vy, vx, rng))

    pred2 = predict_with_obs(ig, obs, model, tau=100.0, sigma=0.3)
    es2 = get_pred_entropy_scores(pred2, ig)
    for vy, vx in pos1 + pos2:
        r0 = max(0, vy - VP + 1); r1 = min(es2.shape[0], vy + 1)
        c0 = max(0, vx - VP + 1); c1 = min(es2.shape[1], vx + 1)
        es2[r0:r1, c0:c1] *= 0.2
    pos3 = pick_viewports(es2, 1, min_dist=6)
    for vy, vx in pos3:
        for _ in range(5):
            obs.append(sample_viewport(gt, ig, vy, vx, rng))
    return obs  # 1 + 4 + 5 = 10

def make_obs_regime_first(gt, ig, model, rng):
    """2q regime detect, then 3vp x (3,3,2)q."""
    hs = get_hotspot_scores(ig)
    pos_r = pick_viewports(hs, 1)
    obs = []
    for vy, vx in pos_r:
        for _ in range(2):
            obs.append(sample_viewport(gt, ig, vy, vx, rng))
    pred1 = predict_with_obs(ig, obs, model, tau=100.0, sigma=0.3)
    es = get_pred_entropy_scores(pred1, ig)
    for vy, vx in pos_r:
        r0 = max(0, vy - VP + 1); r1 = min(es.shape[0], vy + 1)
        c0 = max(0, vx - VP + 1); c1 = min(es.shape[1], vx + 1)
        es[r0:r1, c0:c1] *= 0.3
    pos_e = pick_viewports(es, 3, min_dist=8)
    allocs = [3, 3, 2]
    for i, (vy, vx) in enumerate(pos_e):
        for _ in range(allocs[i] if i < len(allocs) else 2):
            obs.append(sample_viewport(gt, ig, vy, vx, rng))
    return obs  # 2 + 8 = 10

def make_obs_4vp_adaptive(gt, ig, model, rng):
    """4vp x 2q + 2 extra at most uncertain."""
    pred0, _ = predict_with_residuals(
        ig, model["prt"], model["prc"], model["pt"], model["pc"],
        model["train_rounds"], model["res"], model["res_c"],
        residual_weight=0.5, observations=None)
    es = get_pred_entropy_scores(pred0, ig)
    pos = pick_viewports(es, 4, min_dist=6)
    obs = []
    for vy, vx in pos:
        for _ in range(2):
            obs.append(sample_viewport(gt, ig, vy, vx, rng))
    # Find most uncertain viewport for 2 extra
    pred1 = predict_with_obs(ig, obs, model, tau=100.0, sigma=0.3)
    best_vp, best_s = pos[0], -1
    for vy, vx in pos:
        p = pred1[vy:vy+VP, vx:vx+VP]
        ip = ig[vy:vy+VP, vx:vx+VP]
        d = (ip != OCEAN) & (ip != MOUNTAIN)
        if d.any():
            ent = -np.sum(np.maximum(p, 1e-10) * np.log2(np.maximum(p, 1e-10)), axis=2)
            s = ent[d].sum()
            if s > best_s:
                best_s = s
                best_vp = (vy, vx)
    for _ in range(2):
        obs.append(sample_viewport(gt, ig, best_vp[0], best_vp[1], rng))
    return obs  # 8 + 2 = 10


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("QUERY ALLOCATION OPTIMIZATION")
    print("=" * 70)

    t0 = time.time()
    rounds = load_ground_truth()
    replays = load_replays()
    all_rns = sorted(rounds.keys())
    print(f"Loaded {len(all_rns)} rounds, {len(replays)} replays")
    sys.stdout.flush()

    # Convert to numpy
    for rn in rounds:
        for sn in rounds[rn]:
            if isinstance(rounds[rn][sn]["ground_truth"], list):
                rounds[rn][sn]["ground_truth"] = np.array(rounds[rn][sn]["ground_truth"], dtype=np.float64)
            if isinstance(rounds[rn][sn]["initial_grid"], list):
                rounds[rn][sn]["initial_grid"] = np.array(rounds[rn][sn]["initial_grid"], dtype=np.int32)

    # Pre-build all LOOCV models
    print("Pre-building LOOCV models...", end=" ", flush=True)
    models = {}
    for held_out in all_rns:
        train_rns = [r for r in all_rns if r != held_out]
        train_replays = {k: v for k, v in replays.items() if k[0] != held_out}
        prt, prc, pt, pc = build_round_bucket_tables(rounds, train_rns)
        res, res_c = compute_pooled_residuals(train_replays, rounds, train_rns)
        models[held_out] = {
            "prt": prt, "prc": prc, "pt": pt, "pc": pc,
            "train_rounds": train_rns, "res": res, "res_c": res_c,
        }
    print(f"done ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

    # Also pre-compute hotspot scores and base predictions for each seed
    print("Pre-computing base predictions...", end=" ", flush=True)
    base_preds = {}  # (held_out, sn) -> pred_no_obs
    hotspot_cache = {}  # (held_out, sn) -> scores
    for held_out in all_rns:
        model = models[held_out]
        for sn in sorted(rounds[held_out].keys()):
            ig = rounds[held_out][sn]["initial_grid"]
            pred, _ = predict_with_residuals(
                ig, model["prt"], model["prc"], model["pt"], model["pc"],
                model["train_rounds"], model["res"], model["res_c"],
                residual_weight=0.5, observations=None)
            base_preds[(held_out, sn)] = pred
            hotspot_cache[(held_out, sn)] = get_hotspot_scores(ig)
    print(f"done ({time.time()-t0:.0f}s)")
    sys.stdout.flush()

    # Define strategies
    def eval_strategy(name, obs_fn, n_trials=3):
        """Evaluate a strategy across all LOOCV folds."""
        trial_means = []
        for trial in range(n_trials):
            rng = np.random.RandomState(42 + trial * 1000)
            round_wkls = []
            for held_out in all_rns:
                model = models[held_out]
                seed_wkls = []
                for sn in sorted(rounds[held_out].keys()):
                    gt = rounds[held_out][sn]["ground_truth"]
                    ig = rounds[held_out][sn]["initial_grid"]
                    obs = obs_fn(gt, ig, model, held_out, sn, rng)
                    pred = predict_with_obs(ig, obs, model, tau=100.0, sigma=0.3)
                    seed_wkls.append(wkl_score(pred, gt, ig))
                round_wkls.append(np.mean(seed_wkls))
            trial_means.append(np.mean(round_wkls))
        return np.mean(trial_means), np.std(trial_means)

    strategies = {}

    # 1. Uniform strategies with hotspot placement
    for n_vp, q_per in [(1, 10), (2, 5), (3, 3), (5, 2)]:
        label = f"{n_vp}vp x {q_per}q (hotspot)"
        def make_fn(nv=n_vp, qp=q_per):
            def fn(gt, ig, model, ho, sn, rng):
                return make_obs_uniform(gt, ig, nv, qp, hotspot_cache[(ho, sn)], rng)
            return fn
        strategies[label] = make_fn()

    # 2. Uniform with pred entropy placement
    for n_vp, q_per in [(2, 5), (3, 3), (4, 2), (5, 2)]:
        label = f"{n_vp}vp x {q_per}q (pred_ent)"
        def make_fn(nv=n_vp, qp=q_per):
            def fn(gt, ig, model, ho, sn, rng):
                scores = get_pred_entropy_scores(base_preds[(ho, sn)], ig)
                return make_obs_uniform(gt, ig, nv, qp, scores, rng)
            return fn
        strategies[label] = make_fn()

    # 3. GT entropy placement (oracle upper bound)
    for n_vp, q_per in [(3, 3)]:
        label = f"{n_vp}vp x {q_per}q (gt_ent ORACLE)"
        def make_fn(nv=n_vp, qp=q_per):
            def fn(gt, ig, model, ho, sn, rng):
                from bayesian_template_predictor import entropy as ent_fn
                h, w, _ = gt.shape
                scores = np.zeros((h - VP + 1, w - VP + 1))
                for vy in range(h - VP + 1):
                    for vx in range(w - VP + 1):
                        patch = gt[vy:vy+VP, vx:vx+VP]
                        ip = ig[vy:vy+VP, vx:vx+VP]
                        d = (ip != OCEAN) & (ip != MOUNTAIN)
                        if d.any():
                            e = -np.sum(np.maximum(patch, 1e-10) * np.log2(np.maximum(patch, 1e-10)), axis=2)
                            scores[vy, vx] = e[d].sum()
                return make_obs_uniform(gt, ig, nv, qp, scores, rng)
            return fn
        strategies[label] = make_fn()

    # 4. Adaptive sequential
    strategies["Sequential adaptive"] = lambda gt, ig, model, ho, sn, rng: \
        make_obs_adaptive_3phase(gt, ig, model, rng)

    # 5. Regime-first
    strategies["Regime-first"] = lambda gt, ig, model, ho, sn, rng: \
        make_obs_regime_first(gt, ig, model, rng)

    # 6. 4vp adaptive
    strategies["4vp x 2q + 2 adaptive"] = lambda gt, ig, model, ho, sn, rng: \
        make_obs_4vp_adaptive(gt, ig, model, rng)

    # 7. Hybrid: 3 hotspot + 1 pred_entropy viewport
    def make_hybrid(gt, ig, model, ho, sn, rng):
        hs = hotspot_cache[(ho, sn)]
        pos_h = pick_viewports(hs, 2, min_dist=8)
        es = get_pred_entropy_scores(base_preds[(ho, sn)], ig)
        # Suppress areas near hotspot viewports
        for vy, vx in pos_h:
            r0 = max(0, vy - VP + 1); r1 = min(es.shape[0], vy + 1)
            c0 = max(0, vx - VP + 1); c1 = min(es.shape[1], vx + 1)
            es[r0:r1, c0:c1] *= 0.2
        pos_e = pick_viewports(es, 1, min_dist=6)
        obs = []
        for vy, vx in pos_h:
            for _ in range(3):
                obs.append(sample_viewport(gt, ig, vy, vx, rng))
        for vy, vx in pos_e:
            for _ in range(4):
                obs.append(sample_viewport(gt, ig, vy, vx, rng))
        return obs  # 6 + 4 = 10
    strategies["Hybrid 2hot+1ent (3+3+4)"] = make_hybrid

    # 8. Seed-adaptive across seeds
    # This needs special handling since it pools across seeds
    # We'll implement it differently

    # 9. No-obs baseline
    strategies["No observations"] = lambda gt, ig, model, ho, sn, rng: []

    # 10. Vary tau for best placement
    # We test different tau values with the best structural strategy

    print("\nRunning strategy evaluations...")
    print(f"{'Strategy':<40s} {'wKL':>10s} {'Std':>10s} {'Time':>8s}")
    print("-" * 72)
    sys.stdout.flush()

    results = {}
    for name, fn in strategies.items():
        t1 = time.time()
        mean, std = eval_strategy(name, fn, n_trials=3)
        elapsed = time.time() - t1
        results[name] = (mean, std)
        print(f"{name:<40s} {mean:10.6f} {std:10.6f} {elapsed:7.1f}s")
        sys.stdout.flush()

    # Now test tau sweep for the best-so-far strategy
    print("\n--- Tau sweep for best placement strategy ---")
    # Find best non-oracle strategy
    non_oracle = {k: v for k, v in results.items() if "ORACLE" not in k and k != "No observations"}
    best_name = min(non_oracle, key=lambda k: non_oracle[k][0])
    best_fn = strategies[best_name]
    print(f"Best strategy so far: {best_name} (wKL={results[best_name][0]:.6f})")

    for tau_val in [20, 50, 75, 100, 150, 200, 300]:
        def eval_tau(name_t, obs_fn_t, tau_t, n_trials=3):
            trial_means = []
            for trial in range(n_trials):
                rng = np.random.RandomState(42 + trial * 1000)
                round_wkls = []
                for held_out in all_rns:
                    model = models[held_out]
                    seed_wkls = []
                    for sn in sorted(rounds[held_out].keys()):
                        gt = rounds[held_out][sn]["ground_truth"]
                        ig = rounds[held_out][sn]["initial_grid"]
                        obs = obs_fn_t(gt, ig, model, held_out, sn, rng)
                        pred = predict_with_obs(ig, obs, model, tau=tau_t, sigma=0.3)
                        seed_wkls.append(wkl_score(pred, gt, ig))
                    round_wkls.append(np.mean(seed_wkls))
                trial_means.append(np.mean(round_wkls))
            return np.mean(trial_means), np.std(trial_means)

        t1 = time.time()
        mean, std = eval_tau(f"tau={tau_val}", best_fn, tau_val)
        elapsed = time.time() - t1
        results[f"{best_name} tau={tau_val}"] = (mean, std)
        print(f"  tau={tau_val:<5d} wKL={mean:.6f} +/- {std:.6f} ({elapsed:.1f}s)")
        sys.stdout.flush()

    # Sigma sweep
    print("\n--- Sigma sweep ---")
    for sigma_val in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
        def eval_sigma(obs_fn_s, tau_s, sigma_s, n_trials=3):
            trial_means = []
            for trial in range(n_trials):
                rng = np.random.RandomState(42 + trial * 1000)
                round_wkls = []
                for held_out in all_rns:
                    model = models[held_out]
                    seed_wkls = []
                    for sn in sorted(rounds[held_out].keys()):
                        gt = rounds[held_out][sn]["ground_truth"]
                        ig = rounds[held_out][sn]["initial_grid"]
                        obs = obs_fn_s(gt, ig, model, held_out, sn, rng)
                        pred = predict_with_obs(ig, obs, model, tau=tau_s, sigma=sigma_s)
                        seed_wkls.append(wkl_score(pred, gt, ig))
                    round_wkls.append(np.mean(seed_wkls))
                trial_means.append(np.mean(round_wkls))
            return np.mean(trial_means), np.std(trial_means)

        t1 = time.time()
        mean, std = eval_sigma(best_fn, 100.0, sigma_val)
        elapsed = time.time() - t1
        results[f"{best_name} sigma={sigma_val}"] = (mean, std)
        print(f"  sigma={sigma_val:<5.1f} wKL={mean:.6f} +/- {std:.6f} ({elapsed:.1f}s)")
        sys.stdout.flush()

    # Final ranking
    print("\n" + "=" * 72)
    print("FINAL RANKING (lower wKL is better)")
    print("=" * 72)
    ranked = sorted(results.items(), key=lambda x: x[1][0])
    for rank, (name, (mean, std)) in enumerate(ranked, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"  {rank:2d}. {name:<50s} {mean:.6f} +/- {std:.6f}{marker}")

    best_name, (best_mean, best_std) = ranked[0]
    print(f"\nBest: {best_name} = {best_mean:.6f}")
    print(f"Baseline (3vp x 3q hotspot, tau=20): 0.028")
    if best_mean < 0.028:
        print(f"Improvement: {(0.028 - best_mean)/0.028*100:.1f}%")

    total = time.time() - t0
    print(f"Total time: {total:.0f}s")

    return results, ranked


if __name__ == "__main__":
    results, ranked = main()

    # Write report
    report_path = BASE_DIR / "query_optimization.md"
    with open(report_path, "w") as f:
        f.write("# Query Allocation Optimization Results\n\n")
        f.write("## Setup\n")
        f.write("- 50 queries total, 5 seeds per round, 10 queries per seed\n")
        f.write("- Each query: 15x15 viewport of 40x40 grid, stochastic simulation\n")
        f.write("- Base predictor: replay_boosted_predictor (residual_weight=0.5)\n")
        f.write("- Default post-processing: tau=100 overlay, sigma=0.3 smoothing\n")
        f.write("- Evaluation: Leave-one-round-out CV, 3 random trials averaged\n\n")

        f.write("## Results (ranked by mean wKL, lower is better)\n\n")
        f.write("| Rank | Strategy | wKL | Std |\n")
        f.write("|------|----------|-----|-----|\n")
        for rank, (name, (mean, std)) in enumerate(ranked, 1):
            f.write(f"| {rank} | {name} | {mean:.6f} | {std:.6f} |\n")

        best_name, (best_mean, best_std) = ranked[0]
        f.write(f"\n## Best Strategy\n\n")
        f.write(f"**{best_name}** with wKL = {best_mean:.6f} +/- {best_std:.6f}\n\n")

        f.write("## Strategy Descriptions\n\n")
        f.write("### Viewport Placement Methods\n")
        f.write("- **hotspot**: Place viewports near settlements (highest civ density)\n")
        f.write("- **pred_ent**: Place where prediction entropy is highest (most uncertain cells)\n")
        f.write("- **gt_ent ORACLE**: Place where ground-truth entropy is highest (upper bound, not achievable in practice)\n\n")

        f.write("### Coverage vs Depth\n")
        f.write("- NvpxMq = N viewport positions, M queries each (N*M = 10 per seed)\n")
        f.write("- More viewports = better coverage, fewer repeated samples\n")
        f.write("- More queries per viewport = better distribution estimation at each location\n\n")

        f.write("### Advanced Strategies\n")
        f.write("- **Sequential adaptive**: 3-phase approach using early results to guide later placement\n")
        f.write("- **Regime-first**: 2 queries for regime detection, 8 for estimation\n")
        f.write("- **4vp adaptive**: 4 viewports x 2q, then 2 extra at most uncertain viewport\n")
        f.write("- **Hybrid**: Mix hotspot + entropy placement\n\n")

        f.write("## Key Findings\n\n")

        # Extract key findings from results
        non_oracle = [(n, v) for n, v in ranked if "ORACLE" not in n and n != "No observations"
                       and "tau=" not in n and "sigma=" not in n]
        if non_oracle:
            f.write(f"1. Best achievable strategy: **{non_oracle[0][0]}** (wKL={non_oracle[0][1][0]:.6f})\n")

        oracle = [(n, v) for n, v in ranked if "ORACLE" in n]
        if oracle:
            f.write(f"2. Oracle upper bound: {oracle[0][0]} (wKL={oracle[0][1][0]:.6f})\n")

        no_obs = [(n, v) for n, v in ranked if n == "No observations"]
        if no_obs:
            f.write(f"3. No observations baseline: wKL={no_obs[0][1][0]:.6f}\n")
            if non_oracle:
                imp = (no_obs[0][1][0] - non_oracle[0][1][0]) / no_obs[0][1][0] * 100
                f.write(f"4. Observation value: {imp:.1f}% improvement over no-obs\n")

        f.write(f"\nCurrent production baseline: 0.028 wKL\n")
        if non_oracle:
            delta = (0.028 - non_oracle[0][1][0]) / 0.028 * 100
            if delta > 0:
                f.write(f"Best strategy improvement over baseline: {delta:.1f}%\n")
            else:
                f.write(f"No improvement over baseline found (delta: {delta:.1f}%)\n")

    print(f"\nReport written to {report_path}")

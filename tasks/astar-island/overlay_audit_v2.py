#!/usr/bin/env python3
"""Supplementary audit: test with mixed viewports (some overlap) to validate min_samples,
and fine-tune tau in the 20-50 range."""

import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, 'tasks/astar-island')
import regime_predictor as rp

GT_DIR = Path("tasks/astar-island/ground_truth")
N_CLASSES = 6
FLOOR = 0.005
VIEWPORT = 15
TEST_ROUNDS = [9, 13, 14, 15]
N_OBS = 10
N_TRIALS = 5


def load_ground_truth():
    rounds = {}
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        parts = f.stem.split("_")
        rn = int(parts[0].replace("round", ""))
        sn = int(parts[1].replace("seed", ""))
        with open(f) as fh:
            data = json.load(fh)
        if "ground_truth" not in data or "initial_grid" not in data:
            continue
        rounds.setdefault(rn, {})[sn] = {
            "initial_grid": np.array(data["initial_grid"], dtype=np.int32),
            "ground_truth": np.array(data["ground_truth"], dtype=np.float64),
        }
    return rounds


def entropy(p):
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(p_safe * np.log2(p_safe), axis=2)


def kl_divergence(p, q):
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=2)


def score_prediction(gt, pred):
    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)
    kl = kl_divergence(gt, pred)
    H = entropy(gt)
    wkl = H * kl
    dynamic = H > 0.01
    if not dynamic.any():
        return 0.0
    return float(wkl[dynamic].mean())


def find_top_viewports(gt, ig, n=3):
    """Find top N non-overlapping viewports by entropy."""
    h, w, _ = gt.shape
    H = entropy(gt)
    ig_arr = np.array(ig)
    H[(ig_arr == 10) | (ig_arr == 5)] = 0.0

    max_vy = h - VIEWPORT
    max_vx = w - VIEWPORT
    if max_vy < 0 or max_vx < 0:
        return [(0, 0)]

    H_pad = np.zeros((h + 1, w + 1))
    H_pad[1:, 1:] = np.cumsum(np.cumsum(H, axis=0), axis=1)

    results = []
    used = set()
    for _ in range(n):
        best_score = -1.0
        best_pos = (0, 0)
        for vy in range(max_vy + 1):
            for vx in range(max_vx + 1):
                too_close = any(abs(vy - uvy) < 8 and abs(vx - uvx) < 8 for uvy, uvx in used)
                if too_close:
                    continue
                s = (H_pad[vy + VIEWPORT, vx + VIEWPORT]
                     - H_pad[vy, vx + VIEWPORT]
                     - H_pad[vy + VIEWPORT, vx]
                     + H_pad[vy, vx])
                if s > best_score:
                    best_score = s
                    best_pos = (vy, vx)
        results.append(best_pos)
        used.add(best_pos)
    return results


def simulate_mixed_observations(gt, ig, n_obs, rng_seed=42):
    """Simulate observations across multiple viewports (some overlap, varying counts per cell)."""
    rng = np.random.RandomState(rng_seed)
    h, w, _ = gt.shape
    viewports = find_top_viewports(gt, ig, n=3)

    observations = []
    # Distribute: 5 on hottest, 3 on second, 2 on third
    allocation = [5, 3, 2]
    for vp_idx, (vy, vx) in enumerate(viewports):
        n = allocation[vp_idx] if vp_idx < len(allocation) else 0
        for _ in range(n):
            grid = []
            for dy in range(VIEWPORT):
                row = []
                for dx in range(VIEWPORT):
                    y, x = vy + dy, vx + dx
                    if 0 <= y < h and 0 <= x < w:
                        row.append(int(rng.choice(N_CLASSES, p=gt[y, x])))
                    else:
                        row.append(0)
                grid.append(row)
            observations.append({"grid": grid, "viewport_x": int(vx), "viewport_y": int(vy)})
    return observations


def cell_code_to_class(cell):
    if cell in (0, 10, 11): return 0
    if cell == 1: return 1
    if cell == 2: return 2
    if cell == 3: return 3
    if cell == 4: return 4
    if cell == 5: return 5
    return 0


def count_observations(pred, observations):
    h, w, nc = pred.shape
    counts = np.zeros_like(pred)
    obs_count = np.zeros((h, w), dtype=np.float64)
    for obs in observations:
        grid = obs.get("grid", [])
        vx = int(obs.get("viewport_x", 0))
        vy = int(obs.get("viewport_y", 0))
        for dy, row in enumerate(grid):
            for dx, cell in enumerate(row):
                y, x = vy + dy, vx + dx
                if 0 <= y < h and 0 <= x < w:
                    counts[y, x, cell_code_to_class(int(cell))] += 1.0
                    obs_count[y, x] += 1.0
    return counts, obs_count


def overlay_dirichlet(pred, ig, observations, tau, min_samples=3):
    if not observations:
        return pred
    counts, obs_count = count_observations(pred, observations)
    observed = obs_count >= min_samples
    alpha = tau * pred + counts
    posterior = alpha / alpha.sum(axis=2, keepdims=True)
    out = pred.copy()
    out[observed] = posterior[observed]
    out = np.maximum(out, 1e-6)
    out /= out.sum(axis=2, keepdims=True)
    return out


def overlay_entropy_weighted(pred, ig, observations, tau_base=10, min_samples=3):
    if not observations:
        return pred
    counts, obs_count = count_observations(pred, observations)
    observed = obs_count >= min_samples
    p_safe = np.maximum(pred, 1e-10)
    cell_entropy = -np.sum(p_safe * np.log(p_safe), axis=2)
    max_entropy = np.log(N_CLASSES)
    norm_entropy = cell_entropy / max_entropy
    tau = tau_base * (1.0 - 0.8 * norm_entropy)
    tau = np.maximum(tau, 0.5)
    alpha = tau[:, :, np.newaxis] * pred + counts
    posterior = alpha / alpha.sum(axis=2, keepdims=True)
    out = pred.copy()
    out[observed] = posterior[observed]
    out = np.maximum(out, 1e-6)
    out /= out.sum(axis=2, keepdims=True)
    return out


def run():
    print("Loading ground truth...")
    all_rounds = load_ground_truth()

    results = {}
    for test_round in TEST_ROUNDS:
        if test_round not in all_rounds:
            continue
        train_rounds = {rn: rd for rn, rd in all_rounds.items() if rn != test_round}
        model = rp.build_model_from_data(train_rounds)

        for sn, sd in sorted(all_rounds[test_round].items()):
            key = f"R{test_round}S{sn}"
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]

            trial_scores = defaultdict(list)
            for trial in range(N_TRIALS):
                obs = simulate_mixed_observations(gt, ig, N_OBS, rng_seed=42 + trial)
                pred_with_obs = rp.predict_with_model(ig, model, observations=obs)

                trial_scores["no_overlay"].append(score_prediction(gt, pred_with_obs))

                # Fine-tune tau in 20-50 range
                for tau in [15, 20, 25, 30, 35, 40, 50, 75, 100]:
                    s = score_prediction(gt, overlay_dirichlet(pred_with_obs.copy(), ig, obs, tau=tau, min_samples=3))
                    trial_scores[f"dirichlet_tau{tau}_min3"].append(s)

                # Min samples with mixed viewports (now matters!)
                for min_s in [1, 2, 3, 5, 8]:
                    s = score_prediction(gt, overlay_dirichlet(pred_with_obs.copy(), ig, obs, tau=30, min_samples=min_s))
                    trial_scores[f"dirichlet_tau30_min{min_s}"].append(s)

                # Entropy weighted at optimal-ish tau
                for tau_base in [20, 30, 40, 50]:
                    s = score_prediction(gt, overlay_entropy_weighted(pred_with_obs.copy(), ig, obs, tau_base=tau_base, min_samples=3))
                    trial_scores[f"entropy_weighted_tau{tau_base}_min3"].append(s)

            results[key] = {k: float(np.mean(v)) for k, v in trial_scores.items()}
            print(f"  {key}: no_overlay={results[key]['no_overlay']:.6f}")

    # Summary
    all_methods = set()
    for v in results.values():
        all_methods.update(v.keys())

    summary = {}
    for method in sorted(all_methods):
        vals = [results[k].get(method, None) for k in sorted(results.keys())]
        vals = [v for v in vals if v is not None]
        if vals:
            summary[method] = {"mean_wkl": float(np.mean(vals)), "std_wkl": float(np.std(vals))}

    print("\n" + "=" * 70)
    print("MIXED VIEWPORTS AUDIT (5+3+2 across 3 viewports)")
    print("=" * 70)
    print(f"{'Method':<45} {'Mean wKL':>10} {'Std':>10}")
    print("-" * 70)
    for method in sorted(summary.keys(), key=lambda m: summary[m]["mean_wkl"]):
        s = summary[method]
        print(f"{method:<45} {s['mean_wkl']:>10.6f} {s['std_wkl']:>10.6f}")

    return summary


if __name__ == "__main__":
    summary = run()

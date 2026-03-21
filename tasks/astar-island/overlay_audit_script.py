#!/usr/bin/env python3
"""Audit observation overlay strategies against ground truth.

Tests 5 variations across rounds 9, 13, 14, 15 with 10 blitz observations
per seed on the hottest viewport, sampling from GT distributions.
"""

import json
import sys
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, 'tasks/astar-island')
import regime_predictor as rp

GT_DIR = Path("tasks/astar-island/ground_truth")
N_CLASSES = 6
FLOOR = 0.005
VIEWPORT = 15
TEST_ROUNDS = [9, 13, 14, 15]
N_OBS = 10  # blitz observations per seed
N_TRIALS = 5  # average over multiple RNG seeds for stability


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


def find_hottest_viewport(gt, ig):
    """Find the viewport position with highest entropy sum."""
    h, w, _ = gt.shape
    H = entropy(gt)
    # Zero out static cells
    ig_arr = np.array(ig)
    H[(ig_arr == 10) | (ig_arr == 5)] = 0.0

    max_vy = h - VIEWPORT
    max_vx = w - VIEWPORT
    if max_vy < 0 or max_vx < 0:
        return 0, 0

    H_pad = np.zeros((h + 1, w + 1))
    H_pad[1:, 1:] = np.cumsum(np.cumsum(H, axis=0), axis=1)

    best_score = -1.0
    best_pos = (0, 0)
    for vy in range(max_vy + 1):
        for vx in range(max_vx + 1):
            s = (H_pad[vy + VIEWPORT, vx + VIEWPORT]
                 - H_pad[vy, vx + VIEWPORT]
                 - H_pad[vy + VIEWPORT, vx]
                 + H_pad[vy, vx])
            if s > best_score:
                best_score = s
                best_pos = (vy, vx)
    return best_pos


def simulate_observations(gt, ig, n_obs, rng_seed=42):
    """Simulate n_obs observations on the hottest viewport, sampling from GT."""
    rng = np.random.RandomState(rng_seed)
    h, w, _ = gt.shape
    vy, vx = find_hottest_viewport(gt, ig)

    observations = []
    for _ in range(n_obs):
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
        observations.append({
            "grid": grid,
            "viewport_x": int(vx),
            "viewport_y": int(vy),
        })
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
    """Build observation count arrays."""
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


# =========================================================================
# VARIATION 1: Standard Dirichlet overlay with different tau values
# =========================================================================
def overlay_dirichlet(pred, ig, observations, tau, min_samples=3):
    """Dirichlet posterior: alpha = tau * prior + counts, on cells with >= min_samples."""
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


# =========================================================================
# VARIATION 2: Different min_samples thresholds (with fixed tau)
# =========================================================================
# (Uses overlay_dirichlet with different min_samples parameter)


# =========================================================================
# VARIATION 3: Pure multinomial MLE on observed cells (no prior blend)
# =========================================================================
def overlay_mle(pred, ig, observations, min_samples=3):
    """Replace observed cells with pure MLE (counts / total). No prior."""
    if not observations:
        return pred
    counts, obs_count = count_observations(pred, observations)
    observed = obs_count >= min_samples

    mle = counts.copy()
    total = obs_count[:, :, np.newaxis]
    # Avoid division by zero
    safe_total = np.maximum(total, 1.0)
    mle = mle / safe_total

    out = pred.copy()
    out[observed] = mle[observed]
    out = np.maximum(out, 1e-6)
    out /= out.sum(axis=2, keepdims=True)
    return out


# =========================================================================
# VARIATION 4: Entropy-weighted overlay
# =========================================================================
def overlay_entropy_weighted(pred, ig, observations, tau_base=10, min_samples=3):
    """Weight observation updates by cell entropy - high entropy cells trust observations more."""
    if not observations:
        return pred
    counts, obs_count = count_observations(pred, observations)
    observed = obs_count >= min_samples

    # Cell entropy from prior
    p_safe = np.maximum(pred, 1e-10)
    cell_entropy = -np.sum(p_safe * np.log(p_safe), axis=2)
    max_entropy = np.log(N_CLASSES)
    norm_entropy = cell_entropy / max_entropy  # 0 to 1

    # Scale tau inversely with entropy: high entropy -> low tau -> more data weight
    # Low entropy -> high tau -> trust prior more
    tau = tau_base * (1.0 - 0.8 * norm_entropy)  # tau ranges from 0.2*base to base
    tau = np.maximum(tau, 0.5)

    alpha = tau[:, :, np.newaxis] * pred + counts
    posterior = alpha / alpha.sum(axis=2, keepdims=True)

    out = pred.copy()
    out[observed] = posterior[observed]
    out = np.maximum(out, 1e-6)
    out /= out.sum(axis=2, keepdims=True)
    return out


# =========================================================================
# VARIATION 5: Ratio correction factor
# =========================================================================
def overlay_ratio_correction(pred, ig, observations, strength=0.5, min_samples=3):
    """Use ratio of observed class freq to regime prior as correction factor.

    correction = (observed_freq / prior_prob)
    adjusted = prior * correction^strength, then renormalize
    """
    if not observations:
        return pred
    counts, obs_count = count_observations(pred, observations)
    observed = obs_count >= min_samples

    # MLE from observations
    safe_total = np.maximum(obs_count[:, :, np.newaxis], 1.0)
    obs_freq = counts / safe_total

    # Ratio correction
    safe_prior = np.maximum(pred, 1e-6)
    ratio = obs_freq / safe_prior  # > 1 means observed more than expected

    # Apply power-damped correction
    correction = np.power(np.maximum(ratio, 1e-6), strength)

    adjusted = pred * correction
    adjusted = np.maximum(adjusted, 1e-6)
    adjusted /= adjusted.sum(axis=2, keepdims=True)

    out = pred.copy()
    out[observed] = adjusted[observed]
    out = np.maximum(out, 1e-6)
    out /= out.sum(axis=2, keepdims=True)
    return out


def get_baseline_prediction(ig, train_rounds):
    """Get regime predictor baseline (no observations, trained on train_rounds)."""
    model = rp.build_model_from_data(train_rounds)
    return rp.predict_with_model(ig, model)


def run_audit():
    print("Loading ground truth...")
    all_rounds = load_ground_truth()
    print(f"Loaded {len(all_rounds)} rounds")

    results = {}

    for test_round in TEST_ROUNDS:
        if test_round not in all_rounds:
            print(f"Round {test_round} not in ground truth, skipping")
            continue

        # Leave-one-out: train on everything except test round
        train_rounds = {rn: rd for rn, rd in all_rounds.items() if rn != test_round}
        model = rp.build_model_from_data(train_rounds)

        for sn, sd in sorted(all_rounds[test_round].items()):
            key = f"R{test_round}S{sn}"
            ig = sd["initial_grid"]
            gt = sd["ground_truth"]

            # Get baseline (no overlay)
            pred_base = rp.predict_with_model(ig, model)
            base_score = score_prediction(gt, pred_base)

            # Also get the prediction WITH observations for regime detection
            # (the regime predictor uses obs for regime classification)
            trial_scores = defaultdict(list)

            for trial in range(N_TRIALS):
                obs = simulate_observations(gt, ig, N_OBS, rng_seed=42 + trial)

                # Prediction with observations (regime detection only, no overlay)
                pred_with_obs = rp.predict_with_model(ig, model, observations=obs)
                no_overlay_score = score_prediction(gt, pred_with_obs)
                trial_scores["no_overlay"].append(no_overlay_score)

                # VARIATION 1: Different tau values
                for tau in [5, 10, 20, 50]:
                    for min_s in [3]:  # default min_samples
                        s = score_prediction(gt, overlay_dirichlet(pred_with_obs.copy(), ig, obs, tau=tau, min_samples=min_s))
                        trial_scores[f"dirichlet_tau{tau}_min{min_s}"].append(s)

                # VARIATION 2: Different min_samples (fixed tau=10)
                for min_s in [2, 3, 5]:
                    s = score_prediction(gt, overlay_dirichlet(pred_with_obs.copy(), ig, obs, tau=10, min_samples=min_s))
                    trial_scores[f"dirichlet_tau10_min{min_s}"].append(s)

                # VARIATION 3: Pure MLE
                for min_s in [3, 5]:
                    s = score_prediction(gt, overlay_mle(pred_with_obs.copy(), ig, obs, min_samples=min_s))
                    trial_scores[f"mle_min{min_s}"].append(s)

                # VARIATION 4: Entropy-weighted
                for tau_base in [5, 10, 20]:
                    s = score_prediction(gt, overlay_entropy_weighted(pred_with_obs.copy(), ig, obs, tau_base=tau_base, min_samples=3))
                    trial_scores[f"entropy_weighted_tau{tau_base}_min3"].append(s)

                # VARIATION 5: Ratio correction
                for strength in [0.3, 0.5, 0.7, 1.0]:
                    s = score_prediction(gt, overlay_ratio_correction(pred_with_obs.copy(), ig, obs, strength=strength, min_samples=3))
                    trial_scores[f"ratio_s{strength}_min3"].append(s)

            results[key] = {
                "baseline_no_obs": base_score,
                **{k: float(np.mean(v)) for k, v in trial_scores.items()},
            }
            print(f"  {key}: baseline={base_score:.6f}  no_overlay={np.mean(trial_scores['no_overlay']):.6f}")

    # Compute mean across all test seeds
    all_methods = set()
    for v in results.values():
        all_methods.update(v.keys())

    summary = {}
    for method in sorted(all_methods):
        vals = [results[k].get(method, None) for k in sorted(results.keys())]
        vals = [v for v in vals if v is not None]
        if vals:
            summary[method] = {
                "mean_wkl": float(np.mean(vals)),
                "std_wkl": float(np.std(vals)),
                "min_wkl": float(np.min(vals)),
                "max_wkl": float(np.max(vals)),
                "n": len(vals),
            }

    print("\n" + "=" * 80)
    print("SUMMARY: Mean weighted KL across all test rounds/seeds")
    print("=" * 80)
    print(f"{'Method':<40} {'Mean wKL':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("-" * 80)
    for method in sorted(summary.keys(), key=lambda m: summary[m]["mean_wkl"]):
        s = summary[method]
        marker = " <-- BEST" if method == sorted(summary.keys(), key=lambda m: summary[m]["mean_wkl"])[0] else ""
        print(f"{method:<40} {s['mean_wkl']:>10.6f} {s['std_wkl']:>10.6f} {s['min_wkl']:>10.6f} {s['max_wkl']:>10.6f}{marker}")

    return results, summary


if __name__ == "__main__":
    results, summary = run_audit()

    # Save raw results
    output_path = Path("tasks/astar-island/overlay_audit_results.json")
    serializable = {
        "per_seed": results,
        "summary": summary,
    }
    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to {output_path}")

#!/usr/bin/env python3
"""Experiments for template predictor: observation budget, blending, floor.

Tests:
1. Template predictor with different numbers of simulated observations
2. Blending template_predictor with neighborhood_predictor
3. Different floor values
"""

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from benchmark import load_ground_truth, kl_divergence, entropy

GT_DIR = Path(__file__).parent / "ground_truth"
VIEWPORT_SIZE = 15
RNG = np.random.RandomState(42)


def simulate_observations_from_gt(ground_truth, initial_grid, n_viewports, strategy="hotspot"):
    """Simulate observations by sampling from ground truth distribution.

    Returns list of observation dicts compatible with template_predictor.
    """
    gt = np.array(ground_truth)
    ig = np.array(initial_grid)
    h, w, _ = gt.shape
    observations = []

    if strategy == "hotspot":
        # Pick viewports with highest expected information (entropy)
        from scipy.ndimage import distance_transform_cdt
        civ = (ig == 1) | (ig == 2)
        if civ.any():
            dist = distance_transform_cdt(~civ, metric="taxicab")
        else:
            dist = np.full((h, w), 20)

        # Score each viewport position
        scores = np.zeros((h - VIEWPORT_SIZE + 1, w - VIEWPORT_SIZE + 1))
        for vy in range(h - VIEWPORT_SIZE + 1):
            for vx in range(w - VIEWPORT_SIZE + 1):
                patch = gt[vy:vy+VIEWPORT_SIZE, vx:vx+VIEWPORT_SIZE]
                ig_patch = ig[vy:vy+VIEWPORT_SIZE, vx:vx+VIEWPORT_SIZE]
                dynamic = (ig_patch != 10) & (ig_patch != 5)
                if dynamic.any():
                    ent = -np.sum(np.maximum(patch, 1e-10) * np.log2(np.maximum(patch, 1e-10)), axis=2)
                    scores[vy, vx] = ent[dynamic].sum()

        # Pick top viewports (with some diversity)
        used_positions = set()
        for _ in range(n_viewports):
            best_score = -1
            best_pos = (0, 0)
            flat = scores.ravel()
            sorted_idx = np.argsort(-flat)
            for idx in sorted_idx:
                vy = idx // scores.shape[1]
                vx = idx % scores.shape[1]
                # Require minimum distance from previous viewports
                too_close = False
                for uvy, uvx in used_positions:
                    if abs(vy - uvy) < 8 and abs(vx - uvx) < 8:
                        too_close = True
                        break
                if not too_close:
                    best_pos = (vy, vx)
                    break

            vy, vx = best_pos
            used_positions.add((vy, vx))

            # Sample a cell outcome from ground truth distribution
            grid_sample = []
            for dy in range(VIEWPORT_SIZE):
                row = []
                for dx in range(VIEWPORT_SIZE):
                    y, x = vy + dy, vx + dx
                    if 0 <= y < h and 0 <= x < w:
                        probs = gt[y, x]
                        cls = RNG.choice(6, p=probs)
                        # Map class back to cell code
                        code = [0, 1, 2, 3, 4, 5][cls]
                        row.append(code)
                    else:
                        row.append(0)
                grid_sample.append(row)

            observations.append({
                "grid": grid_sample,
                "viewport_x": int(vx),
                "viewport_y": int(vy),
            })

    return observations


def experiment_observation_budget():
    """Test template_predictor with different observation counts."""
    import template_predictor as tp
    tp.get_model()
    tp._GRID_FINGERPRINTS.clear()  # Force unknown-grid path

    rounds = load_ground_truth()
    budgets = [0, 5, 10, 20, 30, 50]
    results = {}

    for n_obs in budgets:
        all_wkl = []
        all_kl = []

        for rn in sorted(rounds.keys()):
            for sn in sorted(rounds[rn].keys()):
                sd = rounds[rn][sn]
                gt = sd["ground_truth"]
                ig = sd["initial_grid"]

                if n_obs > 0:
                    obs = simulate_observations_from_gt(gt, ig, n_obs)
                else:
                    obs = None

                pred = tp.predict(ig, observations=obs)
                pred = np.maximum(pred, 0.01)
                pred /= pred.sum(axis=2, keepdims=True)

                kl = kl_divergence(gt, pred)
                H = entropy(gt)
                wkl = H * kl
                dynamic = H > 0.01
                if dynamic.any():
                    all_kl.append(float(kl[dynamic].mean()))
                    all_wkl.append(float(wkl[dynamic].mean()))

        mean_wkl = np.mean(all_wkl)
        mean_kl = np.mean(all_kl)
        results[n_obs] = {"wkl": float(mean_wkl), "kl": float(mean_kl)}
        print(f"  {n_obs:3d} viewports: wKL={mean_wkl:.6f}  KL={mean_kl:.6f}")

    return results


def experiment_blend_weights():
    """Test blending template_predictor with neighborhood_predictor."""
    import template_predictor as tp
    import neighborhood_predictor as nb

    tp.get_model()
    tp._GRID_FINGERPRINTS.clear()

    rounds = load_ground_truth()
    weights = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    results = {}

    # Pre-compute all predictions
    nb_preds = {}
    tp_preds = {}
    for rn in sorted(rounds.keys()):
        for sn in sorted(rounds[rn].keys()):
            key = (rn, sn)
            ig = rounds[rn][sn]["initial_grid"]
            nb_preds[key] = nb.predict(ig)
            tp_preds[key] = tp.predict(ig)

    for w_tp in weights:
        all_wkl = []
        all_kl = []

        for rn in sorted(rounds.keys()):
            for sn in sorted(rounds[rn].keys()):
                key = (rn, sn)
                gt = rounds[rn][sn]["ground_truth"]

                blend = w_tp * tp_preds[key] + (1.0 - w_tp) * nb_preds[key]
                blend = np.maximum(blend, 0.01)
                blend /= blend.sum(axis=2, keepdims=True)

                kl = kl_divergence(gt, blend)
                H = entropy(gt)
                wkl = H * kl
                dynamic = H > 0.01
                if dynamic.any():
                    all_kl.append(float(kl[dynamic].mean()))
                    all_wkl.append(float(wkl[dynamic].mean()))

        mean_wkl = np.mean(all_wkl)
        mean_kl = np.mean(all_kl)
        results[w_tp] = {"wkl": float(mean_wkl), "kl": float(mean_kl)}
        print(f"  tp_weight={w_tp:.1f}: wKL={mean_wkl:.6f}  KL={mean_kl:.6f}")

    return results


def experiment_floor_values():
    """Test different floor values on template_predictor."""
    import template_predictor as tp
    tp.get_model()
    tp._GRID_FINGERPRINTS.clear()

    rounds = load_ground_truth()
    floors = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05]
    results = {}

    # Pre-compute raw predictions
    raw_preds = {}
    for rn in sorted(rounds.keys()):
        for sn in sorted(rounds[rn].keys()):
            key = (rn, sn)
            ig = rounds[rn][sn]["initial_grid"]
            raw_preds[key] = tp.predict(ig)

    for floor in floors:
        all_wkl = []
        all_kl = []

        for rn in sorted(rounds.keys()):
            for sn in sorted(rounds[rn].keys()):
                key = (rn, sn)
                gt = rounds[rn][sn]["ground_truth"]
                pred = raw_preds[key].copy()

                pred = np.maximum(pred, floor)
                pred /= pred.sum(axis=2, keepdims=True)

                kl = kl_divergence(gt, pred)
                H = entropy(gt)
                wkl = H * kl
                dynamic = H > 0.01
                if dynamic.any():
                    all_kl.append(float(kl[dynamic].mean()))
                    all_wkl.append(float(wkl[dynamic].mean()))

        mean_wkl = np.mean(all_wkl)
        mean_kl = np.mean(all_kl)
        results[floor] = {"wkl": float(mean_wkl), "kl": float(mean_kl)}
        print(f"  floor={floor:.3f}: wKL={mean_wkl:.6f}  KL={mean_kl:.6f}")

    return results


def main():
    target = sys.argv[1] if len(sys.argv) > 1 else "all"

    all_results = {}

    if target in ("budget", "all"):
        print("\n=== EXPERIMENT 1: Observation Budget ===")
        t0 = time.time()
        all_results["budget"] = experiment_observation_budget()
        print(f"  Time: {time.time() - t0:.1f}s")

    if target in ("blend", "all"):
        print("\n=== EXPERIMENT 2: Blend Weights ===")
        t0 = time.time()
        all_results["blend"] = experiment_blend_weights()
        print(f"  Time: {time.time() - t0:.1f}s")

    if target in ("floor", "all"):
        print("\n=== EXPERIMENT 3: Floor Values ===")
        t0 = time.time()
        all_results["floor"] = experiment_floor_values()
        print(f"  Time: {time.time() - t0:.1f}s")

    # Save results
    out_path = Path(__file__).parent / "benchmark_results" / "experiments.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

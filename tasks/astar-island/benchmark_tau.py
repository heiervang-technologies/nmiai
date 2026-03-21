#!/usr/bin/env python3
"""Test tau-based observation overlay on top of regime predictor.

Applies Dirichlet posterior update: posterior = counts + tau * prior
Tests tau=2, 5, 10, 20 and no-overlay baseline.
Uses leave-one-round-out CV with simulated observations.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from benchmark import load_ground_truth, kl_divergence, entropy
from benchmark_experiments import simulate_observations_from_gt

GT_DIR = Path(__file__).parent / "ground_truth"
FLOOR = 0.005


def apply_obs_overlay(pred, ig, observations, tau):
    """Apply Dirichlet observation overlay with given tau."""
    if not observations:
        return pred

    h, w, nc = pred.shape
    counts = np.zeros_like(pred)

    cell_code_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

    for obs in observations:
        grid = obs.get("grid", [])
        vx = int(obs.get("viewport_x", 0))
        vy = int(obs.get("viewport_y", 0))
        for dy, row in enumerate(grid):
            for dx, cell in enumerate(row):
                y = vy + dy
                x = vx + dx
                if 0 <= y < h and 0 <= x < w:
                    cls = cell_code_map.get(int(cell), 0)
                    counts[y, x, cls] += 1.0

    obs_count = counts.sum(axis=2)
    observed = obs_count > 0

    # Dirichlet posterior: alpha = tau * prior + counts
    alpha = tau * pred + counts
    posterior = alpha / alpha.sum(axis=2, keepdims=True)

    # Only update observed cells
    out = pred.copy()
    out[observed] = posterior[observed]

    out = np.maximum(out, FLOOR)
    out /= out.sum(axis=2, keepdims=True)
    return out


def run_experiment():
    import regime_predictor as rp

    rounds = load_ground_truth()
    round_nums = sorted(rounds.keys())

    tau_values = [0, 2, 5, 10, 20, 50]
    # tau=0 means no overlay (just regime detection from obs, no local update)

    print(f"Rounds available: {round_nums}")
    print(f"Testing tau values: {tau_values}\n")

    all_results = {tau: [] for tau in tau_values}
    per_round = {tau: {} for tau in tau_values}

    for held_out in round_nums:
        # Hide held-out files
        hidden = []
        for f in GT_DIR.glob(f"round{held_out}_seed*.json"):
            tmp = f.with_suffix(".json.hidden")
            shutil.move(str(f), str(tmp))
            hidden.append((tmp, f))

        rp._MODEL = None
        try:
            model = rp.get_model()
        except Exception as e:
            for tmp, orig in hidden:
                shutil.move(str(tmp), str(orig))
            print(f"  R{held_out}: SKIP ({e})")
            continue

        for tau in tau_values:
            wkls = []
            for sn in sorted(rounds[held_out].keys()):
                sd = rounds[held_out][sn]
                ig = sd["initial_grid"]
                gt = sd["ground_truth"]

                obs = simulate_observations_from_gt(gt, ig, 5)

                # Get regime prediction (always uses observations for regime detection)
                pred = rp.predict(ig, observations=obs)

                # Optionally apply tau overlay
                if tau > 0:
                    pred = apply_obs_overlay(pred, ig, obs, tau)

                kl = kl_divergence(gt, pred)
                H = entropy(gt)
                wkl = H * kl
                dynamic = H > 0.01
                if dynamic.any():
                    wkls.append(float(wkl[dynamic].mean()))

            mean_wkl = np.mean(wkls)
            all_results[tau].append(mean_wkl)
            per_round[tau][held_out] = mean_wkl

        # Restore
        for tmp, orig in hidden:
            shutil.move(str(tmp), str(orig))
        rp._MODEL = None

        # Print this round's results
        line = f"  R{held_out:2d}:"
        for tau in tau_values:
            line += f"  tau={tau:2d}→{per_round[tau][held_out]:.4f}"
        print(line)

    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY (Leave-One-Round-Out CV, 5 simulated observations)")
    print(f"{'='*60}")
    for tau in tau_values:
        mean = np.mean(all_results[tau])
        label = "no overlay" if tau == 0 else f"tau={tau}"
        print(f"  {label:12s}: mean wKL = {mean:.6f}")

    # Best/worst rounds
    print(f"\n{'='*60}")
    print(f"BEST/WORST ROUNDS (tau=0, regime detection only)")
    print(f"{'='*60}")
    sorted_rounds = sorted(per_round[0].items(), key=lambda x: x[1])
    print("  BEST:")
    for rn, wkl in sorted_rounds[:5]:
        print(f"    R{rn}: wKL={wkl:.6f}")
    print("  WORST:")
    for rn, wkl in sorted_rounds[-5:]:
        print(f"    R{rn}: wKL={wkl:.6f}")

    return {"tau_results": {str(t): float(np.mean(v)) for t, v in all_results.items()},
            "per_round": {str(t): {str(r): float(w) for r, w in d.items()} for t, d in per_round.items()}}


if __name__ == "__main__":
    results = run_experiment()
    out_path = Path(__file__).parent / "benchmark_results" / "tau_experiment.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

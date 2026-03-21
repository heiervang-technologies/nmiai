#!/usr/bin/env python3
"""Experiment: Compare value-based viewport selector vs heuristic selector.

For each of 75 GT files:
1. Get initial_grid
2. Select viewports using both methods
3. Simulate observations by sampling from ground truth
4. Update predictions using observations
5. Compute wKL score
6. Compare

The key question: does selecting viewports based on expected information gain
beat the settlement-density heuristic?
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import regime_predictor as rp
from benchmark import load_ground_truth, kl_divergence, entropy
from query_runner import select_viewports_adaptive
from value_viewport_selector import (
    select_viewports,
    select_viewports_for_regime_detection,
)

GT_DIR = Path(__file__).parent / "ground_truth"
VIEWPORT_SIZE = 15


def simulate_observations(initial_grid, ground_truth, viewports, n_samples=1):
    """Simulate observations by sampling from ground truth distributions.

    Args:
        initial_grid: (H, W) array
        ground_truth: (H, W, 6) probability array
        viewports: list of (vx, vy) tuples
        n_samples: number of simulation samples per viewport

    Returns:
        list of observation dicts matching the API format
    """
    ig = np.array(initial_grid)
    gt = np.array(ground_truth)
    h, w = ig.shape
    observations = []

    for vx, vy in viewports:
        for _ in range(n_samples):
            vh = min(VIEWPORT_SIZE, h - vy)
            vw = min(VIEWPORT_SIZE, w - vx)
            grid = []
            for dy in range(vh):
                row = []
                for dx in range(vw):
                    y, x = vy + dy, vx + dx
                    # Sample from ground truth distribution
                    probs = gt[y, x]
                    cls = np.random.choice(6, p=probs)
                    # Map class index to cell code
                    code_map = {0: 10, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
                    # But class 0 = ocean/empty, need to check initial grid
                    if cls == 0:
                        code = int(ig[y, x]) if ig[y, x] in (10, 11, 0) else 11
                    else:
                        code = code_map[cls]
                    row.append(code)
                grid.append(row)
            observations.append({
                "viewport_x": vx,
                "viewport_y": vy,
                "grid": grid,
            })

    return observations


def score_prediction(pred, ground_truth):
    """Compute mean wKL (entropy-weighted KL divergence)."""
    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)
    gt = np.array(ground_truth, dtype=np.float64)

    kl = kl_divergence(gt, pred)
    H = entropy(gt)
    wkl = H * kl

    dynamic = H > 0.01
    if dynamic.any():
        return float(wkl[dynamic].mean())
    return 0.0


def evaluate_selector(selector_name, selector_fn, rounds_data, n_viewports=5,
                       n_queries_per=1, n_trials=3, verbose=False):
    """Evaluate a viewport selector across all GT data.

    Args:
        selector_name: name for reporting
        selector_fn: function(initial_grid, n_viewports) -> list of (vx, vy)
        rounds_data: ground truth data
        n_viewports: number of viewports to select
        n_queries_per: queries per viewport
        n_trials: number of random trials (observation sampling is stochastic)
        verbose: print per-seed results

    Returns:
        dict with aggregate metrics
    """
    all_wkl_no_obs = []
    all_wkl_with_obs = []
    all_improvements = []
    per_seed_results = []

    for rn in sorted(rounds_data.keys()):
        for sn in sorted(rounds_data[rn].keys()):
            ig = rounds_data[rn][sn]["initial_grid"]
            gt = np.array(rounds_data[rn][sn]["ground_truth"])

            # Baseline: prediction without observations
            pred_base = rp.predict(ig)
            wkl_base = score_prediction(pred_base, gt)

            # Select viewports
            try:
                viewports = selector_fn(ig, n_viewports)
            except Exception as e:
                if verbose:
                    print(f"  R{rn}S{sn} {selector_name}: selector failed: {e}")
                viewports = [(0, 0)] * n_viewports

            # Run multiple trials with different observation samples
            trial_wkls = []
            for trial in range(n_trials):
                np.random.seed(rn * 100 + sn * 10 + trial)
                obs = simulate_observations(ig, gt, viewports, n_samples=n_queries_per)
                pred_obs = rp.predict(ig, observations=obs)
                wkl_obs = score_prediction(pred_obs, gt)
                trial_wkls.append(wkl_obs)

            mean_wkl_obs = np.mean(trial_wkls)
            improvement = wkl_base - mean_wkl_obs

            all_wkl_no_obs.append(wkl_base)
            all_wkl_with_obs.append(mean_wkl_obs)
            all_improvements.append(improvement)

            result = {
                "round": rn, "seed": sn,
                "wkl_no_obs": wkl_base,
                "wkl_with_obs": mean_wkl_obs,
                "improvement": improvement,
                "improvement_pct": 100 * improvement / wkl_base if wkl_base > 0 else 0,
                "viewports": viewports,
            }
            per_seed_results.append(result)

            if verbose:
                print(f"  R{rn}S{sn} {selector_name}: "
                      f"base={wkl_base:.4f} obs={mean_wkl_obs:.4f} "
                      f"impr={improvement:.4f} ({result['improvement_pct']:.1f}%)")

    return {
        "selector": selector_name,
        "mean_wkl_no_obs": float(np.mean(all_wkl_no_obs)),
        "mean_wkl_with_obs": float(np.mean(all_wkl_with_obs)),
        "mean_improvement": float(np.mean(all_improvements)),
        "mean_improvement_pct": float(100 * np.mean(all_improvements) / np.mean(all_wkl_no_obs))
            if np.mean(all_wkl_no_obs) > 0 else 0,
        "median_improvement_pct": float(np.median([r["improvement_pct"] for r in per_seed_results])),
        "n_improved": sum(1 for r in per_seed_results if r["improvement"] > 0),
        "n_total": len(per_seed_results),
        "per_seed": per_seed_results,
    }


def heuristic_selector(initial_grid, n_viewports):
    """Wrap the existing heuristic selector."""
    ig = np.array(initial_grid)
    h, w = ig.shape
    vps = select_viewports_adaptive(initial_grid, h, w, n_viewports)
    return [(vp[0], vp[1]) for vp in vps]


def value_selector(initial_grid, n_viewports):
    """Wrap the value-based selector."""
    return select_viewports(initial_grid, n_viewports, n_queries_per=1)


def regime_selector(initial_grid, n_viewports):
    """Wrap the regime-discriminability selector."""
    return select_viewports_for_regime_detection(initial_grid, n_viewports)


def main():
    print("=" * 70)
    print("VALUE-BASED VIEWPORT SELECTOR EXPERIMENT")
    print("=" * 70)

    print("\nLoading ground truth...")
    rounds_data = load_ground_truth()
    total_seeds = sum(len(seeds) for seeds in rounds_data.values())
    print(f"Loaded {len(rounds_data)} rounds, {total_seeds} seeds")

    # Build model once
    print("Building regime predictor model...")
    model = rp.get_model()

    # Test configurations
    configs = [
        ("heuristic", heuristic_selector),
        ("value_based", value_selector),
        ("regime_disc", regime_selector),
    ]

    n_viewports_configs = [1, 3, 5]
    all_results = {}

    for n_vp in n_viewports_configs:
        print(f"\n{'='*70}")
        print(f"N_VIEWPORTS = {n_vp}")
        print(f"{'='*70}")

        for name, fn in configs:
            t0 = time.time()
            print(f"\nEvaluating: {name} (n_viewports={n_vp})...")
            result = evaluate_selector(name, fn, rounds_data,
                                        n_viewports=n_vp, n_trials=3, verbose=False)
            elapsed = time.time() - t0
            print(f"  Mean wKL (no obs):   {result['mean_wkl_no_obs']:.6f}")
            print(f"  Mean wKL (with obs): {result['mean_wkl_with_obs']:.6f}")
            print(f"  Mean improvement:    {result['mean_improvement']:.6f} "
                  f"({result['mean_improvement_pct']:.2f}%)")
            print(f"  Median improvement:  {result['median_improvement_pct']:.2f}%")
            print(f"  Seeds improved:      {result['n_improved']}/{result['n_total']}")
            print(f"  Time: {elapsed:.1f}s")

            all_results[f"{name}_vp{n_vp}"] = result

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'wKL base':>10} {'wKL obs':>10} {'Impr%':>8} {'Med%':>8} {'Won':>6}")
    print("-" * 70)

    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"{key:<25} {r['mean_wkl_no_obs']:>10.6f} {r['mean_wkl_with_obs']:>10.6f} "
              f"{r['mean_improvement_pct']:>7.2f}% {r['median_improvement_pct']:>7.2f}% "
              f"{r['n_improved']:>3}/{r['n_total']}")

    # Per-round breakdown for best config
    # Find best value-based config
    best_value = None
    best_heuristic = None
    for key, r in all_results.items():
        if "value" in key and (best_value is None or r['mean_wkl_with_obs'] < all_results[best_value]['mean_wkl_with_obs']):
            best_value = key
        if "heuristic" in key and (best_heuristic is None or r['mean_wkl_with_obs'] < all_results[best_heuristic]['mean_wkl_with_obs']):
            best_heuristic = key

    if best_value and best_heuristic:
        rv = all_results[best_value]
        rh = all_results[best_heuristic]
        diff_pct = 100 * (rh['mean_wkl_with_obs'] - rv['mean_wkl_with_obs']) / rh['mean_wkl_with_obs']
        print(f"\nBest value-based ({best_value}) vs best heuristic ({best_heuristic}):")
        print(f"  Value wKL:     {rv['mean_wkl_with_obs']:.6f}")
        print(f"  Heuristic wKL: {rh['mean_wkl_with_obs']:.6f}")
        print(f"  Difference:    {diff_pct:+.2f}% {'(value wins)' if diff_pct > 0 else '(heuristic wins)'}")

        # Per-round analysis
        print(f"\nPer-round breakdown ({best_value} vs {best_heuristic}):")
        rounds_seen = set()
        v_by_round = {}
        h_by_round = {}
        for s in rv['per_seed']:
            rn = s['round']
            v_by_round.setdefault(rn, []).append(s['wkl_with_obs'])
        for s in rh['per_seed']:
            rn = s['round']
            h_by_round.setdefault(rn, []).append(s['wkl_with_obs'])

        for rn in sorted(v_by_round.keys()):
            v_mean = np.mean(v_by_round[rn])
            h_mean = np.mean(h_by_round[rn])
            diff = 100 * (h_mean - v_mean) / h_mean if h_mean > 0 else 0
            winner = "V" if diff > 0 else "H"
            print(f"  R{rn:2d}: value={v_mean:.4f} heur={h_mean:.4f} diff={diff:+.1f}% [{winner}]")

    # Save results
    output_path = Path(__file__).parent / "value_selector_results.json"
    # Strip per_seed for smaller output
    save_results = {}
    for key, r in all_results.items():
        save_results[key] = {k: v for k, v in r.items() if k != "per_seed"}
    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    results = main()

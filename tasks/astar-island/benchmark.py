#!/usr/bin/env python3
"""Benchmarking harness for Astar Island predictors.

Evaluates any predictor function against ground truth using leave-one-round-out
cross-validation. Reports entropy-weighted KL divergence (the actual scoring metric).

Usage:
    uv run python3 benchmark.py <predictor_module>

    The module must expose: predict(initial_grid) -> np.ndarray (40x40x6)

Example:
    uv run python3 benchmark.py neighborhood_predictor
    uv run python3 benchmark.py predictor
"""

import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

GT_DIR = Path(__file__).parent / "ground_truth"


def load_ground_truth():
    """Load all ground truth files, grouped by round."""
    rounds = {}
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        name = f.stem  # round1_seed0
        parts = name.split("_")
        round_num = int(parts[0].replace("round", ""))
        seed_num = int(parts[1].replace("seed", ""))

        with open(f) as fh:
            data = json.load(fh)

        if "ground_truth" not in data or "initial_grid" not in data:
            continue

        if round_num not in rounds:
            rounds[round_num] = {}
        rounds[round_num][seed_num] = {
            "initial_grid": data["initial_grid"],
            "ground_truth": np.array(data["ground_truth"]),
        }
    return rounds


def kl_divergence(p, q):
    """KL(p || q) per cell, returns H x W array."""
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=2)


def entropy(p):
    """Shannon entropy per cell in bits, returns H x W array."""
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(p_safe * np.log2(p_safe), axis=2)


def evaluate_predictor(predict_fn, rounds_data, leave_out_round=None):
    """Evaluate a predictor against ground truth.

    Args:
        predict_fn: function(initial_grid) -> np.ndarray (H x W x 6)
        rounds_data: dict of {round_num: {seed_num: {initial_grid, ground_truth}}}
        leave_out_round: if set, only evaluate on this round (for CV)

    Returns:
        dict with metrics
    """
    all_kl = []
    all_weighted_kl = []
    all_entropy = []
    per_seed = []

    eval_rounds = [leave_out_round] if leave_out_round else sorted(rounds_data.keys())

    for rn in eval_rounds:
        if rn not in rounds_data:
            continue
        for sn in sorted(rounds_data[rn].keys()):
            gt = rounds_data[rn][sn]["ground_truth"]
            initial_grid = rounds_data[rn][sn]["initial_grid"]

            t0 = time.time()
            pred = predict_fn(initial_grid)
            elapsed = time.time() - t0

            # Ensure valid probabilities
            pred = np.maximum(pred, 0.01)
            pred /= pred.sum(axis=2, keepdims=True)

            kl = kl_divergence(gt, pred)
            H = entropy(gt)
            wkl = H * kl

            # Mask out static cells (entropy ~0)
            dynamic = H > 0.01

            mean_kl = kl[dynamic].mean() if dynamic.any() else 0
            mean_wkl = wkl[dynamic].mean() if dynamic.any() else 0
            mean_H = H[dynamic].mean() if dynamic.any() else 0

            seed_result = {
                "round": rn, "seed": sn,
                "mean_kl": float(mean_kl),
                "mean_weighted_kl": float(mean_wkl),
                "mean_entropy": float(mean_H),
                "dynamic_cells": int(dynamic.sum()),
                "elapsed_ms": elapsed * 1000,
            }
            per_seed.append(seed_result)
            all_kl.append(mean_kl)
            all_weighted_kl.append(mean_wkl)
            all_entropy.append(mean_H)

    return {
        "mean_kl": float(np.mean(all_kl)),
        "mean_weighted_kl": float(np.mean(all_weighted_kl)),
        "mean_entropy": float(np.mean(all_entropy)),
        "num_seeds": len(per_seed),
        "per_seed": per_seed,
    }


def cross_validate(predict_fn_factory, rounds_data):
    """Leave-one-round-out cross-validation.

    Args:
        predict_fn_factory: function(training_rounds_data) -> predict_fn
            Takes the training data, returns a predict function.
        rounds_data: all ground truth data

    Returns:
        dict with CV metrics
    """
    round_nums = sorted(rounds_data.keys())
    cv_results = []

    for held_out in round_nums:
        # Train on everything except held_out
        train_data = {rn: data for rn, data in rounds_data.items() if rn != held_out}
        predict_fn = predict_fn_factory(train_data)

        # Evaluate on held_out
        result = evaluate_predictor(predict_fn, rounds_data, leave_out_round=held_out)
        result["held_out_round"] = held_out
        cv_results.append(result)

    overall_kl = np.mean([r["mean_kl"] for r in cv_results])
    overall_wkl = np.mean([r["mean_weighted_kl"] for r in cv_results])

    return {
        "cv_mean_kl": float(overall_kl),
        "cv_mean_weighted_kl": float(overall_wkl),
        "per_round": cv_results,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: benchmark.py <predictor_module>")
        print("  The module must expose: predict(initial_grid) -> np.ndarray (40x40x6)")
        sys.exit(1)

    module_name = sys.argv[1]
    print(f"Loading predictor: {module_name}")

    # Add task dir to path
    sys.path.insert(0, str(Path(__file__).parent))
    mod = importlib.import_module(module_name)

    if not hasattr(mod, "predict"):
        print(f"ERROR: {module_name} has no predict() function")
        sys.exit(1)

    print("Loading ground truth...")
    rounds_data = load_ground_truth()
    total_seeds = sum(len(seeds) for seeds in rounds_data.values())
    print(f"Loaded {len(rounds_data)} rounds, {total_seeds} seeds")

    print("\n=== IN-SAMPLE EVALUATION ===")
    result = evaluate_predictor(mod.predict, rounds_data)
    print(f"Mean KL:          {result['mean_kl']:.6f}")
    print(f"Mean Weighted KL: {result['mean_weighted_kl']:.6f}")
    print(f"Mean Entropy:     {result['mean_entropy']:.4f}")

    print("\nPer-seed breakdown:")
    for s in result["per_seed"]:
        print(f"  R{s['round']}S{s['seed']}: KL={s['mean_kl']:.4f} wKL={s['mean_weighted_kl']:.4f} H={s['mean_entropy']:.3f} ({s['elapsed_ms']:.0f}ms)")

    # Save results
    results_path = Path(__file__).parent / "benchmark_results" / f"{module_name}.json"
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"module": module_name, "in_sample": result}, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()

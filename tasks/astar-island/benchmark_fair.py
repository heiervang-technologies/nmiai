#!/usr/bin/env python3
"""Fair benchmark: forces predictors to use their unknown-grid path.

This patches out memorization/fingerprinting so we see actual generalization.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from benchmark import load_ground_truth, evaluate_predictor


def benchmark_neighborhood():
    """Neighborhood predictor doesn't cheat, run as-is."""
    import neighborhood_predictor as nb
    return evaluate_predictor(nb.predict, load_ground_truth())


def benchmark_calibrated():
    """Calibrated predictor - run as-is."""
    import calibrated_predictor as cp
    return evaluate_predictor(cp.predict, load_ground_truth())


def benchmark_template_no_fingerprint():
    """Template predictor with fingerprint cache cleared."""
    import template_predictor as tp
    tp.get_model()  # ensure model is built
    tp._GRID_FINGERPRINTS.clear()  # clear fingerprint cache
    return evaluate_predictor(lambda grid: tp.predict(grid), load_ground_truth())


def benchmark_diffusion_no_memory():
    """Diffusion model with memory bank cleared."""
    import diffusion_model as dm
    artifacts = dm.get_artifacts()
    saved_bank = artifacts.memory_bank.copy()
    artifacts.memory_bank.clear()  # force denoiser path
    result = evaluate_predictor(dm.predict, load_ground_truth())
    artifacts.memory_bank = saved_bank  # restore
    return result


def benchmark_spatial():
    """Spatial model - run as-is."""
    import spatial_model as sm
    return evaluate_predictor(sm.predict, load_ground_truth())


def main():
    results = {}

    predictors = [
        ("neighborhood", benchmark_neighborhood),
        ("calibrated", benchmark_calibrated),
        ("template_NO_FINGERPRINT", benchmark_template_no_fingerprint),
        ("diffusion_NO_MEMORY", benchmark_diffusion_no_memory),
        ("spatial", benchmark_spatial),
    ]

    if len(sys.argv) > 1:
        # Run specific predictor
        name = sys.argv[1]
        for pname, fn in predictors:
            if name in pname.lower():
                print(f"\n=== {pname} ===")
                r = fn()
                print(f"  Mean KL:  {r['mean_kl']:.6f}")
                print(f"  Mean wKL: {r['mean_weighted_kl']:.6f}")
                results[pname] = r
                break
    else:
        # Run all
        for pname, fn in predictors:
            print(f"\n=== {pname} ===")
            try:
                r = fn()
                print(f"  Mean KL:  {r['mean_kl']:.6f}")
                print(f"  Mean wKL: {r['mean_weighted_kl']:.6f}")
                results[pname] = r
            except Exception as e:
                print(f"  ERROR: {e}")
                results[pname] = {"error": str(e)}

    print("\n" + "="*60)
    print("LEADERBOARD (by weighted KL, lower is better):")
    print("="*60)
    ranked = sorted(
        [(k, v) for k, v in results.items() if "error" not in v],
        key=lambda x: x[1]["mean_weighted_kl"]
    )
    for i, (name, r) in enumerate(ranked, 1):
        print(f"  {i}. {name:35s} wKL={r['mean_weighted_kl']:.6f}  KL={r['mean_kl']:.6f}")

    out_path = Path(__file__).parent / "benchmark_results" / "fair_comparison.json"
    out_path.parent.mkdir(exist_ok=True)
    # Convert numpy to native python for JSON
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()

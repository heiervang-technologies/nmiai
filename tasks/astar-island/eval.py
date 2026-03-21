#!/usr/bin/env python3
"""Clean eval pipeline for Astar Island predictors.

Single script, one command. Gives honest out-of-sample signal via
leave-one-round-out cross-validation.

Usage:
    uv run python3 eval.py regime_predictor          # CV, no observations
    uv run python3 eval.py regime_predictor --obs 5   # CV, 5 deterministic observations per seed
    uv run python3 eval.py regime_predictor --insample # In-sample only (fast sanity check)
    uv run python3 eval.py regime_predictor --obs 5 --tau 20  # With Dirichlet overlay

Predictor contract:
    predict(initial_grid, observations=None) -> np.ndarray (H x W x 6)

Observation simulation is DETERMINISTIC: uses argmax of GT distribution per
cell (no RNG). This gives perfectly reproducible scores.
"""

import importlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

GT_DIR = Path(__file__).parent / "ground_truth"
N_CLASSES = 6
FLOOR = 0.005
VIEWPORT = 15


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def kl_divergence(p, q):
    """KL(p || q) per cell, returns H x W array."""
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=2)


def entropy(p):
    """Shannon entropy per cell in bits, returns H x W array."""
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(p_safe * np.log2(p_safe), axis=2)


def score_prediction(gt, pred):
    """Score a single prediction. Returns dict with wKL, KL, etc."""
    pred = np.maximum(pred, FLOOR)
    pred /= pred.sum(axis=2, keepdims=True)

    kl = kl_divergence(gt, pred)
    H = entropy(gt)
    wkl = H * kl
    dynamic = H > 0.01

    if not dynamic.any():
        return {"wkl": 0.0, "kl": 0.0, "entropy": 0.0, "dynamic_cells": 0}

    return {
        "wkl": float(wkl[dynamic].mean()),
        "kl": float(kl[dynamic].mean()),
        "entropy": float(H[dynamic].mean()),
        "dynamic_cells": int(dynamic.sum()),
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_ground_truth():
    """Load all GT files, grouped by round."""
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
            "initial_grid": data["initial_grid"],
            "ground_truth": np.array(data["ground_truth"]),
        }
    return rounds


# ---------------------------------------------------------------------------
# Deterministic observation simulation
# ---------------------------------------------------------------------------

def make_observations(gt, ig, n_viewports, rng_seed=42):
    """Simulate observations from ground truth.

    Samples one outcome per cell from the GT distribution using a fixed RNG
    seed for reproducibility. Viewports placed at highest-entropy positions.
    """
    if n_viewports <= 0:
        return None

    rng = np.random.RandomState(rng_seed)
    h, w, _ = gt.shape
    H = entropy(gt)

    max_vy = h - VIEWPORT
    max_vx = w - VIEWPORT
    if max_vy < 0 or max_vx < 0:
        return None

    # Integral image for fast window entropy sums
    H_pad = np.zeros((h + 1, w + 1))
    H_pad[1:, 1:] = np.cumsum(np.cumsum(H, axis=0), axis=1)

    def window_sum(vy, vx):
        return (H_pad[vy + VIEWPORT, vx + VIEWPORT]
                - H_pad[vy, vx + VIEWPORT]
                - H_pad[vy + VIEWPORT, vx]
                + H_pad[vy, vx])

    observations = []
    used = set()

    for _ in range(n_viewports):
        best_score = -1.0
        best_pos = (0, 0)

        for vy in range(max_vy + 1):
            for vx in range(max_vx + 1):
                too_close = any(
                    abs(vy - uvy) < 8 and abs(vx - uvx) < 8
                    for uvy, uvx in used
                )
                if too_close:
                    continue
                s = window_sum(vy, vx)
                if s > best_score:
                    best_score = s
                    best_pos = (vy, vx)

        vy, vx = best_pos
        used.add((vy, vx))

        # Sample one outcome per cell from GT distribution
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


# ---------------------------------------------------------------------------
# Tau overlay
# ---------------------------------------------------------------------------

def apply_tau_overlay(pred, ig, observations, tau):
    """Dirichlet posterior update on observed cells."""
    if not observations or tau <= 0:
        return pred

    h, w, nc = pred.shape
    counts = np.zeros_like(pred)

    for obs in observations:
        grid = obs.get("grid", [])
        vx = int(obs.get("viewport_x", 0))
        vy = int(obs.get("viewport_y", 0))
        for dy, row in enumerate(grid):
            for dx, cell in enumerate(row):
                y, x = vy + dy, vx + dx
                if 0 <= y < h and 0 <= x < w:
                    counts[y, x, int(cell)] += 1.0

    observed = counts.sum(axis=2) > 0
    alpha = tau * pred + counts
    posterior = alpha / alpha.sum(axis=2, keepdims=True)

    out = pred.copy()
    out[observed] = posterior[observed]
    out = np.maximum(out, FLOOR)
    out /= out.sum(axis=2, keepdims=True)
    return out


# ---------------------------------------------------------------------------
# Model rebuilding for CV (no file-system hacks)
# ---------------------------------------------------------------------------

def rebuild_predictor(module_name, train_rounds):
    """Rebuild a predictor's model from specific training rounds only.

    Returns a predict function that uses only train_rounds data.
    Requires the module to expose build_model_from_data() and predict_with_model().
    """
    mod = importlib.import_module(module_name)

    if not hasattr(mod, "build_model_from_data"):
        raise RuntimeError(
            f"{module_name} does not expose build_model_from_data(). "
            "Add it to support leave-one-round-out CV."
        )

    model = mod.build_model_from_data(train_rounds)

    def predict_fn(initial_grid, observations=None):
        return mod.predict_with_model(initial_grid, model, observations=observations)

    return predict_fn


# ---------------------------------------------------------------------------
# Evaluation modes
# ---------------------------------------------------------------------------

def eval_insample(mod, rounds, n_obs, tau):
    """Quick in-sample evaluation (no CV)."""
    results = []
    for rn in sorted(rounds.keys()):
        for sn in sorted(rounds[rn].keys()):
            sd = rounds[rn][sn]
            obs = make_observations(sd["ground_truth"], sd["initial_grid"], n_obs) if n_obs else None
            pred = mod.predict(sd["initial_grid"], observations=obs) if obs else mod.predict(sd["initial_grid"])
            if tau and obs:
                pred = apply_tau_overlay(pred, sd["initial_grid"], obs, tau)
            s = score_prediction(sd["ground_truth"], pred)
            s["round"] = rn
            s["seed"] = sn
            results.append(s)
    return results


def eval_cv(module_name, rounds, n_obs, tau):
    """Leave-one-round-out cross-validation."""
    round_nums = sorted(rounds.keys())
    results = []

    for held_out in round_nums:
        train_rounds = {rn: seeds for rn, seeds in rounds.items() if rn != held_out}
        predict_fn = rebuild_predictor(module_name, train_rounds)

        for sn in sorted(rounds[held_out].keys()):
            sd = rounds[held_out][sn]
            obs = make_observations(sd["ground_truth"], sd["initial_grid"], n_obs) if n_obs else None
            if obs:
                pred = predict_fn(sd["initial_grid"], observations=obs)
            else:
                pred = predict_fn(sd["initial_grid"])
            if tau and obs:
                pred = apply_tau_overlay(pred, sd["initial_grid"], obs, tau)
            s = score_prediction(sd["ground_truth"], pred)
            s["round"] = held_out
            s["seed"] = sn
            results.append(s)

        # Clear model for next fold
        mod = importlib.import_module(module_name)
        if hasattr(mod, "_MODEL"):
            mod._MODEL = None
        if hasattr(mod, "_MODEL_CACHE"):
            mod._MODEL_CACHE = None

    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(results, mode_label):
    """Print clean summary."""
    # Per-round aggregation
    by_round = defaultdict(list)
    for r in results:
        by_round[r["round"]].append(r["wkl"])

    round_means = {rn: np.mean(wkls) for rn, wkls in by_round.items()}
    overall = np.mean([w for r in results for w in [r["wkl"]]])

    print(f"\n{'─' * 50}")
    print(f"  {mode_label}")
    print(f"{'─' * 50}")
    print(f"  {'Round':>6s}  {'wKL':>8s}  {'seeds':>5s}")
    print(f"  {'─'*6}  {'─'*8}  {'─'*5}")

    for rn in sorted(round_means.keys()):
        n = len(by_round[rn])
        print(f"  R{rn:<5d} {round_means[rn]:8.4f}  {n:5d}")

    print(f"  {'─'*6}  {'─'*8}  {'─'*5}")
    print(f"  {'MEAN':>6s} {overall:8.4f}  {len(results):5d}")

    # Best/worst
    sorted_rounds = sorted(round_means.items(), key=lambda x: x[1])
    best_rn, best_wkl = sorted_rounds[0]
    worst_rn, worst_wkl = sorted_rounds[-1]
    print(f"\n  Best:  R{best_rn} ({best_wkl:.4f})")
    print(f"  Worst: R{worst_rn} ({worst_wkl:.4f})")
    print(f"{'─' * 50}")

    return {"overall_wkl": float(overall), "per_round": round_means}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Astar Island eval pipeline")
    parser.add_argument("module", help="Predictor module name")
    parser.add_argument("--obs", type=int, default=0, help="Number of simulated observation viewports")
    parser.add_argument("--tau", type=float, default=0, help="Dirichlet overlay tau (0=disabled)")
    parser.add_argument("--insample", action="store_true", help="In-sample only (no CV)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    mod = importlib.import_module(args.module)

    if not hasattr(mod, "predict"):
        print(f"ERROR: {args.module} has no predict() function")
        sys.exit(1)

    rounds = load_ground_truth()
    n_rounds = len(rounds)
    n_seeds = sum(len(s) for s in rounds.values())
    obs_label = f"{args.obs} obs" if args.obs else "no obs"
    tau_label = f", tau={args.tau}" if args.tau else ""

    print(f"Predictor: {args.module}")
    print(f"Data:      {n_rounds} rounds, {n_seeds} seeds")
    print(f"Config:    {obs_label}{tau_label}")

    t0 = time.time()

    if args.insample:
        results = eval_insample(mod, rounds, args.obs, args.tau)
        summary = report(results, f"IN-SAMPLE ({obs_label}{tau_label})")
    else:
        results = eval_cv(args.module, rounds, args.obs, args.tau)
        summary = report(results, f"LEAVE-ONE-ROUND-OUT CV ({obs_label}{tau_label})")

    elapsed = time.time() - t0
    print(f"\n  Time: {elapsed:.1f}s")

    # Save
    out = {
        "module": args.module,
        "mode": "insample" if args.insample else "cv",
        "obs": args.obs,
        "tau": args.tau,
        "overall_wkl": summary["overall_wkl"],
        "per_round": {str(k): v for k, v in summary["per_round"].items()},
    }
    out_path = Path(__file__).parent / "benchmark_results" / f"eval_{args.module}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()

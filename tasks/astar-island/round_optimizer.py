#!/usr/bin/env python3
"""Per-round optimization script for Astar Island.

Tries multiple prediction strategies against a proxy evaluation target
(closest historical round by regime similarity), picks the best one,
and submits it.

Usage:
    uv run python3 tasks/astar-island/round_optimizer.py
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

# ── Setup paths ──────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
GT_DIR = BASE_DIR / "ground_truth"
LOG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

import regime_predictor
import spatial_model

# ── API setup ────────────────────────────────────────────────────────────────

API_BASE = "https://api.ainm.no"

TOKEN = os.environ.get("AINM_TOKEN", "")
if not TOKEN:
    token_file = BASE_DIR / ".token"
    if token_file.exists():
        TOKEN = token_file.read_text().strip()

session = requests.Session()
if TOKEN:
    session.cookies.set("access_token", TOKEN)
    session.headers["Authorization"] = f"Bearer {TOKEN}"

N_CLASSES = 6
FLOOR = 1e-6


# ── API helpers ──────────────────────────────────────────────────────────────

def api_get(endpoint):
    url = f"{API_BASE}{endpoint}"
    resp = session.get(url)
    return resp.json() if resp.ok else {"error": resp.text}


def api_post(endpoint, payload):
    url = f"{API_BASE}{endpoint}"
    resp = session.post(url, json=payload)
    data = resp.json() if resp.ok else {"error": resp.text, "status": resp.status_code}
    return data, resp.status_code


def find_active_round():
    rounds = api_get("/astar-island/rounds")
    if isinstance(rounds, dict) and "error" in rounds:
        print(f"ERROR fetching rounds: {rounds}")
        return None
    for r in rounds:
        if r["status"] == "active":
            return r
    return None


def get_round_details(round_id):
    return api_get(f"/astar-island/rounds/{round_id}")


def submit_prediction(round_id, seed_index, prediction):
    payload = {
        "round_id": round_id,
        "seed_index": seed_index,
        "prediction": prediction.tolist(),
    }
    return api_post("/astar-island/submit", payload)


# ── Ground truth + evaluation helpers ────────────────────────────────────────

def load_ground_truth():
    """Load all GT files grouped by round."""
    rounds = {}
    for f in sorted(GT_DIR.glob("round*_seed*.json")):
        rn = int(f.stem.split("_")[0].replace("round", ""))
        sn = int(f.stem.split("_")[1].replace("seed", ""))
        with open(f) as fh:
            data = json.load(fh)
        if "ground_truth" not in data or "initial_grid" not in data:
            continue
        rounds.setdefault(rn, {})[sn] = {
            "initial_grid": np.array(data["initial_grid"], dtype=np.int32),
            "ground_truth": np.array(data["ground_truth"], dtype=np.float64),
        }
    return rounds


def kl_divergence(p, q):
    """KL(p || q) per cell."""
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    return np.sum(p_safe * np.log(p_safe / q_safe), axis=2)


def entropy(p):
    """Shannon entropy per cell in bits."""
    p_safe = np.maximum(p, 1e-10)
    return -np.sum(p_safe * np.log2(p_safe), axis=2)


def compute_wkl(gt, pred):
    """Compute mean weighted KL for dynamic cells."""
    pred = np.maximum(pred, 0.01)
    pred = pred / pred.sum(axis=2, keepdims=True)
    kl = kl_divergence(gt, pred)
    H = entropy(gt)
    wkl = H * kl
    dynamic = H > 0.01
    if dynamic.any():
        return float(wkl[dynamic].mean())
    return 0.0


# ── Regime similarity ────────────────────────────────────────────────────────

def compute_frontier_settle_rate(initial_grid, ground_truth):
    """Compute frontier settlement rate for a seed."""
    ig = np.array(initial_grid, dtype=np.int32)
    gt = np.array(ground_truth, dtype=np.float64)
    dist_civ, _, _, _ = regime_predictor.compute_features(ig)
    frontier = (
        (dist_civ >= 1) & (dist_civ <= 5)
        & (ig != regime_predictor.OCEAN)
        & (ig != regime_predictor.MOUNTAIN)
    )
    if frontier.any():
        return float(gt[frontier, 1].mean())
    return 0.1


def detect_regime_from_obs(ig, observations):
    """Get regime weights from observations using Bayesian detection."""
    return regime_predictor.detect_regime_from_observations(ig, observations)


def find_closest_historical_round(regime_weights, gt_rounds):
    """Find the historical round most similar by regime profile.

    Computes a similarity score based on how close the detected regime
    weights are to each historical round's actual regime classification.
    """
    # Map regime labels to weight vectors for comparison
    regime_vectors = {
        "harsh": np.array([1.0, 0.0, 0.0]),
        "moderate": np.array([0.0, 1.0, 0.0]),
        "prosperous": np.array([0.0, 0.0, 1.0]),
    }

    detected_vec = np.array([
        regime_weights.get("harsh", 0.0),
        regime_weights.get("moderate", 0.0),
        regime_weights.get("prosperous", 0.0),
    ])
    detected_vec /= detected_vec.sum()

    best_round = None
    best_similarity = -1.0

    for rn, seeds in gt_rounds.items():
        # Classify this historical round
        regime = regime_predictor.classify_round(seeds)
        round_vec = regime_vectors[regime]

        # Cosine similarity
        sim = float(np.dot(detected_vec, round_vec))
        if sim > best_similarity:
            best_similarity = sim
            best_round = rn

    return best_round, best_similarity


# ── Prediction strategies ────────────────────────────────────────────────────

def strategy_regime_base(ig, observations, model):
    """Strategy A: regime_predictor with observations (current SOTA)."""
    return regime_predictor.predict_with_model(ig, model, observations)


def strategy_regime_mod5(ig, observations, model):
    """Strategy B: regime_predictor with mod-5 cycle prior boost.

    Boosts the prior weight for rounds that fall on a mod-5 cycle,
    which empirically shows higher settlement persistence.
    """
    pred = regime_predictor.predict_with_model(ig, model, observations)

    # Mod-5 cycle boost: slightly increase settlement probability
    # for cells near existing settlements
    ig_arr = np.array(ig, dtype=np.int32)
    dist_civ, _, _, _ = regime_predictor.compute_features(ig_arr)

    boost_mask = (dist_civ <= 3) & (ig_arr != regime_predictor.OCEAN) & (ig_arr != regime_predictor.MOUNTAIN)
    boost_factor = 1.05  # 5% boost to settlement class

    pred_mod = pred.copy()
    pred_mod[boost_mask, 1] *= boost_factor  # settlement class
    pred_mod = np.maximum(pred_mod, FLOOR)
    pred_mod /= pred_mod.sum(axis=2, keepdims=True)

    return pred_mod


def strategy_regime_tau(ig, observations, model, tau_override):
    """Strategy C: regime_predictor with modified tau (prior strength).

    Instead of changing the model, we post-process by blending the
    regime prediction with a uniform prior at strength 1/tau.
    """
    pred = regime_predictor.predict_with_model(ig, model, observations)

    # Blend with uniform: higher tau = trust model more
    ig_arr = np.array(ig, dtype=np.int32)
    uniform = np.ones(N_CLASSES) / N_CLASSES

    alpha = 1.0 / tau_override  # smaller tau -> more uniform blending
    pred_tau = pred * (1.0 - alpha) + uniform * alpha
    pred_tau = np.maximum(pred_tau, FLOOR)
    pred_tau /= pred_tau.sum(axis=2, keepdims=True)

    return pred_tau


def strategy_blend_regime_spatial(ig, observations, model, regime_weight=0.7):
    """Strategy D: Blend regime_predictor + spatial_model."""
    pred_regime = regime_predictor.predict_with_model(ig, model, observations)
    pred_spatial = spatial_model.predict(ig)

    pred_blend = regime_weight * pred_regime + (1.0 - regime_weight) * pred_spatial
    pred_blend = np.maximum(pred_blend, FLOOR)
    pred_blend /= pred_blend.sum(axis=2, keepdims=True)

    return pred_blend


# ── Load observations from disk ──────────────────────────────────────────────

def load_observations(round_number, seed_idx):
    """Load observations from logs/roundN/observations_seedM.json."""
    obs_path = LOG_DIR / f"round{round_number}" / f"observations_seed{seed_idx}.json"
    if not obs_path.exists():
        return []
    with open(obs_path) as f:
        return json.load(f)


# ── Main optimizer logic ─────────────────────────────────────────────────────

def run_optimizer():
    """Main optimization loop."""
    print("=" * 60)
    print("ASTAR ISLAND ROUND OPTIMIZER")
    print("=" * 60)

    if not TOKEN:
        print("ERROR: No auth token. Set AINM_TOKEN or create tasks/astar-island/.token")
        sys.exit(1)

    # 1. Find active round
    active = find_active_round()
    if not active:
        print("No active round found.")
        sys.exit(0)

    round_id = active["id"]
    round_number = active["round_number"]
    print(f"Active round: {round_number} (ID: {round_id})")
    print(f"Closes at: {active.get('closes_at', 'unknown')}")

    # 2. Get round details
    details = get_round_details(round_id)
    if "error" in details:
        print(f"ERROR getting round details: {details}")
        sys.exit(1)

    height = details["map_height"]
    width = details["map_width"]
    seeds_count = details["seeds_count"]
    initial_states = details["initial_states"]

    print(f"Map: {width}x{height}, Seeds: {seeds_count}")

    # 3. Build regime predictor model from GT
    print("\nBuilding regime predictor model...")
    gt_rounds = load_ground_truth()
    model = regime_predictor.build_model_from_data(gt_rounds)
    print(f"  Loaded {len(gt_rounds)} historical rounds")

    # Save round details
    round_dir = LOG_DIR / f"round{round_number}"
    round_dir.mkdir(exist_ok=True)

    # 4. Define strategies to try
    strategies = {
        "regime_base": lambda ig, obs: strategy_regime_base(ig, obs, model),
        "regime_mod5": lambda ig, obs: strategy_regime_mod5(ig, obs, model),
        "regime_tau20": lambda ig, obs: strategy_regime_tau(ig, obs, model, 20),
        "regime_tau30": lambda ig, obs: strategy_regime_tau(ig, obs, model, 30),
        "regime_tau40": lambda ig, obs: strategy_regime_tau(ig, obs, model, 40),
        "blend_70_30": lambda ig, obs: strategy_blend_regime_spatial(ig, obs, model, 0.7),
        "blend_80_20": lambda ig, obs: strategy_blend_regime_spatial(ig, obs, model, 0.8),
        "blend_90_10": lambda ig, obs: strategy_blend_regime_spatial(ig, obs, model, 0.9),
    }

    # 5. For each seed, try all strategies and pick the best
    results = {}
    winning_predictions = {}

    for seed_idx in range(seeds_count):
        ig = initial_states[seed_idx]["grid"]
        ig_arr = np.array(ig, dtype=np.int32)

        # Load observations
        observations = load_observations(round_number, seed_idx)
        print(f"\nSeed {seed_idx}: {len(observations)} observations loaded")

        # Detect regime
        regime_weights = detect_regime_from_obs(ig_arr, observations)
        print(f"  Regime weights: " + ", ".join(
            f"{r}={w:.3f}" for r, w in sorted(regime_weights.items())
        ))

        # Find closest historical round for proxy evaluation
        closest_round, similarity = find_closest_historical_round(regime_weights, gt_rounds)
        print(f"  Closest historical round: {closest_round} (similarity: {similarity:.3f})")

        if closest_round is None or closest_round not in gt_rounds:
            print(f"  WARNING: No valid proxy round found, using regime_base")
            pred = strategy_regime_base(ig_arr, observations, model)
            winning_predictions[seed_idx] = pred
            results[seed_idx] = {
                "winner": "regime_base",
                "reason": "no proxy round available",
                "regime_weights": regime_weights,
            }
            continue

        # Get proxy GT from closest round
        # Use the same seed index if available, otherwise use seed 0
        proxy_seeds = gt_rounds[closest_round]
        proxy_seed_idx = seed_idx if seed_idx in proxy_seeds else min(proxy_seeds.keys())
        proxy_gt = proxy_seeds[proxy_seed_idx]["ground_truth"]
        proxy_ig = proxy_seeds[proxy_seed_idx]["initial_grid"]

        # Evaluate each strategy against the proxy GT
        strategy_scores = {}
        strategy_preds = {}

        for name, strategy_fn in strategies.items():
            try:
                # Run strategy on the ACTUAL initial grid
                pred = strategy_fn(ig_arr, observations)
                strategy_preds[name] = pred

                # Also run strategy on the PROXY initial grid for evaluation
                proxy_pred = strategy_fn(proxy_ig, [])  # No observations for proxy
                wkl = compute_wkl(proxy_gt, proxy_pred)
                strategy_scores[name] = wkl
            except Exception as e:
                print(f"  Strategy {name} FAILED: {e}")
                strategy_scores[name] = float("inf")

        # Pick winner (lowest estimated wKL)
        winner = min(strategy_scores, key=strategy_scores.get)
        winning_wkl = strategy_scores[winner]

        print(f"\n  Strategy scores (proxy wKL on round {closest_round}):")
        for name in sorted(strategy_scores, key=strategy_scores.get):
            marker = " <-- WINNER" if name == winner else ""
            print(f"    {name}: {strategy_scores[name]:.6f}{marker}")

        winning_predictions[seed_idx] = strategy_preds[winner]
        results[seed_idx] = {
            "winner": winner,
            "proxy_round": closest_round,
            "proxy_similarity": similarity,
            "proxy_wkl": winning_wkl,
            "all_scores": {k: float(v) for k, v in strategy_scores.items()},
            "regime_weights": {k: float(v) for k, v in regime_weights.items()},
        }

    # 6. Submit winning predictions
    print(f"\n{'=' * 60}")
    print("SUBMITTING WINNING PREDICTIONS")
    print(f"{'=' * 60}")

    for seed_idx in range(seeds_count):
        winner = results[seed_idx]["winner"]
        pred = winning_predictions[seed_idx]
        print(f"\nSeed {seed_idx}: submitting '{winner}'")

        result, status = submit_prediction(round_id, seed_idx, pred)
        print(f"  Submit status={status}, result={result}")

        # Save prediction
        np.save(round_dir / f"optimizer_pred_seed{seed_idx}.npy", pred)
        time.sleep(0.25)

    # 7. Save optimizer results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "round_number": round_number,
        "round_id": round_id,
        "seeds_count": seeds_count,
        "per_seed": {str(k): v for k, v in results.items()},
        "summary": {
            "strategies_tried": list(strategies.keys()),
            "winners": {str(k): v["winner"] for k, v in results.items()},
        },
    }

    output_path = round_dir / "optimizer_result.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("OPTIMIZER SUMMARY")
    print(f"{'=' * 60}")
    for seed_idx in range(seeds_count):
        r = results[seed_idx]
        print(f"  Seed {seed_idx}: {r['winner']}"
              f" (proxy round {r.get('proxy_round', 'N/A')},"
              f" est. wKL {r.get('proxy_wkl', 'N/A')})")


if __name__ == "__main__":
    run_optimizer()

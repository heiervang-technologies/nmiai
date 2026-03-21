#!/usr/bin/env python3
"""Regime-detecting predictor for Astar Island.

Strategy:
1. Use 1-2 queries per seed to check if initial settlements survived
2. Classify regime: harsh (<10% survival), moderate (10-33%), prosperous (>33%)
3. Apply regime-specific bucket priors (36% better than pooled)
4. Remaining queries: repeat hot viewport for empirical distributions
"""

import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from scipy.ndimage import binary_dilation, distance_transform_cdt

BASE_DIR = Path(__file__).parent
GT_DIR = BASE_DIR / "ground_truth"
LOG_DIR = BASE_DIR / "logs"

N_CLASSES = 6
REGIMES = ['harsh', 'moderate', 'prosperous']

# Regime thresholds (settlement survival rate)
HARSH_THRESHOLD = 0.10
MODERATE_THRESHOLD = 0.33


def build_regime_priors():
    """Build per-regime bucket priors from all ground truth."""
    # First classify each round
    round_regimes = {}
    for f in sorted(GT_DIR.glob('round*_seed0.json')):
        with open(f) as fh:
            data = json.load(fh)
        if 'ground_truth' not in data:
            continue
        rn = int(f.stem.split('_')[0].replace('round', ''))
        init = np.array(data['initial_grid'])
        gt = np.array(data['ground_truth'])
        settle = (init == 1)
        surv = gt[settle][:, 1].mean() if settle.any() else 0
        if surv < HARSH_THRESHOLD:
            round_regimes[rn] = 'harsh'
        elif surv < MODERATE_THRESHOLD:
            round_regimes[rn] = 'moderate'
        else:
            round_regimes[rn] = 'prosperous'

    # Build bucket priors per regime
    regime_priors = {r: defaultdict(lambda: {'sum': np.zeros(N_CLASSES), 'count': 0}) for r in REGIMES}

    for f in sorted(GT_DIR.glob('round*_seed*.json')):
        with open(f) as fh:
            data = json.load(fh)
        if 'ground_truth' not in data:
            continue
        rn = int(f.stem.split('_')[0].replace('round', ''))
        if rn not in round_regimes:
            continue
        regime = round_regimes[rn]

        init = np.array(data['initial_grid'])
        gt = np.array(data['ground_truth'])
        civ_mask = (init == 1) | (init == 2)
        ocean = (init == 10)
        ocean_adj = binary_dilation(ocean, np.ones((3, 3))) & ~ocean
        civ_dist = distance_transform_cdt(~civ_mask, metric='taxicab') if civ_mask.any() else np.full(init.shape, 99)

        for y in range(40):
            for x in range(40):
                cell = init[y, x]
                if cell in (10, 5):
                    continue
                itype = 'F' if cell == 4 else ('S' if cell == 1 else ('P' if cell == 2 else 'X'))
                d = min(int(civ_dist[y, x]), 15)
                oa = bool(ocean_adj[y, x])
                key = (itype, d, oa)
                regime_priors[regime][key]['sum'] += gt[y, x]
                regime_priors[regime][key]['count'] += 1

    return regime_priors, round_regimes


# Module-level cache
_REGIME_PRIORS = None
_ROUND_REGIMES = None


def get_priors():
    global _REGIME_PRIORS, _ROUND_REGIMES
    if _REGIME_PRIORS is None:
        _REGIME_PRIORS, _ROUND_REGIMES = build_regime_priors()
    return _REGIME_PRIORS, _ROUND_REGIMES


def detect_regime_from_observations(initial_grid, observations):
    """Detect regime by checking settlement survival in observations."""
    init = np.array(initial_grid)
    settle_positions = set(zip(*np.where(init == 1)))

    survived = 0
    checked = 0

    for obs in observations:
        vx, vy = obs['viewport_x'], obs['viewport_y']
        for dy, row in enumerate(obs['grid']):
            for dx, cell in enumerate(row):
                y, x = vy + dy, vx + dx
                if (y, x) in settle_positions:
                    checked += 1
                    if cell == 1:  # Still a settlement
                        survived += 1

    if checked == 0:
        return 'moderate'  # Default if no settlements observed

    survival_rate = survived / checked
    if survival_rate < HARSH_THRESHOLD:
        return 'harsh'
    elif survival_rate < MODERATE_THRESHOLD:
        return 'moderate'
    else:
        return 'prosperous'


def predict(initial_grid, regime=None, observations=None):
    """Predict using regime-specific priors.

    Args:
        initial_grid: 40x40 grid
        regime: 'harsh', 'moderate', 'prosperous', or None (auto-detect/default)
        observations: list of observation dicts for regime detection
    """
    regime_priors, _ = get_priors()

    if regime is None and observations:
        regime = detect_regime_from_observations(initial_grid, observations)
    elif regime is None:
        regime = 'moderate'  # Default to moderate (closest to pooled average)

    init = np.array(initial_grid)
    pred = np.zeros((40, 40, N_CLASSES))

    civ_mask = (init == 1) | (init == 2)
    ocean = (init == 10)
    ocean_adj = binary_dilation(ocean, np.ones((3, 3))) & ~ocean
    civ_dist = distance_transform_cdt(~civ_mask, metric='taxicab') if civ_mask.any() else np.full(init.shape, 99)

    priors = regime_priors[regime]

    for y in range(40):
        for x in range(40):
            cell = init[y, x]
            if cell == 10:
                pred[y, x] = [1, 0, 0, 0, 0, 0]
                continue
            if cell == 5:
                pred[y, x] = [0, 0, 0, 0, 0, 1]
                continue

            itype = 'F' if cell == 4 else ('S' if cell == 1 else ('P' if cell == 2 else 'X'))
            d = min(int(civ_dist[y, x]), 15)
            oa = bool(ocean_adj[y, x])
            key = (itype, d, oa)

            entry = priors.get(key)
            if entry and entry['count'] >= 3:
                pred[y, x] = entry['sum'] / entry['count']
            else:
                key2 = (itype, d, not oa)
                entry2 = priors.get(key2)
                if entry2 and entry2['count'] >= 3:
                    pred[y, x] = entry2['sum'] / entry2['count']
                else:
                    # Fallback: try nearby distance
                    for dd in [d - 1, d + 1, d - 2, d + 2]:
                        if dd < 0:
                            continue
                        for o in [oa, not oa]:
                            e = priors.get((itype, dd, o))
                            if e and e['count'] >= 3:
                                pred[y, x] = e['sum'] / e['count']
                                break
                        if pred[y, x].sum() > 0:
                            break
                    if pred[y, x].sum() == 0:
                        pred[y, x] = np.ones(N_CLASSES) / N_CLASSES

    # Also blend in empirical observations if available and cell has 3+ samples
    if observations:
        counts = np.zeros((40, 40, N_CLASSES))
        obs_count = np.zeros((40, 40), dtype=int)
        for obs in observations:
            vx, vy = obs['viewport_x'], obs['viewport_y']
            for dy, row in enumerate(obs['grid']):
                for dx, cell in enumerate(row):
                    y, x = vy + dy, vx + dx
                    if 0 <= y < 40 and 0 <= x < 40:
                        cls = cell if cell < 6 else 0
                        if cell in (0, 10, 11):
                            cls = 0
                        counts[y, x, cls] += 1
                        obs_count[y, x] += 1

        # Bayesian update for cells with 3+ observations
        for y in range(40):
            for x in range(40):
                if obs_count[y, x] >= 3 and init[y, x] not in (10, 5):
                    tau = 2.0
                    alpha = tau * pred[y, x]
                    posterior = counts[y, x] + alpha
                    pred[y, x] = posterior / posterior.sum()

    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred


def submit_active_round():
    """Find active round, run queries for regime detection, predict and submit."""
    TOKEN = ""
    token_file = BASE_DIR / ".token"
    if token_file.exists():
        TOKEN = token_file.read_text().strip()
    if not TOKEN:
        TOKEN = os.environ.get("AINM_TOKEN", "")
    if not TOKEN:
        print("No auth token")
        return

    session = requests.Session()
    session.cookies.set("access_token", TOKEN)
    session.headers["Authorization"] = f"Bearer {TOKEN}"
    BASE = "https://api.ainm.no"

    rounds = session.get(f"{BASE}/astar-island/rounds").json()
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round")
        return

    round_id = active["id"]
    rn = active["round_number"]
    details = session.get(f"{BASE}/astar-island/rounds/{round_id}").json()

    budget = session.get(f"{BASE}/astar-island/budget").json()
    remaining = budget["queries_max"] - budget["queries_used"]

    print(f"R{rn}: {remaining} queries available")

    round_dir = LOG_DIR / f"round{rn}"
    round_dir.mkdir(exist_ok=True, parents=True)

    # Phase 1: 1 scouting query per seed for regime detection (5 queries)
    # Phase 2: 9 repeat queries per seed on hottest viewport (45 queries)
    all_obs = {}

    for seed_idx in range(details['seeds_count']):
        init = np.array(details['initial_states'][seed_idx]['grid'])
        civ_mask = (init == 1) | (init == 2)
        civ_dist = distance_transform_cdt(~civ_mask, metric='taxicab') if civ_mask.any() else np.full(init.shape, 99)

        # Find viewport with most initial settlements visible
        best_score, best_vp = -1, (0, 0)
        for vy in range(0, 26, 3):
            for vx in range(0, 26, 3):
                has_s = (init[vy:vy + 15, vx:vx + 15] == 1).sum()
                near = ((civ_dist[vy:vy + 15, vx:vx + 15] >= 1) & (civ_dist[vy:vy + 15, vx:vx + 15] <= 4)).sum()
                score = 3 * has_s + near
                if score > best_score:
                    best_score = score
                    best_vp = (vx, vy)

        n_queries = min(10, remaining // (details['seeds_count'] - seed_idx))
        observations = []

        for q in range(n_queries):
            vx, vy = best_vp
            resp = session.post(f"{BASE}/astar-island/simulate", json={
                "round_id": round_id, "seed_index": seed_idx,
                "viewport_x": vx, "viewport_y": vy, "viewport_w": 15, "viewport_h": 15,
            })
            if resp.status_code == 200:
                observations.append({"grid": resp.json()["grid"], "viewport_x": vx, "viewport_y": vy})
                remaining -= 1
            elif resp.status_code == 429:
                break
            time.sleep(0.15)

        all_obs[seed_idx] = observations
        with open(round_dir / f"observations_seed{seed_idx}.json", "w") as f:
            json.dump(observations, f)

    # Detect regime from ALL observations combined
    all_combined = []
    for obs_list in all_obs.values():
        all_combined.extend(obs_list)

    regime = detect_regime_from_observations(details['initial_states'][0]['grid'], all_combined)
    print(f"Detected regime: {regime}")

    # Submit with regime-specific priors + empirical observations
    for seed_idx in range(details['seeds_count']):
        obs = all_obs.get(seed_idx, [])
        pred_val = predict(details['initial_states'][seed_idx]['grid'], regime=regime, observations=obs)

        for attempt in range(3):
            resp = session.post(f"{BASE}/astar-island/submit", json={
                "round_id": round_id, "seed_index": seed_idx, "prediction": pred_val.tolist(),
            })
            if resp.status_code == 200:
                print(f"  Seed {seed_idx}: accepted ({len(obs)} obs, regime={regime})")
                break
            time.sleep(2)
        time.sleep(0.3)

    print("Done")


if __name__ == "__main__":
    submit_active_round()

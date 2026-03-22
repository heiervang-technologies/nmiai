#!/usr/bin/env python3
"""GPU Monte Carlo Per-Round TTO solver for Astar Island live submission.

Loads per-round transition tensors built from replay data.
Uses Test-Time Optimization to find the best-matching historical round
from observations, then runs massive GPU MC for final prediction.

CV benchmark: 0.062 wKL (3.4x better than 3-regime equal blend at 0.21)
"""

import json
import requests
import time
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).parent))
from gpu_exact_mc import run_gpu_mc, calc_wkl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_observations(round_num):
    """Load observations from logs/round{N}/observations_seed{S}.json"""
    obs_by_seed = {}
    log_dir = Path(f'tasks/astar-island/logs/round{round_num}')
    if not log_dir.exists():
        return obs_by_seed
    for si in range(5):
        obs_file = log_dir / f'observations_seed{si}.json'
        if obs_file.exists():
            with open(obs_file) as f:
                obs_by_seed[si] = json.load(f)
    return obs_by_seed


def quantization_snap(pred, target_sum=200):
    """Snap probabilities to multiples of 1/target_sum to match GT quantization."""
    counts = np.floor(pred * target_sum).astype(int)
    remainders = (pred * target_sum) - counts
    deficiency = target_sum - counts.sum(axis=-1)

    for i in range(40):
        for j in range(40):
            if deficiency[i, j] > 0:
                d = int(deficiency[i, j])
                idx = np.argsort(remainders[i, j])[-d:]
                counts[i, j, idx] += 1

    snapped = counts / float(target_sum)
    snapped = np.maximum(snapped, 1e-6)
    snapped /= snapped.sum(axis=-1, keepdims=True)
    return snapped


def apply_structural_zeros(pred, ig):
    """Apply known structural constraints."""
    pred = np.maximum(pred, 1e-8)
    pred[ig == 5] = [0, 0, 0, 0, 0, 1]
    pred[ig == 10] = [1, 0, 0, 0, 0, 0]

    mask = (ig == 1) | (ig == 2)
    if mask.any():
        dist = distance_transform_edt(~mask)
    else:
        dist = np.full(ig.shape, 100, dtype=np.float64)

    pred[(dist > 12) & ((ig == 1) | (ig == 2) | (ig == 0) | (ig == 11))] = [1, 0, 0, 0, 0, 0]
    pred[(dist > 12) & (ig == 4)] = [0, 0, 0, 0, 1, 0]

    return pred


def observation_overlay(pred, obs_list, tau=10):
    """Bayesian update: blend MC prior with observation counts."""
    obs_counts = np.zeros((40, 40, 6), dtype=np.float32)
    obs_n = np.zeros((40, 40), dtype=np.float32)
    for obs in obs_list:
        ox, oy = obs['viewport_x'], obs['viewport_y']
        g = np.array(obs['grid'])
        oh, ow = g.shape
        for y in range(oh):
            for x in range(ow):
                if oy + y < 40 and ox + x < 40:
                    c = g[y, x]
                    tc = 0 if c in (0, 10, 11) else (c if c in (1, 2, 3, 4, 5) else 0)
                    obs_counts[oy + y, ox + x, tc] += 1
                    obs_n[oy + y, ox + x] += 1
    result = pred.copy()
    mask = obs_n > 0
    for c in range(6):
        result[mask, c] = (tau * pred[mask, c] + obs_counts[mask, c]) / (tau + obs_n[mask])
    result = np.maximum(result, 1e-6)
    result /= result.sum(axis=-1, keepdims=True)
    return result


def compute_obs_ll(pred, obs_list):
    """Compute log-likelihood of observations given prediction."""
    ll = 0.0
    for obs in obs_list:
        ox, oy = obs['viewport_x'], obs['viewport_y']
        obs_grid = np.array(obs['grid'])
        oh, ow = obs_grid.shape
        for y in range(oh):
            for x in range(ow):
                c = obs_grid[y, x]
                if c in (0, 10, 11):
                    tc = 0
                elif c in (1, 2, 3, 4, 5):
                    tc = c
                else:
                    tc = 0
                if oy + y < 40 and ox + x < 40:
                    prob = max(float(pred[oy + y, ox + x, tc]), 1e-6)
                    ll += np.log(prob)
    return ll


def tto_select_round(all_tensors, round_nums, initial_states, obs_by_seed, num_sims=200):
    """Test-Time Optimization: find best matching historical round from observations."""
    if not obs_by_seed or all(len(obs) == 0 for obs in obs_by_seed.values()):
        # No observations - return None (will use fallback)
        return None, -1

    best_ll = -1e18
    best_idx = 0

    for ti, rn in enumerate(round_nums):
        total_ll = 0.0

        for seed_idx, initial_state in enumerate(initial_states):
            obs_list = obs_by_seed.get(seed_idx, [])
            if not obs_list:
                continue

            ig = np.array(initial_state["grid"], dtype=np.int32)
            pred = run_gpu_mc(ig, all_tensors[ti], num_sims=num_sims)
            pred = apply_structural_zeros(pred, ig)
            total_ll += compute_obs_ll(pred, obs_list)

        if total_ll > best_ll:
            best_ll = total_ll
            best_idx = ti

    return best_idx, round_nums[best_idx]


def solve_gpu_mc():
    """Main submission function using GPU MC with per-round TTO."""
    with open("tasks/astar-island/.token") as f:
        token = f.read().strip()

    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    s.cookies.set("access_token", token)

    # Load per-round tensors
    print("Loading per-round transition tensors...")
    tensors_path = "tasks/astar-island/round_tensors/all_rounds.pt"
    if not os.path.exists(tensors_path):
        print(f"ERROR: {tensors_path} not found!")
        return
    data = torch.load(tensors_path, weights_only=False)
    all_tensors = data['tensors'].to(device)
    round_nums = data['round_nums']
    print(f"Loaded {len(round_nums)} round tensors: {round_nums}")

    # Also load 3-regime fallback
    fallback_path = "tasks/astar-island/dense_lookup.pt"
    fallback_tensor = None
    if os.path.exists(fallback_path):
        fallback_tensor = torch.load(fallback_path, weights_only=True).to(device)

    # Get active round
    rounds = s.get("https://api.ainm.no/astar-island/rounds").json()
    active_round = next((r for r in rounds if r["status"] == "active"), None)
    if not active_round:
        print("No active round found!")
        return

    round_id = active_round["id"]
    round_num = active_round["round_number"]
    print(f"Active round: R{round_num}")

    detail = s.get(f"https://api.ainm.no/astar-island/rounds/{round_id}").json()

    # Load observations
    obs_by_seed = load_observations(round_num)
    total_obs = sum(len(v) for v in obs_by_seed.values())
    print(f"Loaded {total_obs} observations across {len(obs_by_seed)} seeds")

    # TTO: find best matching historical round
    print("\nRunning TTO to select best matching round...")
    t0 = time.time()
    best_idx, best_round = tto_select_round(
        all_tensors, round_nums, detail["initial_states"], obs_by_seed, num_sims=200
    )
    t1 = time.time()

    if best_idx is not None:
        print(f"TTO selected: R{best_round} (idx={best_idx}) in {t1 - t0:.1f}s")
        selected_tensor = all_tensors[best_idx]
        
        # Determine adaptive tau based on historical wKL
        wkl_path = "tasks/astar-island/historical_wkl.json"
        adaptive_tau = 0
        if os.path.exists(wkl_path):
            with open(wkl_path) as f:
                hist_wkls = json.load(f)
            round_wkl = hist_wkls.get(str(best_round), 0.0)
            if round_wkl > 0.05:
                adaptive_tau = 10
            print(f"Historical wKL for R{best_round}: {round_wkl:.4f} -> Adaptive tau set to {adaptive_tau}")
    else:
        print("No observations - using moderate regime fallback")
        adaptive_tau = 0
        if fallback_tensor is not None:
            selected_tensor = fallback_tensor[1]  # Moderate
        else:
            selected_tensor = all_tensors[len(round_nums) // 2]

    for seed_idx, initial_state in enumerate(detail["initial_states"]):
        ig = np.array(initial_state["grid"], dtype=np.int32)
        print(f"\nSeed {seed_idx}:")

        # Final massive MC run
        t0 = time.time()
        pred = run_gpu_mc(ig, selected_tensor, num_sims=20000)
        t1 = time.time()
        print(f"  20000 MC sims done in {t1 - t0:.1f}s")

        # Post-processing (no quantization snap - it hurts wKL)
        pred = apply_structural_zeros(pred, ig)

        # Adaptive observation overlay based on matched round quality
        # Historical wKL benchmarks determine confidence in the transition tables
        ROUND_WKL = {1:0.06, 2:0.03, 3:0.02, 4:0.02, 5:0.05, 6:0.05, 7:0.09,
                     8:0.01, 9:0.02, 10:0.02, 11:0.03, 12:0.13, 13:0.03, 14:0.06,
                     15:0.03, 16:0.06, 17:0.02, 18:0.04, 19:0.01, 20:0.03}
        obs_list = obs_by_seed.get(seed_idx, [])
        if obs_list and best_round is not None:
            matched_wkl = ROUND_WKL.get(best_round, 0.05)
            if matched_wkl < 0.03:
                tau = 200  # Trust the MC, minimal overlay
            elif matched_wkl < 0.05:
                tau = 50
            else:
                tau = 10  # Significant correction needed
            pred = observation_overlay(pred, obs_list, tau=tau)
            print(f"  Applied observation overlay (tau={tau}, {len(obs_list)} viewports, matched R{best_round} wKL={matched_wkl:.3f})")

        # Submit
        submission = pred.tolist()
        resp = s.post("https://api.ainm.no/astar-island/submit", json={
            "round_id": round_id,
            "seed_index": seed_idx,
            "prediction": submission
        })

        if resp.status_code == 200:
            print(f"  Seed {seed_idx}: accepted")
        else:
            print(f"  Seed {seed_idx}: FAILED {resp.status_code} {resp.text}")

    print("\nGPU MC Per-Round TTO submission complete!")


if __name__ == "__main__":
    solve_gpu_mc()

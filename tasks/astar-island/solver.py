#!/usr/bin/env python3
"""Astar Island solver
Every API call and response is logged to disk under logs/
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import requests

BASE = "https://api.ainm.no"
ROUND_ID = None  # Auto-detect active round
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Auth
TOKEN = os.environ.get("AINM_TOKEN", "")
if not TOKEN:
    token_file = Path(__file__).parent / ".token"
    if token_file.exists():
        TOKEN = token_file.read_text().strip()

session = requests.Session()
if TOKEN:
    session.cookies.set("access_token", TOKEN)
    session.headers["Authorization"] = f"Bearer {TOKEN}"


def log_api_call(endpoint, method, request_data, response_data, status_code, elapsed_ms):
    """Log every API call to a JSONL file and individual JSON files."""
    timestamp = datetime.now(timezone.utc).isoformat()
    entry = {
        "timestamp": timestamp,
        "method": method,
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data,
        "status_code": status_code,
        "elapsed_ms": elapsed_ms,
    }

    # Append to master log
    with open(LOG_DIR / "api_calls.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Individual file with sequential numbering
    existing = list(LOG_DIR.glob("call_*.json"))
    call_num = len(existing) + 1
    with open(LOG_DIR / f"call_{call_num:04d}.json", "w") as f:
        json.dump(entry, f, indent=2)

    print(f"  [LOG] call_{call_num:04d} | {method} {endpoint} | {status_code} | {elapsed_ms:.0f}ms")


def api_get(endpoint):
    url = f"{BASE}{endpoint}"
    t0 = time.time()
    resp = session.get(url)
    elapsed = (time.time() - t0) * 1000
    data = resp.json() if resp.ok else {"error": resp.text}
    log_api_call(endpoint, "GET", None, data, resp.status_code, elapsed)
    return data


def api_post(endpoint, payload):
    url = f"{BASE}{endpoint}"
    t0 = time.time()
    resp = session.post(url, json=payload)
    elapsed = (time.time() - t0) * 1000
    data = resp.json() if resp.ok else {"error": resp.text, "status": resp.status_code}
    # For logging, don't save the full prediction tensor (huge), save shape instead
    log_payload = payload.copy()
    if "prediction" in log_payload:
        pred = log_payload["prediction"]
        log_payload["prediction"] = f"<tensor {len(pred)}x{len(pred[0])}x{len(pred[0][0])}>"
    log_api_call(endpoint, "POST", log_payload, data, resp.status_code, elapsed)
    return data, resp.status_code


def get_round_details():
    print("=== Fetching round details ===")
    return api_get(f"/astar-island/rounds/{ROUND_ID}")


def check_budget():
    print("=== Checking budget ===")
    return api_get("/astar-island/budget")


def simulate(seed_index, vx, vy, vw=15, vh=15):
    print(f"=== Simulate seed={seed_index} viewport=({vx},{vy},{vw},{vh}) ===")
    payload = {
        "round_id": ROUND_ID,
        "seed_index": seed_index,
        "viewport_x": vx,
        "viewport_y": vy,
        "viewport_w": vw,
        "viewport_h": vh,
    }
    data, status = api_post("/astar-island/simulate", payload)
    if status == 429:
        print("  BUDGET EXHAUSTED or RATE LIMITED - stopping")
    return data, status


def submit_prediction(seed_index, prediction):
    print(f"=== Submitting prediction for seed {seed_index} ===")
    payload = {
        "round_id": ROUND_ID,
        "seed_index": seed_index,
        "prediction": prediction.tolist(),
    }
    data, status = api_post("/astar-island/submit", payload)
    return data, status


def full_coverage_viewports():
    """3x3 grid of 15x15 viewports covering the full 40x40 map."""
    positions = [0, 13, 25]  # 0-14, 13-27, 25-39 = full coverage
    viewports = []
    for y in positions:
        for x in positions:
            viewports.append((x, y, 15, 15))
    return viewports


def cell_code_to_class(cell):
    """Map simulator cell codes to prediction class indices."""
    if cell in (0, 10, 11):
        return 0  # Empty (ocean, plains, empty)
    elif cell == 1:
        return 1  # Settlement
    elif cell == 2:
        return 2  # Port
    elif cell == 3:
        return 3  # Ruin
    elif cell == 4:
        return 4  # Forest
    elif cell == 5:
        return 5  # Mountain
    return 0


def compute_prior(height, width, initial_grid):
    """Compute informed Dirichlet prior mean and strength per cell.

    Returns:
        prior_mean: (H, W, 6) array of prior class probabilities
        tau: (H, W) array of prior strength (total pseudo-count mass)
    """
    init = np.array(initial_grid)
    prior_mean = np.zeros((height, width, 6), dtype=np.float64)
    tau = np.zeros((height, width), dtype=np.float64)

    # Precompute settlement and coast masks
    settlement_mask = (init == 1)
    port_mask = (init == 2)
    ocean_mask = (init == 10)
    mountain_mask = (init == 5)
    forest_mask = (init == 4)

    # Distance to nearest settlement or port (Manhattan approx via dilation)
    from scipy.ndimage import distance_transform_cdt
    civ_mask = settlement_mask | port_mask
    settlement_dist = distance_transform_cdt(~civ_mask, metric='taxicab')

    # Coast adjacency: non-ocean cell adjacent to ocean
    from scipy.ndimage import binary_dilation
    ocean_adjacent = binary_dilation(ocean_mask, np.ones((3, 3))) & ~ocean_mask

    # Forest interior vs edge: erode forest mask with cross kernel (4-neighbor)
    from scipy.ndimage import binary_erosion
    cross_kernel = np.array([[0,1,0],[1,1,1],[0,1,0]])
    forest_interior = binary_erosion(forest_mask, cross_kernel)

    # Empirical prior means from Round 1 data (n=11,250 cell observations)
    PRIOR_SETTLEMENT = np.array([0.3931, 0.4151, 0.0126, 0.0252, 0.1541, 0.00])
    PRIOR_PORT       = np.array([0.1250, 0.1250, 0.3750, 0.0000, 0.3750, 0.00])
    PRIOR_FOREST_INT = np.array([0.0663, 0.1615, 0.0122, 0.0130, 0.7469, 0.00])
    PRIOR_FOREST_EDGE= np.array([0.0663, 0.1615, 0.0122, 0.0130, 0.7469, 0.00])
    PRIOR_PLAINS     = np.array([0.7748, 0.1620, 0.0126, 0.0113, 0.0394, 0.00])
    PRIOR_COASTAL    = np.array([0.7748, 0.1620, 0.0126, 0.0113, 0.0394, 0.00])

    for y in range(height):
        for x in range(width):
            cell = init[y, x]
            dist = settlement_dist[y, x]
            is_coast = ocean_adjacent[y, x]

            # Static cells: hardcoded, high tau
            if cell == 5:  # Mountain
                prior_mean[y, x] = [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]
                tau[y, x] = 100.0
                continue
            if cell == 10:  # Ocean
                prior_mean[y, x] = [1.00, 0.00, 0.00, 0.00, 0.00, 0.00]
                tau[y, x] = 100.0
                continue

            # Assign prior mean based on initial type + features
            if cell == 1:  # Settlement
                prior_mean[y, x] = PRIOR_SETTLEMENT
            elif cell == 2:  # Port
                prior_mean[y, x] = PRIOR_PORT
            elif cell == 4:  # Forest
                if forest_interior[y, x]:
                    prior_mean[y, x] = PRIOR_FOREST_INT
                else:
                    prior_mean[y, x] = PRIOR_FOREST_EDGE
            elif is_coast:
                prior_mean[y, x] = PRIOR_COASTAL
            else:  # Plains/empty
                prior_mean[y, x] = PRIOR_PLAINS

            # Adjust prior for cells near settlements (more dynamic)
            if dist <= 1:
                prior_mean[y, x] = prior_mean[y, x] * 0.5 + PRIOR_SETTLEMENT * 0.5

            # Non-mountain cells: zero out mountain class
            prior_mean[y, x, 5] = 0.0
            # Non-coastal cells: suppress port class
            if not is_coast:
                prior_mean[y, x, 2] = 0.0

            # Renormalize prior mean
            s = prior_mean[y, x].sum()
            if s > 0:
                prior_mean[y, x] /= s

            # Assign tau based on zone (hot/warm/cold)
            if dist <= 2 or cell in (1, 2):
                tau[y, x] = 4.0  # hot
            elif dist <= 4 or is_coast or (cell == 4 and not forest_interior[y, x]):
                tau[y, x] = 2.0  # warm
            else:
                tau[y, x] = 0.75  # cold

    return prior_mean, tau


def build_prediction(height, width, initial_grid, observations):
    """Build probability distribution using informed Dirichlet priors.

    observations: list of (grid_2d, vx, vy) tuples, each from one simulate call
    """
    # Compute informed prior
    prior_mean, tau = compute_prior(height, width, initial_grid)

    # Count observations per cell per class
    counts = np.zeros((height, width, 6), dtype=np.float64)
    obs_count = np.zeros((height, width), dtype=np.int32)

    for obs_grid, vx, vy in observations:
        for dy, row in enumerate(obs_grid):
            for dx, cell in enumerate(row):
                y, x = vy + dy, vx + dx
                if 0 <= y < height and 0 <= x < width:
                    cls = cell_code_to_class(cell)
                    counts[y, x, cls] += 1
                    obs_count[y, x] += 1

    # Posterior = counts + tau * prior_mean
    # For static cells (tau=100), prior dominates completely
    # For observed cells, data blends with prior based on tau strength
    alpha = tau[:, :, np.newaxis] * prior_mean  # (H, W, 6) pseudo-counts from prior
    posterior = counts + alpha
    pred = posterior / posterior.sum(axis=2, keepdims=True)

    # For unobserved non-static cells, prediction is just the prior mean
    # (already handled: counts=0 so posterior = alpha, which normalizes to prior_mean)

    # Floor at 0.01 and renormalize (safety against KL infinity)
    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)

    return pred


def save_observations(seed_idx, observations):
    """Save raw observations to disk."""
    path = LOG_DIR / f"observations_seed{seed_idx}.json"
    serializable = []
    for grid, vx, vy in observations:
        serializable.append({"grid": grid, "viewport_x": vx, "viewport_y": vy})
    with open(path, "w") as f:
        json.dump(serializable, f)
    print(f"  Saved {len(observations)} observations to {path}")


def save_prediction(seed_idx, prediction):
    """Save prediction tensor to disk."""
    path = LOG_DIR / f"prediction_seed{seed_idx}.npy"
    np.save(path, prediction)
    print(f"  Saved prediction to {path}")


def find_active_round():
    """Find the currently active round."""
    rounds = api_get("/astar-island/rounds")
    for r in rounds:
        if r["status"] == "active":
            return r
    return None


def compute_hotspot_scores(initial_grid, height, width):
    """Score each 15x15 viewport position by expected information value."""
    init = np.array(initial_grid)
    from scipy.ndimage import distance_transform_cdt, binary_dilation

    civ_mask = (init == 1) | (init == 2)
    ocean_mask = (init == 10)
    mountain_mask = (init == 5)
    forest_mask = (init == 4)

    civ_dist = distance_transform_cdt(~civ_mask, metric='taxicab') if civ_mask.any() else np.full((height, width), 99)
    coast_mask = binary_dilation(ocean_mask, np.ones((3, 3))) & ~ocean_mask
    forest_edge = binary_dilation(forest_mask, np.ones((3, 3))) & ~forest_mask & ~ocean_mask & ~mountain_mask

    # Score each possible viewport position
    scores = {}
    for vy in range(0, height - 4):  # min viewport 5
        for vx in range(0, width - 4):
            vw = min(15, width - vx)
            vh = min(15, height - vy)
            region_civ = civ_mask[vy:vy+vh, vx:vx+vw].sum()
            region_near_civ = (civ_dist[vy:vy+vh, vx:vx+vw] <= 2).sum()
            region_coast = coast_mask[vy:vy+vh, vx:vx+vw].sum()
            region_forest_edge = forest_edge[vy:vy+vh, vx:vx+vw].sum()
            region_static = (ocean_mask[vy:vy+vh, vx:vx+vw] | mountain_mask[vy:vy+vh, vx:vx+vw]).sum()

            score = (1.0 * region_civ + 0.5 * region_near_civ +
                     0.3 * region_coast + 0.4 * region_forest_edge -
                     0.3 * region_static)
            scores[(vx, vy)] = score

    return scores


def select_viewports_adaptive(initial_grid, height, width, n_queries):
    """V2 adaptive strategy: reconnaissance + exploitation."""
    scores = compute_hotspot_scores(initial_grid, height, width)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    # Greedy selection: pick top viewports with some spacing
    selected = []
    used_cells = set()

    for (vx, vy), score in ranked:
        if len(selected) >= n_queries:
            break
        # Check overlap with already selected viewports
        new_cells = set()
        for dy in range(min(15, height - vy)):
            for dx in range(min(15, width - vx)):
                new_cells.add((vy + dy, vx + dx))
        overlap = len(new_cells & used_cells) / len(new_cells) if new_cells else 1.0

        # Recon phase: require low overlap for diversity. Exploit phase: allow repeats.
        if len(selected) < n_queries // 2 and overlap > 0.3:
            continue  # Skip overlapping viewports in recon phase
        if len(selected) >= n_queries // 2 and overlap > 0.95:
            continue  # Even in exploit, skip near-duplicates

        selected.append((vx, vy, min(15, width - vx), min(15, height - vy)))
        used_cells |= new_cells

    return selected


def main():
    global ROUND_ID

    if not TOKEN:
        print("ERROR: No auth token. Set AINM_TOKEN env var or create tasks/astar-island/.token file")
        sys.exit(1)

    # Find active round
    active = find_active_round()
    if not active:
        print("No active round found.")
        sys.exit(0)

    ROUND_ID = active["id"]
    print(f"Active round: {active['round_number']} (ID: {ROUND_ID})")
    print(f"Closes at: {active['closes_at']}")
    print(f"Weight: {active['round_weight']}")

    # Get round details with initial states
    details = get_round_details()
    details_path = LOG_DIR / f"round{active['round_number']}_details.json"
    with open(details_path, "w") as f:
        json.dump(details, f)

    height = details["map_height"]
    width = details["map_width"]
    seeds_count = details["seeds_count"]
    initial_states = details["initial_states"]

    print(f"Map: {width}x{height}, Seeds: {seeds_count}")

    # Check budget
    budget = check_budget()
    queries_remaining = budget.get("queries_max", 50) - budget.get("queries_used", 0)
    print(f"Budget: {budget.get('queries_used', 0)}/{budget.get('queries_max', 50)} used, {queries_remaining} remaining")

    if queries_remaining <= 0:
        print("No queries remaining!")
        sys.exit(0)

    # V2 Strategy: adaptive viewport selection per seed
    # Allocate queries: 10 per seed
    queries_per_seed = queries_remaining // seeds_count
    extra = queries_remaining % seeds_count

    all_observations = {i: [] for i in range(seeds_count)}
    queries_used = 0
    budget_exhausted = False

    for seed_idx in range(seeds_count):
        if budget_exhausted:
            break

        initial_grid = initial_states[seed_idx]["grid"]
        n_q = queries_per_seed + (1 if seed_idx < extra else 0)

        # Select viewports adaptively based on hotspot scores
        viewports = select_viewports_adaptive(initial_grid, height, width, n_q)

        print(f"\n{'='*40}")
        print(f"SEED {seed_idx} - {len(viewports)} adaptive viewports")
        print(f"{'='*40}")

        for vx, vy, vw, vh in viewports:
            data, status = simulate(seed_idx, vx, vy, vw, vh)
            if status != 200:
                print(f"  ERROR: status {status}, data: {data}")
                if status == 429:
                    print("  Budget exhausted!")
                    budget_exhausted = True
                    break
                continue

            queries_used += 1
            grid = data.get("grid", [])
            all_observations[seed_idx].append((grid, vx, vy))
            time.sleep(0.25)

        save_observations(seed_idx, all_observations[seed_idx])

    print(f"\nTotal queries used: {queries_used}")

    # Build and submit predictions for ALL seeds (even unobserved ones get prior)
    print(f"\n{'='*40}")
    print("BUILDING AND SUBMITTING PREDICTIONS")
    print(f"{'='*40}")

    for seed_idx in range(seeds_count):
        initial_grid = initial_states[seed_idx]["grid"]
        observations = all_observations[seed_idx]

        print(f"\nSeed {seed_idx}: {len(observations)} observations")
        prediction = build_prediction(height, width, initial_grid, observations)
        save_prediction(seed_idx, prediction)

        result, status = submit_prediction(seed_idx, prediction)
        print(f"  Submit result: status={status}, data={result}")
        time.sleep(0.25)

    print(f"\n=== DONE ===")
    print(f"Round {active['round_number']}: Submitted {seeds_count} seeds using {queries_used} queries")

    budget = check_budget()
    print(f"Final budget: {json.dumps(budget)}")


if __name__ == "__main__":
    main()

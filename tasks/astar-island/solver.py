#!/usr/bin/env python3
"""Astar Island solver - Round 1
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
ROUND_ID = "71451d74-be9f-471f-aacd-a41f3b68a9cd"
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


def main():
    if not TOKEN:
        print("ERROR: No auth token. Set AINM_TOKEN env var or create tasks/astar-island/.token file")
        sys.exit(1)

    # Load initial states
    details_path = Path(__file__).parent / "round1_details.json"
    if details_path.exists():
        with open(details_path) as f:
            details = json.load(f)
    else:
        details = get_round_details()
        with open(details_path, "w") as f:
            json.dump(details, f)

    height = details["map_height"]
    width = details["map_width"]
    seeds_count = details["seeds_count"]
    initial_states = details["initial_states"]

    print(f"Map: {width}x{height}, Seeds: {seeds_count}")
    print(f"Round closes at: {details['closes_at']}")

    # Check budget
    budget = check_budget()
    print(f"Budget: {json.dumps(budget)}")

    # Phase 1: Full coverage - 9 queries per seed, 45 total
    viewports = full_coverage_viewports()
    print(f"\nViewport plan: {len(viewports)} viewports per seed = {len(viewports) * seeds_count} total queries")

    all_observations = {i: [] for i in range(seeds_count)}
    queries_used = 0

    for seed_idx in range(seeds_count):
        print(f"\n{'='*40}")
        print(f"SEED {seed_idx} - querying {len(viewports)} viewports")
        print(f"{'='*40}")

        for vx, vy, vw, vh in viewports:
            data, status = simulate(seed_idx, vx, vy, vw, vh)
            if status != 200:
                print(f"  ERROR: status {status}, data: {data}")
                if status == 429:
                    print("  Budget exhausted! Moving to prediction phase.")
                    break
                continue

            queries_used += 1
            grid = data.get("grid", [])
            all_observations[seed_idx].append((grid, vx, vy))

            # Rate limit: max 5 req/sec, be safe with 250ms delay
            time.sleep(0.25)

        save_observations(seed_idx, all_observations[seed_idx])

        if status == 429:
            break

    print(f"\nTotal queries used: {queries_used}")

    # Phase 2: Build and submit predictions
    print(f"\n{'='*40}")
    print("BUILDING AND SUBMITTING PREDICTIONS")
    print(f"{'='*40}")

    for seed_idx in range(seeds_count):
        initial_grid = initial_states[seed_idx]["grid"]
        observations = all_observations[seed_idx]

        print(f"\nSeed {seed_idx}: {len(observations)} observations")
        prediction = build_prediction(height, width, initial_grid, observations)
        save_prediction(seed_idx, prediction)

        # Submit
        result, status = submit_prediction(seed_idx, prediction)
        print(f"  Submit result: status={status}, data={result}")
        time.sleep(0.25)

    print("\n=== DONE ===")
    print(f"Submitted predictions for {seeds_count} seeds using {queries_used} queries")

    # Final budget check
    budget = check_budget()
    print(f"Final budget: {json.dumps(budget)}")


if __name__ == "__main__":
    main()

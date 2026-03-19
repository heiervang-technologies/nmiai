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


def build_prediction(height, width, initial_grid, observations):
    """Build probability distribution from initial state + observations.

    observations: list of (grid_2d, vx, vy) tuples, each from one simulate call
    """
    # Count observations per cell per class
    # Classes: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
    counts = np.zeros((height, width, 6), dtype=np.float64)
    obs_count = np.zeros((height, width), dtype=np.int32)

    for obs_grid, vx, vy in observations:
        for dy, row in enumerate(obs_grid):
            for dx, cell in enumerate(row):
                y, x = vy + dy, vx + dx
                if 0 <= y < height and 0 <= x < width:
                    # Map cell codes to prediction classes
                    if cell == 10 or cell == 11 or cell == 0:
                        cls = 0  # Empty
                    elif cell == 1:
                        cls = 1  # Settlement
                    elif cell == 2:
                        cls = 2  # Port
                    elif cell == 3:
                        cls = 3  # Ruin
                    elif cell == 4:
                        cls = 4  # Forest
                    elif cell == 5:
                        cls = 5  # Mountain
                    else:
                        cls = 0  # Default to empty
                    counts[y, x, cls] += 1
                    obs_count[y, x] += 1

    # Build prediction
    pred = np.full((height, width, 6), 1.0 / 6)  # uniform prior

    # For cells with observations, use empirical distribution
    observed = obs_count > 0
    if observed.any():
        # Empirical frequencies with Laplace smoothing (add 0.5 pseudo-count per class)
        smoothed = counts.copy()
        smoothed[observed] += 0.5  # Laplace smoothing
        totals = smoothed.sum(axis=2, keepdims=True)
        pred[observed] = (smoothed / totals)[observed]

    # For static cells from initial state (even without observations)
    init = np.array(initial_grid)
    for y in range(height):
        for x in range(width):
            cell = init[y, x]
            if cell == 5:  # Mountain - static
                pred[y, x] = [0.01, 0.01, 0.01, 0.01, 0.01, 0.95]
            elif cell == 10:  # Ocean - static
                pred[y, x] = [0.95, 0.01, 0.01, 0.01, 0.01, 0.01]

    # Floor and renormalize
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

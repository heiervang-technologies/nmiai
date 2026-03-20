#!/usr/bin/env python3
"""Query runner - ONLY queries the simulator and saves observations.
Does NOT build or submit predictions. Run predictor.py separately for that.
Designed to be called by auto_watcher.sh
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
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
GT_DIR = Path(__file__).parent / "ground_truth"
GT_DIR.mkdir(exist_ok=True)

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
    with open(LOG_DIR / "api_calls.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


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
    log_payload = payload.copy()
    if "prediction" in log_payload:
        pred = log_payload["prediction"]
        log_payload["prediction"] = f"<tensor>"
    log_api_call(endpoint, "POST", log_payload, data, resp.status_code, elapsed)
    return data, resp.status_code


def full_coverage_viewports():
    """3x3 grid of 15x15 viewports for full 40x40 coverage."""
    positions = [0, 13, 25]
    return [(x, y, 15, 15) for y in positions for x in positions]


def fetch_ground_truth_for_completed():
    """Fetch and save ground truth for any completed rounds we don't have."""
    rounds = api_get("/astar-island/rounds")
    for r in rounds:
        if r["status"] != "completed":
            continue
        rn = r["round_number"]
        # Check if we already have it
        if (GT_DIR / f"round{rn}_seed0.json").exists():
            continue
        print(f"Fetching ground truth for round {rn}...")
        for seed in range(5):
            try:
                url = f"{BASE}/astar-island/analysis/{r['id']}/{seed}"
                resp = session.get(url)
                if resp.ok:
                    with open(GT_DIR / f"round{rn}_seed{seed}.json", "w") as f:
                        json.dump(resp.json(), f)
                time.sleep(0.25)
            except Exception as e:
                print(f"  Error fetching R{rn} seed {seed}: {e}")
        # Also save round details
        details_path = LOG_DIR / f"round{rn}_details.json"
        if not details_path.exists():
            try:
                details = api_get(f"/astar-island/rounds/{r['id']}")
                with open(details_path, "w") as f:
                    json.dump(details, f)
            except Exception:
                pass


def main():
    if not TOKEN:
        print("ERROR: No auth token.")
        sys.exit(1)

    # First, fetch any missing ground truth from completed rounds
    fetch_ground_truth_for_completed()

    # Find active round
    rounds = api_get("/astar-island/rounds")
    active = next((r for r in rounds if r["status"] == "active"), None)
    if not active:
        print("No active round.")
        return

    round_id = active["id"]
    round_num = active["round_number"]
    print(f"Active round: {round_num} (ID: {round_id})")
    print(f"Closes at: {active['closes_at']}, Weight: {active['round_weight']}")

    # Check budget
    budget_resp = session.get(f"{BASE}/astar-island/budget").json()
    queries_used = budget_resp.get("queries_used", 0)
    queries_max = budget_resp.get("queries_max", 50)
    remaining = queries_max - queries_used

    if remaining <= 0:
        print(f"No queries remaining ({queries_used}/{queries_max})")
        return

    print(f"Budget: {queries_used}/{queries_max}, {remaining} remaining")

    # Get round details and save
    details = api_get(f"/astar-island/rounds/{round_id}")
    round_dir = LOG_DIR / f"round{round_num}"
    round_dir.mkdir(exist_ok=True)
    with open(round_dir / "details.json", "w") as f:
        json.dump(details, f)
    # Also save to standard location
    with open(LOG_DIR / f"round{round_num}_details.json", "w") as f:
        json.dump(details, f)

    height = details["map_height"]
    width = details["map_width"]
    seeds_count = details["seeds_count"]
    initial_states = details["initial_states"]

    # Use full coverage: 9 queries per seed = 45, then 5 extra for repeat sampling
    viewports = full_coverage_viewports()
    queries_per_seed = remaining // seeds_count

    for seed_idx in range(seeds_count):
        seed_viewports = viewports[:queries_per_seed]
        print(f"\nSEED {seed_idx} - {len(seed_viewports)} viewports")

        observations = []
        for vx, vy, vw, vh in seed_viewports:
            payload = {
                "round_id": round_id,
                "seed_index": seed_idx,
                "viewport_x": vx, "viewport_y": vy,
                "viewport_w": vw, "viewport_h": vh,
            }
            data, status = api_post("/astar-island/simulate", payload)
            if status != 200:
                print(f"  ERROR: {status}")
                if status == 429:
                    break
                continue
            observations.append({
                "grid": data.get("grid", []),
                "viewport_x": vx, "viewport_y": vy,
            })
            time.sleep(0.25)

        # Save to per-round directory
        obs_path = round_dir / f"observations_seed{seed_idx}.json"
        with open(obs_path, "w") as f:
            json.dump(observations, f)
        print(f"  Saved {len(observations)} observations to {obs_path}")

    print(f"\nQuery runner done for round {round_num}.")


if __name__ == "__main__":
    main()

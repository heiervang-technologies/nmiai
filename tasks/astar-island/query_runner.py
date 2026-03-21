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
from scipy.ndimage import binary_dilation, distance_transform_cdt

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


def compute_hotspot_scores(initial_grid, height, width):
    init = np.array(initial_grid)
    civ_mask = (init == 1) | (init == 2)
    ocean_mask = init == 10
    mountain_mask = init == 5
    forest_mask = init == 4

    civ_dist = distance_transform_cdt(~civ_mask, metric="taxicab") if civ_mask.any() else np.full((height, width), 99)
    coast_mask = binary_dilation(ocean_mask, np.ones((3, 3), dtype=bool)) & ~ocean_mask
    forest_edge = binary_dilation(forest_mask, np.ones((3, 3), dtype=bool)) & ~forest_mask & ~ocean_mask & ~mountain_mask

    scores = {}
    for vy in range(0, height - 4):
        for vx in range(0, width - 4):
            vw = min(15, width - vx)
            vh = min(15, height - vy)
            region_civ = civ_mask[vy:vy+vh, vx:vx+vw].sum()
            region_near_civ = (civ_dist[vy:vy+vh, vx:vx+vw] <= 2).sum()
            region_coast = coast_mask[vy:vy+vh, vx:vx+vw].sum()
            region_forest_edge = forest_edge[vy:vy+vh, vx:vx+vw].sum()
            region_static = (ocean_mask[vy:vy+vh, vx:vx+vw] | mountain_mask[vy:vy+vh, vx:vx+vw]).sum()

            score = (
                1.0 * region_civ
                + 0.5 * region_near_civ
                + 0.3 * region_coast
                + 0.4 * region_forest_edge
                - 0.3 * region_static
            )
            scores[(vx, vy)] = float(score)
    return scores


def select_viewports_adaptive(initial_grid, height, width, n_queries):
    scores = compute_hotspot_scores(initial_grid, height, width)
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    if not ranked or n_queries <= 0:
        return []

    selected = []
    used_cells = set()
    explore_budget = max(1, int(np.ceil(n_queries * 0.6))) if n_queries > 1 else 1

    for (vx, vy), _ in ranked:
        if len(selected) >= explore_budget:
            break

        new_cells = set()
        for dy in range(min(15, height - vy)):
            for dx in range(min(15, width - vx)):
                new_cells.add((vy + dy, vx + dx))
        overlap = len(new_cells & used_cells) / len(new_cells) if new_cells else 1.0

        if len(selected) < max(1, explore_budget // 2) and overlap > 0.3:
            continue
        if len(selected) >= max(1, explore_budget // 2) and overlap > 0.95:
            continue

        selected.append((vx, vy, min(15, width - vx), min(15, height - vy)))
        used_cells |= new_cells

    if not selected:
        vx, vy = ranked[0][0]
        selected.append((vx, vy, min(15, width - vx), min(15, height - vy)))

    # Reserve the rest of the budget for repeated samples on the most dynamic windows.
    repeat_pool_size = min(max(1, explore_budget // 2), max(1, len(ranked)))
    repeat_pool = [
        (vx, vy, min(15, width - vx), min(15, height - vy))
        for (vx, vy), _ in ranked[:repeat_pool_size]
    ]
    idx = 0
    while len(selected) < n_queries:
        selected.append(repeat_pool[idx % len(repeat_pool)])
        idx += 1

    return selected[:n_queries]


def summarize_seed(initial_grid, height, width):
    init = np.array(initial_grid)
    civ_mask = (init == 1) | (init == 2)
    ocean_mask = init == 10
    mountain_mask = init == 5
    forest_mask = init == 4
    plains_mask = np.isin(init, [0, 11])
    coast_mask = binary_dilation(ocean_mask, np.ones((3, 3), dtype=bool)) & ~ocean_mask
    forest_edge = binary_dilation(forest_mask, np.ones((3, 3), dtype=bool)) & ~forest_mask & ~ocean_mask & ~mountain_mask
    civ_dist = distance_transform_cdt(~civ_mask, metric="taxicab") if civ_mask.any() else np.full((height, width), 99)
    hotspot_scores = compute_hotspot_scores(initial_grid, height, width)
    top_hotspots = sorted(hotspot_scores.values(), reverse=True)[:5]

    return {
        "settlements": int((init == 1).sum()),
        "ports": int((init == 2).sum()),
        "forest": int(forest_mask.sum()),
        "plains": int(plains_mask.sum()),
        "mountains": int(mountain_mask.sum()),
        "ocean": int(ocean_mask.sum()),
        "coast_land": int(coast_mask.sum()),
        "coastal_civ": int((coast_mask & civ_mask).sum()),
        "near_civ_land": int(((civ_dist <= 2) & ~ocean_mask & ~mountain_mask).sum()),
        "forest_edge": int(forest_edge.sum()),
        "dynamic_land": int((~ocean_mask & ~mountain_mask).sum()),
        "top_hotspots": [round(v, 2) for v in top_hotspots],
    }


def heuristic_seed_scores(initial_states, height, width):
    scores = []
    summaries = []
    for seed_state in initial_states:
        summary = summarize_seed(seed_state["grid"], height, width)
        summaries.append(summary)
        score = (
            3.5 * summary["settlements"]
            + 8.0 * summary["ports"]
            + 1.0 * summary["coastal_civ"]
            + 0.28 * summary["near_civ_land"]
            + 0.10 * summary["forest_edge"]
            + 0.06 * summary["coast_land"]
            + 0.03 * sum(summary["top_hotspots"])
        )
        scores.append(float(score))
    return summaries, np.array(scores, dtype=np.float64)


def allocate_queries_across_seeds(initial_states, height, width, remaining):
    """Allocate queries for REGIME DETECTION only.

    Strategy (from advisor + eval boss analysis):
    - Use exactly 1 diagnostic query per seed = 5 total
    - Each query targets the most settlement-rich viewport
    - Observations used ONLY for template weight updates, never cell-level
    - 5 queries saturate regime detection benefit; more add zero value
    """
    seed_summaries, heuristic_scores = heuristic_seed_scores(initial_states, height, width)
    seed_count = len(initial_states)

    # Cap at 1 query per seed for regime detection (5 total)
    # This is the optimal strategy per eval boss CV analysis
    REGIME_QUERIES_PER_SEED = 1
    allocation = np.full(seed_count, REGIME_QUERIES_PER_SEED, dtype=np.int32)

    ranked = [int(i) for i in np.argsort(-heuristic_scores)]

    plan = {
        "seed_summaries": seed_summaries,
        "heuristic_scores": [round(v, 3) for v in heuristic_scores.tolist()],
        "allocation": allocation.tolist(),
        "ranked_seeds": ranked,
        "strategy": {
            "mode": "regime_detection_only",
            "queries_per_seed": REGIME_QUERIES_PER_SEED,
            "total_queries": int(allocation.sum()),
            "rationale": "5 viewports saturate regime detection. More queries add zero value. No cell-level observation updates.",
        },
    }
    return allocation.tolist(), plan


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

    allocation, plan = allocate_queries_across_seeds(initial_states, height, width, remaining)
    plan_path = round_dir / "query_plan.json"
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)

    print("Seed allocation plan:")
    for seed_idx in range(seeds_count):
        summary = plan["seed_summaries"][seed_idx]
        print(
            f"  seed {seed_idx}: q={allocation[seed_idx]} alloc={allocation[seed_idx]} "
            f"sett={summary['settlements']} ports={summary['ports']} near_civ={summary['near_civ_land']} "
            f"coast_civ={summary['coastal_civ']} hotspots={summary['top_hotspots'][:3]}"
        )
    for seed_idx in range(seeds_count):
        seed_queries = allocation[seed_idx]
        if seed_queries <= 0:
            continue
        seed_viewports = select_viewports_adaptive(initial_states[seed_idx]["grid"], height, width, seed_queries)
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

#!/usr/bin/env python3
"""Query runner - ONLY queries the simulator and saves observations.
Does NOT build or submit predictions. Run predictor.py separately for that.
Designed to be called by auto_watcher.sh
"""

import json
import os
import re
import subprocess
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
GEMINI_BIN = os.environ.get("GEMINI_BIN", "gemini")

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

    selected = []
    used_cells = set()

    for (vx, vy), _ in ranked:
        if len(selected) >= n_queries:
            break

        new_cells = set()
        for dy in range(min(15, height - vy)):
            for dx in range(min(15, width - vx)):
                new_cells.add((vy + dy, vx + dx))
        overlap = len(new_cells & used_cells) / len(new_cells) if new_cells else 1.0

        if len(selected) < max(1, n_queries // 2) and overlap > 0.3:
            continue
        if len(selected) >= max(1, n_queries // 2) and overlap > 0.95:
            continue

        selected.append((vx, vy, min(15, width - vx), min(15, height - vy)))
        used_cells |= new_cells

    # If budget for this seed exceeds distinct good windows, repeat the hottest windows.
    idx = 0
    ranked_viewports = [(vx, vy, min(15, width - vx), min(15, height - vy)) for (vx, vy), _ in ranked[:max(1, n_queries)]]
    while len(selected) < n_queries and ranked_viewports:
        selected.append(ranked_viewports[idx % len(ranked_viewports)])
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
            3.0 * summary["settlements"]
            + 6.0 * summary["ports"]
            + 0.7 * summary["coastal_civ"]
            + 0.22 * summary["near_civ_land"]
            + 0.12 * summary["forest_edge"]
            + 0.08 * summary["coast_land"]
            + 0.02 * sum(summary["top_hotspots"])
        )
        scores.append(float(score))
    return summaries, np.array(scores, dtype=np.float64)


def extract_json_object(text):
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def query_gemini_seed_ranking(seed_summaries, remaining_queries):
    prompt = {
        "task": "Rank Astar Island seeds by expected information gain for simulator queries.",
        "goal": "Prefer seeds with many settlements/ports, strong coastal trade potential, large dynamic frontiers near civ, and high hotspot scores. Static ocean/mountain heavy maps are less valuable.",
        "remaining_queries": remaining_queries,
        "return_schema": {
            "ranked_seeds": [0, 1, 2, 3, 4],
            "weights": [0.2, 0.2, 0.2, 0.2, 0.2],
            "notes": ["short reason per seed order"]
        },
        "seed_summaries": [
            {"seed": idx, **summary}
            for idx, summary in enumerate(seed_summaries)
        ],
        "instruction": "Return only valid JSON. weights must be positive and sum to 1.0."
    }

    try:
        proc = subprocess.run(
            [
                GEMINI_BIN,
                "-p",
                json.dumps(prompt),
                "--model",
                os.environ.get("GEMINI_MODEL", "gemini-2.5-pro"),
                "--approval-mode",
                "plan",
            ],
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    if proc.returncode != 0:
        return None

    parsed = extract_json_object(proc.stdout)
    if not parsed:
        return None

    ranked = parsed.get("ranked_seeds")
    weights = parsed.get("weights")
    if not isinstance(ranked, list) or sorted(ranked) != list(range(len(seed_summaries))):
        return None
    if not isinstance(weights, list) or len(weights) != len(seed_summaries):
        return None
    weights = np.asarray(weights, dtype=np.float64)
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        return None
    weights = weights / weights.sum()
    return {
        "ranked_seeds": [int(v) for v in ranked],
        "weights": weights,
        "notes": parsed.get("notes", []),
        "raw": parsed,
    }


def allocate_queries_across_seeds(initial_states, height, width, remaining):
    seed_summaries, heuristic_scores = heuristic_seed_scores(initial_states, height, width)
    heuristic_weights = heuristic_scores / heuristic_scores.sum() if heuristic_scores.sum() > 0 else np.full(len(initial_states), 1.0 / len(initial_states))

    gemini_plan = query_gemini_seed_ranking(seed_summaries, remaining)
    if gemini_plan is not None:
        combined_weights = 0.6 * heuristic_weights + 0.4 * gemini_plan["weights"]
        ranked = gemini_plan["ranked_seeds"]
    else:
        combined_weights = heuristic_weights
        ranked = list(np.argsort(-combined_weights))

    combined_weights = combined_weights / combined_weights.sum()
    seed_count = len(initial_states)
    base = min(3, remaining // seed_count) if remaining >= seed_count else 0
    allocation = np.full(seed_count, base, dtype=np.int32)
    extra = int(remaining - allocation.sum())

    if extra > 0:
        fractional = combined_weights * extra
        allocation += np.floor(fractional).astype(np.int32)
        leftover = int(remaining - allocation.sum())
        remainders = fractional - np.floor(fractional)
        priority_order = sorted(range(seed_count), key=lambda i: (-remainders[i], ranked.index(i) if i in ranked else seed_count))
        for idx in priority_order[:leftover]:
            allocation[idx] += 1

    plan = {
        "seed_summaries": seed_summaries,
        "heuristic_scores": [round(v, 3) for v in heuristic_scores.tolist()],
        "heuristic_weights": [round(v, 4) for v in heuristic_weights.tolist()],
        "combined_weights": [round(v, 4) for v in combined_weights.tolist()],
        "allocation": allocation.tolist(),
        "ranked_seeds": ranked,
        "gemini": gemini_plan,
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
            f"  seed {seed_idx}: q={allocation[seed_idx]} combined_w={plan['combined_weights'][seed_idx]:.4f} "
            f"sett={summary['settlements']} ports={summary['ports']} near_civ={summary['near_civ_land']} "
            f"coast_civ={summary['coastal_civ']} hotspots={summary['top_hotspots'][:3]}"
        )
    if plan.get("gemini"):
        print(f"  Gemini ranked seeds: {plan['gemini']['ranked_seeds']}")
        if plan["gemini"].get("notes"):
            for note in plan["gemini"]["notes"][:5]:
                print(f"    note: {note}")

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

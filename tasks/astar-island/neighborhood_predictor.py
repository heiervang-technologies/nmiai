#!/usr/bin/env python3
"""Neighborhood-based cellular automaton predictor for Astar Island.

Builds a lookup table from ground truth data:
  key = (initial_type, n_settlement_neighbors, n_forest_neighbors, n_ocean_neighbors, dist_bin)
  value = averaged ground truth probability vector

Minimizes KL divergence by producing calibrated probabilities.
"""

import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
from scipy.ndimage import distance_transform_cdt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
GT_DIR = BASE_DIR / "ground_truth"
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

BASE_URL = "https://api.ainm.no"

# Class indices: 0=Empty, 1=Settlement, 2=Port, 3=Ruin, 4=Forest, 5=Mountain
N_CLASSES = 6

# Initial grid codes
OCEAN = 10
MOUNTAIN = 5
SETTLEMENT = 1
PORT = 2
FOREST = 4
PLAINS = 11
EMPTY = 0

# Static distributions
OCEAN_DIST = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
MOUNTAIN_DIST = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

# Map initial cell codes to categorical type for bucketing
def cell_to_type(code):
    """Map cell code to a type string for lookup table keys."""
    if code == SETTLEMENT:
        return "settlement"
    elif code == PORT:
        return "port"
    elif code == FOREST:
        return "forest"
    elif code == PLAINS:
        return "plains"
    elif code == EMPTY:
        return "empty"
    else:
        return None  # ocean/mountain are static


def dist_bin(d):
    """Bin distance to nearest settlement/port."""
    if d <= 1:
        return 0
    elif d <= 3:
        return 1
    elif d <= 6:
        return 2
    else:
        return 3


def extract_features(initial_grid):
    """Extract per-cell features from initial grid.

    Returns:
        types: (H, W) array of type strings (None for static)
        n_settle: (H, W) number of settlement/port neighbors
        n_forest: (H, W) number of forest neighbors
        n_ocean: (H, W) number of ocean neighbors
        dist_bins: (H, W) binned distance to nearest settlement/port
    """
    ig = np.array(initial_grid)
    H, W = ig.shape

    # Neighbor counts using convolution-like approach
    settle_mask = ((ig == SETTLEMENT) | (ig == PORT)).astype(np.int32)
    forest_mask = (ig == FOREST).astype(np.int32)
    ocean_mask = (ig == OCEAN).astype(np.int32)

    # Pad and sum 8-connected neighbors
    def count_neighbors(mask):
        padded = np.pad(mask, 1, mode='constant', constant_values=0)
        result = np.zeros_like(mask, dtype=np.int32)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                result += padded[1+dy:H+1+dy, 1+dx:W+1+dx]
        return result

    n_settle = count_neighbors(settle_mask)
    n_forest = count_neighbors(forest_mask)
    n_ocean = count_neighbors(ocean_mask)

    # Distance to nearest settlement/port
    civ_mask = (ig == SETTLEMENT) | (ig == PORT)
    if civ_mask.any():
        dist_map = distance_transform_cdt(~civ_mask, metric='taxicab')
    else:
        dist_map = np.full((H, W), 99, dtype=np.int32)

    dist_bins = np.vectorize(dist_bin)(dist_map)

    # Cell types
    types = np.empty((H, W), dtype=object)
    for y in range(H):
        for x in range(W):
            types[y, x] = cell_to_type(ig[y, x])

    return types, n_settle, n_forest, n_ocean, dist_bins


def build_lookup_table():
    """Build lookup table from all ground truth files.

    Returns dict mapping feature tuples to averaged probability vectors.
    Also returns backoff tables at coarser granularity.
    """
    # Collect: key -> list of probability vectors
    fine_data = defaultdict(list)     # (type, n_settle, n_forest, n_ocean, dist_bin)
    mid_data = defaultdict(list)      # (type, n_settle_bin, n_ocean>0, dist_bin)
    coarse_data = defaultdict(list)   # (type, dist_bin)
    type_data = defaultdict(list)     # (type,)

    gt_files = sorted(GT_DIR.glob("round*_seed*.json"))
    log.info(f"Loading {len(gt_files)} ground truth files")

    for path in gt_files:
        with open(path) as f:
            data = json.load(f)

        ig = np.array(data["initial_grid"])
        gt = np.array(data["ground_truth"])
        H, W = ig.shape

        types, n_settle, n_forest, n_ocean, dist_bins = extract_features(ig)

        for y in range(H):
            for x in range(W):
                t = types[y, x]
                if t is None:  # static cell
                    continue

                prob = gt[y, x]
                ns = int(n_settle[y, x])
                nf = int(n_forest[y, x])
                no = int(n_ocean[y, x])
                db = int(dist_bins[y, x])

                # Cap neighbor counts for sparsity reduction
                ns_cap = min(ns, 4)
                nf_cap = min(nf, 4)
                no_cap = min(no, 3)

                fine_key = (t, ns_cap, nf_cap, no_cap, db)
                fine_data[fine_key].append(prob)

                # Mid: bin settle neighbors more coarsely
                ns_bin = 0 if ns == 0 else (1 if ns <= 2 else 2)
                mid_key = (t, ns_bin, 1 if no > 0 else 0, db)
                mid_data[mid_key].append(prob)

                coarse_key = (t, db)
                coarse_data[coarse_key].append(prob)

                type_key = (t,)
                type_data[type_key].append(prob)

    # Build averaged tables
    MIN_SAMPLES_FINE = 10
    MIN_SAMPLES_MID = 10

    fine_table = {}
    for k, vecs in fine_data.items():
        if len(vecs) >= MIN_SAMPLES_FINE:
            fine_table[k] = np.mean(vecs, axis=0)

    mid_table = {}
    for k, vecs in mid_data.items():
        if len(vecs) >= MIN_SAMPLES_MID:
            mid_table[k] = np.mean(vecs, axis=0)

    coarse_table = {}
    for k, vecs in coarse_data.items():
        coarse_table[k] = np.mean(vecs, axis=0)

    type_table = {}
    for k, vecs in type_data.items():
        type_table[k] = np.mean(vecs, axis=0)

    log.info(f"Lookup table sizes: fine={len(fine_table)}, mid={len(mid_table)}, "
             f"coarse={len(coarse_table)}, type={len(type_table)}")

    # Log sample counts
    fine_counts = [len(v) for v in fine_data.values()]
    log.info(f"Fine bucket sample counts: min={min(fine_counts)}, max={max(fine_counts)}, "
             f"median={np.median(fine_counts):.0f}, total_buckets={len(fine_data)}")

    return fine_table, mid_table, coarse_table, type_table


def predict(initial_grid, fine_table=None, mid_table=None, coarse_table=None, type_table=None):
    """Predict 40x40x6 probability tensor from initial grid.

    Args:
        initial_grid: 40x40 list/array of cell codes
        *_table: prebuilt lookup tables (if None, will build from ground truth)

    Returns:
        40x40x6 numpy array of class probabilities
    """
    if fine_table is None:
        fine_table, mid_table, coarse_table, type_table = build_lookup_table()

    ig = np.array(initial_grid)
    H, W = ig.shape

    types, n_settle, n_forest, n_ocean, dist_bins = extract_features(ig)

    pred = np.zeros((H, W, N_CLASSES), dtype=np.float64)

    for y in range(H):
        for x in range(W):
            code = ig[y, x]

            # Static cells
            if code == OCEAN:
                pred[y, x] = OCEAN_DIST
                continue
            if code == MOUNTAIN:
                pred[y, x] = MOUNTAIN_DIST
                continue

            t = types[y, x]
            ns = min(int(n_settle[y, x]), 4)
            nf = min(int(n_forest[y, x]), 4)
            no = min(int(n_ocean[y, x]), 3)
            db = int(dist_bins[y, x])

            # Try fine lookup
            fine_key = (t, ns, nf, no, db)
            if fine_key in fine_table:
                pred[y, x] = fine_table[fine_key]
                continue

            # Backoff to mid
            ns_bin = 0 if ns == 0 else (1 if ns <= 2 else 2)
            mid_key = (t, ns_bin, 1 if no > 0 else 0, db)
            if mid_key in mid_table:
                pred[y, x] = mid_table[mid_key]
                continue

            # Backoff to coarse
            coarse_key = (t, db)
            if coarse_key in coarse_table:
                pred[y, x] = coarse_table[coarse_key]
                continue

            # Ultimate backoff: type only
            type_key = (t,)
            if type_key in type_table:
                pred[y, x] = type_table[type_key]
            else:
                # Uniform (should never happen)
                pred[y, x] = np.ones(N_CLASSES) / N_CLASSES

    # Floor at 0.01 and renormalize
    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)

    return pred


# ── API helpers ──────────────────────────────────────────────────────────────

def _get_token():
    token = os.environ.get("AINM_TOKEN", "")
    if not token:
        token_file = BASE_DIR / ".token"
        if token_file.exists():
            token = token_file.read_text().strip()
    return token


def _make_session():
    token = _get_token()
    s = requests.Session()
    if token:
        s.cookies.set("access_token", token)
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def log_api_call(endpoint, method, request_data, response_data, status_code, elapsed_ms):
    """Log API call to JSONL file."""
    from datetime import datetime, timezone
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "method": method,
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data,
        "status_code": status_code,
        "elapsed_ms": elapsed_ms,
    }
    with open(LOG_DIR / "api_calls.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    log.info(f"API {method} {endpoint} -> {status_code} ({elapsed_ms:.0f}ms)")


def api_get(session, endpoint):
    url = f"{BASE_URL}{endpoint}"
    t0 = time.time()
    resp = session.get(url)
    elapsed = (time.time() - t0) * 1000
    data = resp.json() if resp.ok else {"error": resp.text}
    log_api_call(endpoint, "GET", None, data, resp.status_code, elapsed)
    return data


def api_post(session, endpoint, payload):
    url = f"{BASE_URL}{endpoint}"
    t0 = time.time()
    resp = session.post(url, json=payload)
    elapsed = (time.time() - t0) * 1000
    data = resp.json() if resp.ok else {"error": resp.text, "status": resp.status_code}
    log_payload = payload.copy()
    if "prediction" in log_payload:
        pred = log_payload["prediction"]
        log_payload["prediction"] = f"<tensor {len(pred)}x{len(pred[0])}x{len(pred[0][0])}>"
    log_api_call(endpoint, "POST", log_payload, data, resp.status_code, elapsed)
    return data, resp.status_code


# ── Main: predict & submit ──────────────────────────────────────────────────

def main():
    token = _get_token()
    if not token:
        log.error("No auth token. Set AINM_TOKEN or create .token file")
        sys.exit(1)

    session = _make_session()

    # Build lookup tables once
    log.info("Building lookup tables from ground truth...")
    fine_table, mid_table, coarse_table, type_table = build_lookup_table()

    # Find active round
    rounds = api_get(session, "/astar-island/rounds")
    active = None
    for r in rounds:
        if r["status"] == "active":
            active = r
            break

    if not active:
        log.warning("No active round found")
        sys.exit(0)

    round_id = active["id"]
    round_num = active["round_number"]
    log.info(f"Active round: {round_num} (ID: {round_id}), closes: {active['closes_at']}")

    # Get round details
    details = api_get(session, f"/astar-island/rounds/{round_id}")
    height = details["map_height"]
    width = details["map_width"]
    seeds_count = details["seeds_count"]
    initial_states = details["initial_states"]
    log.info(f"Map: {width}x{height}, Seeds: {seeds_count}")

    # Predict and submit for each seed
    for seed_idx in range(seeds_count):
        initial_grid = initial_states[seed_idx]["grid"]
        log.info(f"Predicting seed {seed_idx}...")

        prediction = predict(initial_grid, fine_table, mid_table, coarse_table, type_table)

        # Validate
        sums = prediction.sum(axis=2)
        assert np.allclose(sums, 1.0, atol=1e-6), f"Probabilities don't sum to 1: {sums.min()}-{sums.max()}"
        assert prediction.min() >= 0.009, f"Floor violation: min={prediction.min()}"

        # Submit
        payload = {
            "round_id": round_id,
            "seed_index": seed_idx,
            "prediction": prediction.tolist(),
        }
        # Submit with retry on rate limit
        for attempt in range(3):
            result, status = api_post(session, "/astar-island/submit", payload)
            if status == 429:
                wait = 2 ** (attempt + 1)
                log.warning(f"Seed {seed_idx} rate limited, retrying in {wait}s...")
                time.sleep(wait)
                continue
            break
        log.info(f"Seed {seed_idx} submit: status={status}, result={result}")
        time.sleep(0.5)

    # Check budget after submissions
    budget = api_get(session, "/astar-island/budget")
    log.info(f"Final budget: {json.dumps(budget)}")
    log.info("Done!")


if __name__ == "__main__":
    main()

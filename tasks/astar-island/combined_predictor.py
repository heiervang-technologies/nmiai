#!/usr/bin/env python3
"""Combined predictor: continuous regime interpolation + empirical obs overlay.

R13 scored 87.9 with bucket priors + empirical obs.
This adds regime detection for ~25% improvement on top.

Usage: uv run python3 tasks/astar-island/combined_predictor.py
"""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import requests
from scipy.ndimage import binary_dilation, distance_transform_cdt

BASE_DIR = Path(__file__).parent
GT_DIR = BASE_DIR / "ground_truth"
LOG_DIR = BASE_DIR / "logs"

def cell_code_to_class(cell):
    if cell in (0, 10, 11): return 0
    if cell == 1: return 1
    if cell == 2: return 2
    if cell == 3: return 3
    if cell == 4: return 4
    if cell == 5: return 5
    return 0

def build_round_data():
    round_survival = {}
    round_priors = {}
    for f in sorted(GT_DIR.glob('round*_seed*.json')):
        with open(f) as fh:
            data = json.load(fh)
        if 'ground_truth' not in data: continue
        rn = int(f.stem.split('_')[0].replace('round', ''))
        sn = int(f.stem.split('_')[1].replace('seed', ''))
        init = np.array(data['initial_grid'])
        gt = np.array(data['ground_truth'])
        if sn == 0:
            settle = (init == 1)
            round_survival[rn] = gt[settle][:, 1].mean() if settle.any() else 0
        if rn not in round_priors:
            round_priors[rn] = defaultdict(lambda: {'sum': np.zeros(6), 'count': 0})
        civ = (init == 1) | (init == 2)
        ocean = (init == 10)
        oa = binary_dilation(ocean, np.ones((3,3))) & ~ocean
        cd = distance_transform_cdt(~civ, metric='taxicab') if civ.any() else np.full(init.shape, 99)
        for y in range(40):
            for x in range(40):
                cell = init[y, x]
                if cell in (10, 5): continue
                itype = 'F' if cell == 4 else ('S' if cell == 1 else ('P' if cell == 2 else 'X'))
                d = min(int(cd[y, x]), 15)
                key = (itype, d, bool(oa[y, x]))
                round_priors[rn][key]['sum'] += gt[y, x]
                round_priors[rn][key]['count'] += 1
    return round_survival, round_priors

_CACHE = {}
def get_data():
    if not _CACHE:
        s, p = build_round_data()
        _CACHE['survival'] = s
        _CACHE['priors'] = p
    return _CACHE['survival'], _CACHE['priors']

def detect_survival(initial_grid, observations):
    init = np.array(initial_grid)
    sp = set(zip(*np.where(init == 1)))
    survived = checked = 0
    for obs in observations:
        vx, vy = obs['viewport_x'], obs['viewport_y']
        for dy, row in enumerate(obs['grid']):
            for dx, cell in enumerate(row):
                if (vy+dy, vx+dx) in sp:
                    checked += 1
                    if cell == 1: survived += 1
    return survived / checked if checked > 0 else 0.3

def predict(initial_grid, observations=None):
    rs, rp = get_data()
    init = np.array(initial_grid)
    civ = (init == 1) | (init == 2)
    ocean = (init == 10)
    oa = binary_dilation(ocean, np.ones((3,3))) & ~ocean
    cd = distance_transform_cdt(~civ, metric='taxicab') if civ.any() else np.full(init.shape, 99)

    # Detect regime from observations
    det_surv = detect_survival(initial_grid, observations) if observations else 0.3
    
    # Weight rounds by proximity (Gaussian kernel)
    available = sorted(rp.keys())
    survivals = np.array([rs[rn] for rn in available])
    sigma = 0.08
    weights = np.exp(-(survivals - det_surv)**2 / (2*sigma**2))
    if weights.sum() > 0: weights /= weights.sum()
    else: weights = np.ones(len(available)) / len(available)

    pred = np.zeros((40, 40, 6))
    for y in range(40):
        for x in range(40):
            cell = init[y, x]
            if cell == 10: pred[y,x]=[1,0,0,0,0,0]; continue
            if cell == 5: pred[y,x]=[0,0,0,0,0,1]; continue
            itype = 'F' if cell == 4 else ('S' if cell == 1 else ('P' if cell == 2 else 'X'))
            d = min(int(cd[y, x]), 15)
            o = bool(oa[y, x])
            wp, tw = np.zeros(6), 0.0
            for i, rn in enumerate(available):
                for key in [(itype,d,o),(itype,d,not o)]:
                    e = rp[rn].get(key)
                    if e and e['count'] >= 3:
                        wp += weights[i] * (e['sum']/e['count'])
                        tw += weights[i]; break
            pred[y,x] = wp/tw if tw > 0 else np.ones(6)/6

    # Overlay empirical observations (tau=2 blend for cells with 3+ samples)
    if observations:
        counts = np.zeros((40,40,6))
        oc = np.zeros((40,40), dtype=int)
        for obs in observations:
            vx, vy = obs['viewport_x'], obs['viewport_y']
            for dy, row in enumerate(obs['grid']):
                for dx, cell in enumerate(row):
                    y, x = vy+dy, vx+dx
                    if 0<=y<40 and 0<=x<40:
                        counts[y,x,cell_code_to_class(cell)] += 1
                        oc[y,x] += 1
        for y in range(40):
            for x in range(40):
                if oc[y,x] >= 3 and init[y,x] not in (10,5):
                    alpha = 2.0 * pred[y,x]
                    post = counts[y,x] + alpha
                    pred[y,x] = post / post.sum()

    pred = np.maximum(pred, 0.01)
    pred /= pred.sum(axis=2, keepdims=True)
    return pred

def main():
    TOKEN = (BASE_DIR / ".token").read_text().strip() if (BASE_DIR / ".token").exists() else os.environ.get("AINM_TOKEN","")
    if not TOKEN: print("No token"); return
    s = requests.Session()
    s.cookies.set("access_token", TOKEN)
    s.headers["Authorization"] = f"Bearer {TOKEN}"
    BASE = "https://api.ainm.no"

    active = next((r for r in s.get(f"{BASE}/astar-island/rounds").json() if r["status"]=="active"), None)
    if not active: print("No active round"); return
    rid, rn = active["id"], active["round_number"]
    details = s.get(f"{BASE}/astar-island/rounds/{rid}").json()
    budget = s.get(f"{BASE}/astar-island/budget").json()
    remaining = budget["queries_max"] - budget["queries_used"]
    print(f"R{rn}: {remaining} queries available")

    rd = LOG_DIR / f"round{rn}"; rd.mkdir(exist_ok=True, parents=True)
    all_obs = {}
    qps = remaining // 5; extra = remaining % 5

    for si in range(5):
        init = np.array(details['initial_states'][si]['grid'])
        civ = (init==1)|(init==2)
        cd = distance_transform_cdt(~civ, metric='taxicab') if civ.any() else np.full(init.shape, 99)
        bs, bv = -1, (0,0)
        for vy in range(0,26,3):
            for vx in range(0,26,3):
                sc = 3*(init[vy:vy+15,vx:vx+15]==1).sum() + ((cd[vy:vy+15,vx:vx+15]>=1)&(cd[vy:vy+15,vx:vx+15]<=4)).sum()
                if sc > bs: bs=sc; bv=(vx,vy)
        
        # Load existing obs
        op = rd / f"observations_seed{si}.json"
        obs = json.loads(op.read_text()) if op.exists() else []
        
        nq = qps + (1 if si < extra else 0)
        for _ in range(nq):
            r = s.post(f"{BASE}/astar-island/simulate", json={"round_id":rid,"seed_index":si,"viewport_x":bv[0],"viewport_y":bv[1],"viewport_w":15,"viewport_h":15})
            if r.status_code == 200: obs.append({"grid":r.json()["grid"],"viewport_x":bv[0],"viewport_y":bv[1]})
            elif r.status_code == 429: break
            time.sleep(0.15)
        
        all_obs[si] = obs
        with open(op, 'w') as f: json.dump(obs, f)
        print(f"  Seed {si}: {len(obs)} obs")

    # Detect regime from all observations
    all_combined = [o for obs in all_obs.values() for o in obs]
    det = detect_survival(details['initial_states'][0]['grid'], all_combined)
    print(f"  Detected survival: {det:.3f}")

    for si in range(5):
        p = predict(details['initial_states'][si]['grid'], observations=all_obs[si])
        for _ in range(3):
            r = s.post(f"{BASE}/astar-island/submit", json={"round_id":rid,"seed_index":si,"prediction":p.tolist()})
            if r.status_code == 200: print(f"  Seed {si}: accepted"); break
            time.sleep(2)
        time.sleep(0.3)
    print("Done")

if __name__ == "__main__":
    main()

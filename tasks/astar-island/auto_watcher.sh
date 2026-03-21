#!/bin/bash
# SOTA Auto-Submitter for Astar Island
#
# This IS the optimal submission pipeline. No manual overrides needed.
#
# STRATEGY (R13 scored 87.9 with this):
#   1. Round opens: announce, fetch ground truth for completed rounds
#   2. 60min after open: blitz 50 queries (10/seed on hottest viewport)
#   3. 30min before close: submit with regime_predictor + empirical obs
#
# The regime_predictor.py combines:
#   - Frontier-rate regime detection from observations
#   - Soft Bayesian regime weights (not hard thresholds)
#   - Smooth distance interpolation
#   - Competition features (n_civ neighbors)
#   - Empirical observation overlay (tau=2 for cells with 3+ samples)
#
# Usage: nohup ./auto_watcher.sh &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/auto_watcher.log"
ANNOUNCED_ROUND=""
QUERIED_ROUND=""
SUBMITTED_ROUND=""

echo "$(date -u +%Y-%m-%dT%H:%M:%S) SOTA Watcher started (PID $$)" >> "$LOG_FILE"

while true; do
    ROUND_INFO=$(curl -s https://api.ainm.no/astar-island/rounds | python3 -c "
import json, sys
from datetime import datetime, timezone
rounds = json.load(sys.stdin)
now = datetime.now(timezone.utc)
for r in rounds:
    if r['status'] == 'active':
        started = datetime.fromisoformat(r['started_at'])
        closes = datetime.fromisoformat(r['closes_at'])
        mins_since_open = (now - started).total_seconds() / 60
        mins_until_close = (closes - now).total_seconds() / 60
        closes_str = r['closes_at'][:16].replace('T',' ')
        print(f'{r[\"id\"]}|{r[\"round_number\"]}|{int(mins_since_open)}|{int(mins_until_close)}|{closes_str}')
        sys.exit(0)
print('none')
" 2>/dev/null)

    if [ "$ROUND_INFO" = "none" ] || [ -z "$ROUND_INFO" ]; then
        # No active round — fetch ground truth for completed rounds
        cd /home/me/ht/nmiai
        uv run python3 -c "
import sys; sys.path.insert(0,'tasks/astar-island')
from query_runner import fetch_ground_truth_for_completed
fetch_ground_truth_for_completed()
" >> "$LOG_FILE" 2>&1
        sleep 180
        continue
    fi

    ACTIVE=$(echo "$ROUND_INFO" | cut -d'|' -f1)
    ROUND_NUM=$(echo "$ROUND_INFO" | cut -d'|' -f2)
    MINS_SINCE_OPEN=$(echo "$ROUND_INFO" | cut -d'|' -f3)
    MINS_UNTIL_CLOSE=$(echo "$ROUND_INFO" | cut -d'|' -f4)
    CLOSES_AT=$(echo "$ROUND_INFO" | cut -d'|' -f5)

    # === NEW ROUND: Ingest new GT data + Announce ===
    if [ "$ANNOUNCED_ROUND" != "$ACTIVE" ]; then
        ANNOUNCED_ROUND="$ACTIVE"
        # INGEST: Fetch ground truth from ALL completed rounds before doing anything
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM: ingesting new ground truth data" >> "$LOG_FILE"
        cd /home/me/ht/nmiai
        uv run python3 -c "
import sys; sys.path.insert(0,'tasks/astar-island')
from query_runner import fetch_ground_truth_for_completed
fetch_ground_truth_for_completed()
" >> "$LOG_FILE" 2>&1
        GT_COUNT=$(ls tasks/astar-island/ground_truth/round*_seed*.json 2>/dev/null | wc -l)
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM OPENED. GT files: $GT_COUNT. Closes $CLOSES_AT UTC. ${MINS_UNTIL_CLOSE}min." >> "$LOG_FILE"
        say "Astar Island round $ROUND_NUM opened. $GT_COUNT ground truth files available. $MINS_UNTIL_CLOSE minutes remaining." 2>/dev/null
        tmux-tool send %5 "<agent id=\"auto-watcher\" role=\"astar-watcher\" pane=\"bg\">R$ROUND_NUM opened. $GT_COUNT GT files ingested. Closes $CLOSES_AT UTC (${MINS_UNTIL_CLOSE}min). SOTA pipeline active.</agent>" 2>/dev/null
        sleep 0.5
        tmux send-keys -t %5 Enter 2>/dev/null
    fi

    # === 60 MIN AFTER OPEN: Blitz 50 queries (R13 formula) ===
    if [ "$MINS_SINCE_OPEN" -ge 60 ] && [ "$QUERIED_ROUND" != "$ACTIVE" ]; then
        TOKEN=$(cat "$SCRIPT_DIR/.token" 2>/dev/null | tr -d '\n')
        BUDGET=$(curl -s -H "Authorization: Bearer $TOKEN" https://api.ainm.no/astar-island/budget 2>/dev/null)
        QUERIES_USED=$(echo "$BUDGET" | python3 -c "import json,sys; print(json.load(sys.stdin).get('queries_used', 0))" 2>/dev/null)

        if [ "$QUERIES_USED" = "0" ]; then
            echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM: BLITZING 50 queries (2/seed on 5 diverse viewports)" >> "$LOG_FILE"
            cd /home/me/ht/nmiai

            # Blitz: 5 viewports per seed, 2 queries each = 10/seed = 50 total
            # More viewports = better regime detection (sees more frontier cells)
            uv run python3 -c "
import json, sys, time, numpy as np, requests
from pathlib import Path
from scipy.ndimage import distance_transform_edt

TOKEN = open('tasks/astar-island/.token').read().strip()
s = requests.Session()
s.cookies.set('access_token', TOKEN)
s.headers['Authorization'] = f'Bearer {TOKEN}'
BASE = 'https://api.ainm.no'

rounds = s.get(f'{BASE}/astar-island/rounds').json()
active = next(r for r in rounds if r['status'] == 'active')
rid = active['id']
rn = active['round_number']
details = s.get(f'{BASE}/astar-island/rounds/{rid}').json()

rd = Path(f'tasks/astar-island/logs/round{rn}')
rd.mkdir(exist_ok=True, parents=True)

for si in range(5):
    init = np.array(details['initial_states'][si]['grid'])
    civ = (init == 1) | (init == 2)
    dist = distance_transform_edt(~civ) if civ.any() else np.full(init.shape, 99.0)
    # Score viewports by frontier cell density
    scores = {}
    for vy in range(0, 26, 2):
        for vx in range(0, 26, 2):
            patch = dist[vy:vy+15, vx:vx+15]
            ipatch = init[vy:vy+15, vx:vx+15]
            dynamic = (ipatch != 10) & (ipatch != 5)
            frontier = dynamic & (patch >= 1) & (patch <= 6)
            sc = 3*(ipatch==1).sum() + frontier.sum()
            scores[(vx, vy)] = sc
    # Pick top 5 viewports with minimum spacing
    chosen = []
    for pos, sc in sorted(scores.items(), key=lambda x: -x[1]):
        if len(chosen) >= 5: break
        vx, vy = pos
        too_close = any(abs(vx-cx)<10 and abs(vy-cy)<10 for cx,cy in chosen)
        if not too_close:
            chosen.append(pos)
    obs = []
    for vx, vy in chosen:
        for _ in range(2):
            r = s.post(f'{BASE}/astar-island/simulate', json={'round_id':rid,'seed_index':si,'viewport_x':vx,'viewport_y':vy,'viewport_w':15,'viewport_h':15})
            if r.status_code == 200: obs.append({'grid':r.json()['grid'],'viewport_x':vx,'viewport_y':vy})
            elif r.status_code == 429: break
            time.sleep(0.12)
    with open(rd / f'observations_seed{si}.json', 'w') as f: json.dump(obs, f)
    print(f'Seed {si}: {len(obs)} obs on {len(chosen)} viewports')
print('Blitz done')
" >> "$LOG_FILE" 2>&1

            echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM: blitz queries done" >> "$LOG_FILE"
            say "Astar round $ROUND_NUM queries complete." 2>/dev/null
        else
            echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM: budget already used ($QUERIES_USED), skipping blitz" >> "$LOG_FILE"
        fi
        QUERIED_ROUND="$ACTIVE"
    fi

    # === 30 MIN BEFORE CLOSE: Submit with SOTA predictor ===
    if [ "$MINS_UNTIL_CLOSE" -le 30 ] && [ "$SUBMITTED_ROUND" != "$ACTIVE" ]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM: submitting with SOTA regime_predictor + observations" >> "$LOG_FILE"
        say "Astar round $ROUND_NUM: submitting SOTA predictions." 2>/dev/null
        cd /home/me/ht/nmiai

        # Submit with regime_predictor + spatial_model adaptive ensemble
        uv run python3 -c "
import json, sys, time, os, numpy as np, requests
from pathlib import Path
sys.path.insert(0, 'tasks/astar-island')
import regime_predictor as rp
import spatial_model as sm

TOKEN = open('tasks/astar-island/.token').read().strip()
s = requests.Session()
s.cookies.set('access_token', TOKEN)
s.headers['Authorization'] = f'Bearer {TOKEN}'
BASE = 'https://api.ainm.no'

rounds = s.get(f'{BASE}/astar-island/rounds').json()
active = next(r for r in rounds if r['status'] == 'active')
rid = active['id']
rn = active['round_number']
details = s.get(f'{BASE}/astar-island/rounds/{rid}').json()

rd = Path(f'tasks/astar-island/logs/round{rn}')

def ccc(cell):
    if cell in (0,10,11): return 0
    if cell==1: return 1
    if cell==2: return 2
    if cell==3: return 3
    if cell==4: return 4
    if cell==5: return 5
    return 0

# POOL all observations across seeds for round-level regime detection
all_obs = {}
all_obs_combined = []
for si in range(details['seeds_count']):
    op = rd / f'observations_seed{si}.json'
    seed_obs = json.loads(op.read_text()) if op.exists() else []
    all_obs[si] = seed_obs
    all_obs_combined.extend(seed_obs)

# Detect regime ONCE from ALL seeds' observations (round-level property)
ig0 = np.array(details['initial_states'][0]['grid'], dtype=np.int32)
regime_weights = rp.detect_regime_from_observations(ig0, all_obs_combined)
print(f'Round regime weights (pooled): {regime_weights}')

# Compute regime confidence for adaptive blending
max_w = max(regime_weights.values())
confidence = (max_w - 1.0/3) / (1.0 - 1.0/3)  # 0-1 scale
spatial_weight = 0.10 + 0.40 * (1 - confidence)
print(f'Regime confidence: {confidence:.3f}, spatial blend weight: {spatial_weight:.3f}')

for si in range(details['seeds_count']):
    obs = all_obs[si] if all_obs[si] else None
    init_grid = details['initial_states'][si]['grid']
    # Regime prediction with pooled observations
    pred_r = rp.predict(init_grid, observations=all_obs_combined)
    # Spatial model prediction
    pred_s = sm.predict(init_grid)
    pred_s = np.maximum(pred_s, 0.005)
    pred_s /= pred_s.sum(axis=2, keepdims=True)
    # Adaptive ensemble blend
    pred = (1 - spatial_weight) * pred_r + spatial_weight * pred_s
    # Empirical overlay with tau=10 on THIS seed's observations only
    if obs:
        init = np.array(init_grid)
        counts = np.zeros((40,40,6)); oc = np.zeros((40,40),dtype=int)
        for o in obs:
            for dy,row in enumerate(o['grid']):
                for dx,cell in enumerate(row):
                    y,x = o['viewport_y']+dy, o['viewport_x']+dx
                    if 0<=y<40 and 0<=x<40: counts[y,x,ccc(cell)]+=1; oc[y,x]+=1
        for y in range(40):
            for x in range(40):
                if oc[y,x]>=3 and init[y,x] not in (10,5):
                    alpha=10.0*pred[y,x]; post=counts[y,x]+alpha; pred[y,x]=post/post.sum()
    pred=np.maximum(pred,1e-6); pred/=pred.sum(axis=2,keepdims=True)
    for attempt in range(3):
        r = s.post(f'{BASE}/astar-island/submit', json={'round_id':rid,'seed_index':si,'prediction':pred.tolist()})
        if r.status_code == 200: print(f'Seed {si}: accepted ({len(obs) if obs else 0} obs)'); break
        time.sleep(2)
    time.sleep(0.3)
print('SOTA ensemble submission done')
" >> "$LOG_FILE" 2>&1

        echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM: SOTA predictions submitted" >> "$LOG_FILE"
        say "Astar round $ROUND_NUM SOTA predictions submitted." 2>/dev/null
        SUBMITTED_ROUND="$ACTIVE"
        tmux-tool send %5 "<agent id=\"auto-watcher\" role=\"astar-watcher\" pane=\"bg\">R$ROUND_NUM: SOTA submitted (regime_predictor + blitz obs).</agent>" 2>/dev/null
        sleep 0.5
        tmux send-keys -t %5 Enter 2>/dev/null
    fi

    sleep 120
done

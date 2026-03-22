#!/bin/bash
# SAFE-BEST Auto-Submitter for Astar Island
# Pipeline: regime_predictor with per-seed observations, tau=20, 3vp x 3q
# DO NOT ADD UNET OR OTHER UNVALIDATED PREDICTORS

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/auto_watcher.log"
ANNOUNCED_ROUND=""
QUERIED_ROUND=""
SUBMITTED_ROUND=""

echo "$(date -u +%Y-%m-%dT%H:%M:%S) SAFE-BEST Watcher started (PID $$)" >> "$LOG_FILE"

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
        mins_since = int((now - started).total_seconds() / 60)
        mins_left = int((closes - now).total_seconds() / 60)
        closes_str = r['closes_at'][:16].replace('T',' ')
        print(f'{r[\"id\"]}|{r[\"round_number\"]}|{mins_since}|{mins_left}|{closes_str}')
        sys.exit(0)
print('none')
" 2>/dev/null)

    if [ "$ROUND_INFO" = "none" ] || [ -z "$ROUND_INFO" ]; then
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
    MINS_SINCE=$(echo "$ROUND_INFO" | cut -d'|' -f3)
    MINS_LEFT=$(echo "$ROUND_INFO" | cut -d'|' -f4)
    CLOSES_AT=$(echo "$ROUND_INFO" | cut -d'|' -f5)

    # === NEW ROUND: Ingest GT ===
    if [ "$ANNOUNCED_ROUND" != "$ACTIVE" ]; then
        ANNOUNCED_ROUND="$ACTIVE"
        cd /home/me/ht/nmiai
        uv run python3 -c "
import sys; sys.path.insert(0,'tasks/astar-island')
from query_runner import fetch_ground_truth_for_completed
fetch_ground_truth_for_completed()
" >> "$LOG_FILE" 2>&1
        GT_COUNT=$(ls tasks/astar-island/ground_truth/round*_seed*.json 2>/dev/null | wc -l)
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) R$ROUND_NUM opened. GT=$GT_COUNT. Closes $CLOSES_AT. ${MINS_LEFT}min." >> "$LOG_FILE"
        say "Astar round $ROUND_NUM opened. $MINS_LEFT minutes." 2>/dev/null
        tmux-tool send %5 "<agent id=\"auto-watcher\" role=\"astar-watcher\" pane=\"bg\">R$ROUND_NUM opened. GT=$GT_COUNT. ${MINS_LEFT}min.</agent>" 2>/dev/null
        sleep 0.5; tmux send-keys -t %5 Enter 2>/dev/null
    fi

    # === 60 MIN: Blitz 3vp x 3q ===
    if [ "$MINS_SINCE" -ge 60 ] && [ "$QUERIED_ROUND" != "$ACTIVE" ]; then
        TOKEN=$(cat "$SCRIPT_DIR/.token" 2>/dev/null | tr -d '\n')
        BUDGET=$(curl -s -H "Authorization: Bearer $TOKEN" https://api.ainm.no/astar-island/budget 2>/dev/null)
        QUSED=$(echo "$BUDGET" | python3 -c "import json,sys; print(json.load(sys.stdin).get('queries_used',0))" 2>/dev/null)
        if [ "$QUSED" = "0" ]; then
            echo "$(date -u +%Y-%m-%dT%H:%M:%S) R$ROUND_NUM: blitzing 3vp x 3q" >> "$LOG_FILE"
            cd /home/me/ht/nmiai
            uv run python3 -c "
import json,sys,time,numpy as np,requests
from pathlib import Path
from scipy.ndimage import distance_transform_cdt
TOKEN=open('tasks/astar-island/.token').read().strip()
s=requests.Session(); s.cookies.set('access_token',TOKEN); s.headers['Authorization']=f'Bearer {TOKEN}'
BASE='https://api.ainm.no'
rounds=s.get(f'{BASE}/astar-island/rounds').json()
active=next(r for r in rounds if r['status']=='active')
rid=active['id']; rn=active['round_number']
details=s.get(f'{BASE}/astar-island/rounds/{rid}').json()
rd=Path(f'tasks/astar-island/logs/round{rn}'); rd.mkdir(exist_ok=True,parents=True)
for si in range(5):
    init=np.array(details['initial_states'][si]['grid'])
    civ=(init==1)|(init==2)
    cd=distance_transform_cdt(~civ,metric='taxicab') if civ.any() else np.full(init.shape,99)
    ranked=[]
    for vy in range(0,26,3):
        for vx in range(0,26,3):
            sc=3*(init[vy:vy+15,vx:vx+15]==1).sum()+((cd[vy:vy+15,vx:vx+15]>=1)&(cd[vy:vy+15,vx:vx+15]<=4)).sum()
            ranked.append((sc,vx,vy))
    ranked.sort(reverse=True)
    vps=[]; used=set()
    for sc,vx,vy in ranked:
        cells=set((vy+dy,vx+dx) for dy in range(min(15,40-vy)) for dx in range(min(15,40-vx)))
        if len(cells&used)/len(cells)<0.3: vps.append((vx,vy)); used|=cells
        if len(vps)>=3: break
    obs=[]
    for vx,vy in vps:
        for _ in range(3):
            r=s.post(f'{BASE}/astar-island/simulate',json={'round_id':rid,'seed_index':si,'viewport_x':vx,'viewport_y':vy,'viewport_w':15,'viewport_h':15})
            if r.status_code==200: obs.append({'grid':r.json()['grid'],'viewport_x':vx,'viewport_y':vy})
            elif r.status_code==429: break
            time.sleep(0.12)
    with open(rd/f'observations_seed{si}.json','w') as f: json.dump(obs,f)
    print(f'Seed {si}: {len(obs)} obs on {len(vps)} vps')
print('Blitz done')
" >> "$LOG_FILE" 2>&1
            echo "$(date -u +%Y-%m-%dT%H:%M:%S) R$ROUND_NUM: blitz done" >> "$LOG_FILE"
        fi
        QUERIED_ROUND="$ACTIVE"
    fi

    # === 30 MIN: Submit gpu_mc_solver ===
    if [ "$MINS_LEFT" -le 30 ] && [ "$SUBMITTED_ROUND" != "$ACTIVE" ]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) R$ROUND_NUM: submitting gpu_mc_solver" >> "$LOG_FILE"
        say "Astar round $ROUND_NUM submitting using GPU MC model." 2>/dev/null
        cd /home/me/ht/nmiai
        uv run python3 tasks/astar-island/gpu_mc_solver.py >> "$LOG_FILE" 2>&1
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) R$ROUND_NUM: submitted" >> "$LOG_FILE"
        say "Astar round $ROUND_NUM submitted." 2>/dev/null
        SUBMITTED_ROUND="$ACTIVE"
    fi

    sleep 120
done

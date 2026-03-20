#!/bin/bash
# Daemon that pokes the astar-island master agent every 30 minutes
# to keep iterating through the night.
# Usage: nohup ./daemon.sh &

MY_PANE="%3"
LOG="/home/me/ht/nmiai/tasks/astar-island/logs/daemon.log"

echo "$(date -u +%Y-%m-%dT%H:%M:%S) Daemon started (PID $$)" >> "$LOG"

cycle=0
while true; do
    cycle=$((cycle + 1))
    NOW=$(date -u +%Y-%m-%dT%H:%M:%S)

    # Get round status
    ROUND_INFO=$(curl -s https://api.ainm.no/astar-island/rounds | python3 -c "
import json, sys
from datetime import datetime, timezone
now = datetime.now(timezone.utc)
rounds = json.load(sys.stdin)
for r in sorted(rounds, key=lambda x: x['round_number']):
    if r['status'] == 'active':
        closes = datetime.fromisoformat(r['closes_at'])
        started = datetime.fromisoformat(r['started_at'])
        mins_left = int((closes - now).total_seconds() / 60)
        mins_since = int((now - started).total_seconds() / 60)
        print(f'ACTIVE|R{r[\"round_number\"]}|{mins_left}|{mins_since}|{r[\"round_weight\"]:.2f}')
        sys.exit(0)
print('NONE')
" 2>/dev/null)

    TOKEN=$(cat /home/me/ht/nmiai/tasks/astar-island/.token 2>/dev/null | tr -d '\n')
    BUDGET=$(curl -s -H "Authorization: Bearer $TOKEN" https://api.ainm.no/astar-island/budget 2>/dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'{d[\"queries_used\"]}/{d[\"queries_max\"]}')" 2>/dev/null)
    SEEDS=$(curl -s -H "Authorization: Bearer $TOKEN" https://api.ainm.no/astar-island/my-rounds 2>/dev/null | python3 -c "
import json,sys
for r in sorted(json.load(sys.stdin), key=lambda x: x['round_number']):
    if r['seeds_submitted'] > 0 and r['round_score'] is None:
        print(f'R{r[\"round_number\"]}:submitted')
    elif r['round_score'] is not None:
        pass  # already scored
" 2>/dev/null)

    if [ "$ROUND_INFO" = "NONE" ]; then
        MSG="DAEMON CYCLE $cycle ($NOW): No active round. TASK: iterate prediction models, fetch ground truth, improve CV scores. Run eval_system.py on any new approaches. Target: beat 0.089 CV wKL."
    else
        STATUS=$(echo "$ROUND_INFO" | cut -d'|' -f1)
        RN=$(echo "$ROUND_INFO" | cut -d'|' -f2)
        MINS_LEFT=$(echo "$ROUND_INFO" | cut -d'|' -f3)
        MINS_SINCE=$(echo "$ROUND_INFO" | cut -d'|' -f4)
        WEIGHT=$(echo "$ROUND_INFO" | cut -d'|' -f5)

        if [ "$MINS_SINCE" -lt 60 ]; then
            MSG="DAEMON CYCLE $cycle ($NOW): $RN active, ${MINS_LEFT}min left, opened ${MINS_SINCE}min ago. Budget: $BUDGET. TASK: PREPARE for probing. Analyze initial states, plan query allocation for regime detection. Do NOT query yet (wait for 1h mark)."
        elif [ "$MINS_LEFT" -gt 30 ]; then
            MSG="DAEMON CYCLE $cycle ($NOW): $RN active, ${MINS_LEFT}min left. Budget: $BUDGET. TASK: If budget unused, RUN QUERIES NOW for regime detection (8-12 scouting queries on settlement-heavy viewports, then repeat hot windows). Then SUBMIT predictions using template_predictor with observations."
        else
            MSG="DAEMON CYCLE $cycle ($NOW): $RN active, ${MINS_LEFT}min left. Budget: $BUDGET. TASK: SUBMIT NOW if not already submitted. Use template_predictor. Time is running out!"
        fi
    fi

    echo "$NOW $MSG" >> "$LOG"

    # Send to my pane
    tmux-tool send "$MY_PANE" "$MSG"
    sleep 0.5
    tmux send-keys -t "$MY_PANE" Enter
    sleep 0.3
    tmux send-keys -t "$MY_PANE" Enter

    # Wait 30 minutes
    sleep 1800
done

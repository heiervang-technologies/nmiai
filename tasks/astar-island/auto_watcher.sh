#!/bin/bash
# Background watcher: checks for new rounds every 3 minutes, runs solver
# Usage: nohup ./auto_watcher.sh &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/auto_watcher.log"

echo "$(date -u +%Y-%m-%dT%H:%M:%S) Watcher started (PID $$)" >> "$LOG_FILE"

while true; do
    # Check for active round
    ACTIVE=$(curl -s https://api.ainm.no/astar-island/rounds | python3 -c "
import json, sys
rounds = json.load(sys.stdin)
for r in rounds:
    if r['status'] == 'active':
        print(r['id'])
        sys.exit(0)
" 2>/dev/null)

    if [ -z "$ACTIVE" ]; then
        sleep 180  # No active round, check again in 3 min
        continue
    fi

    # Check if we already queried this round
    TOKEN=$(cat "$SCRIPT_DIR/.token" 2>/dev/null | tr -d '\n')
    BUDGET=$(curl -s -H "Authorization: Bearer $TOKEN" https://api.ainm.no/astar-island/budget 2>/dev/null)
    QUERIES_USED=$(echo "$BUDGET" | python3 -c "import json,sys; print(json.load(sys.stdin).get('queries_used', 0))" 2>/dev/null)

    if [ "$QUERIES_USED" = "0" ]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) NEW ROUND $ACTIVE detected, running solver" >> "$LOG_FILE"
        cd /home/me/ht/nmiai
        uv run python3 tasks/astar-island/solver.py >> "$LOG_FILE" 2>&1
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) Solver finished for $ACTIVE" >> "$LOG_FILE"
    fi

    sleep 180  # Check again in 3 min
done

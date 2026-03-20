#!/bin/bash
# Background watcher: checks for new rounds every 3 minutes
# ONLY runs queries and saves observations. Does NOT submit predictions.
# Warns when round is about to close so we can submit predictions.
# Usage: nohup ./auto_watcher.sh &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/auto_watcher.log"
WARNED_ROUND=""

echo "$(date -u +%Y-%m-%dT%H:%M:%S) Watcher started (PID $$)" >> "$LOG_FILE"

while true; do
    # Check all rounds
    ROUND_INFO=$(curl -s https://api.ainm.no/astar-island/rounds | python3 -c "
import json, sys
from datetime import datetime, timezone
rounds = json.load(sys.stdin)
now = datetime.now(timezone.utc)
for r in rounds:
    if r['status'] == 'active':
        closes = datetime.fromisoformat(r['closes_at'])
        mins_left = (closes - now).total_seconds() / 60
        print(f'{r[\"id\"]}|{r[\"round_number\"]}|{int(mins_left)}')
        sys.exit(0)
print('none')
" 2>/dev/null)

    if [ "$ROUND_INFO" = "none" ] || [ -z "$ROUND_INFO" ]; then
        sleep 180
        continue
    fi

    ACTIVE=$(echo "$ROUND_INFO" | cut -d'|' -f1)
    ROUND_NUM=$(echo "$ROUND_INFO" | cut -d'|' -f2)
    MINS_LEFT=$(echo "$ROUND_INFO" | cut -d'|' -f3)

    # DEADLINE WARNING: alert when < 30 minutes remain
    if [ "$MINS_LEFT" -le 30 ] && [ "$WARNED_ROUND" != "$ACTIVE" ]; then
        WARNED_ROUND="$ACTIVE"
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) WARNING: Round $ROUND_NUM closes in $MINS_LEFT minutes! Submit predictions NOW!" >> "$LOG_FILE"
        say "Warning! Astar Island round $ROUND_NUM closes in $MINS_LEFT minutes. Submit predictions now!" 2>/dev/null
        # Also notify master orchestrator
        tmux-tool send %7 "<agent id=\"auto-watcher\" role=\"astar-island-watcher\" pane=\"bg\">WARNING: Round $ROUND_NUM closes in $MINS_LEFT minutes! Submit predictions NOW!</agent>" 2>/dev/null
        sleep 0.5
        tmux send-keys -t %7 Enter 2>/dev/null
    fi

    # Run queries if we haven't yet
    TOKEN=$(cat "$SCRIPT_DIR/.token" 2>/dev/null | tr -d '\n')
    BUDGET=$(curl -s -H "Authorization: Bearer $TOKEN" https://api.ainm.no/astar-island/budget 2>/dev/null)
    QUERIES_USED=$(echo "$BUDGET" | python3 -c "import json,sys; print(json.load(sys.stdin).get('queries_used', 0))" 2>/dev/null)

    if [ "$QUERIES_USED" = "0" ]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) NEW ROUND $ROUND_NUM detected, running query_runner (queries only, no predictions)" >> "$LOG_FILE"
        say "New Astar Island round $ROUND_NUM detected. Running queries now." 2>/dev/null
        cd /home/me/ht/nmiai
        uv run python3 tasks/astar-island/query_runner.py >> "$LOG_FILE" 2>&1
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) Query runner finished for round $ROUND_NUM" >> "$LOG_FILE"
        say "Astar Island queries complete for round $ROUND_NUM. Ready for prediction submission." 2>/dev/null
    fi

    sleep 180
done

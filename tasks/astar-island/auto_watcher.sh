#!/bin/bash
# Background watcher for Astar Island rounds.
#
# TIMING STRATEGY (from master):
#   - Round opens: ANNOUNCE via say. DO NOT query.
#   - 1 hour AFTER opening: Run queries (query_runner.py)
#   - 30 min before close: Submit predictions (predictor.py)
#   - Exception: master can override for specific rounds.
#
# Usage: nohup ./auto_watcher.sh &

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/auto_watcher.log"
ANNOUNCED_ROUND=""
QUERIED_ROUND=""
SUBMITTED_ROUND=""

echo "$(date -u +%Y-%m-%dT%H:%M:%S) Watcher started (PID $$) — timing: announce@open, query@1h-after-open, submit@30min-before-close" >> "$LOG_FILE"

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

    # === NEW ROUND: Announce only ===
    if [ "$ANNOUNCED_ROUND" != "$ACTIVE" ]; then
        ANNOUNCED_ROUND="$ACTIVE"
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM OPENED. Closes at $CLOSES_AT UTC. ${MINS_UNTIL_CLOSE}min remaining." >> "$LOG_FILE"
        say "New Astar Island round opened. Round $ROUND_NUM, closes at $CLOSES_AT U T C. $MINS_UNTIL_CLOSE minutes remaining. Waiting to query." 2>/dev/null
        tmux-tool send %7 "<agent id=\"auto-watcher\" role=\"astar-island-watcher\" pane=\"bg\">NEW ROUND $ROUND_NUM opened. Closes $CLOSES_AT UTC (${MINS_UNTIL_CLOSE}min). Will query at 60min mark, submit at 30min-before-close.</agent>" 2>/dev/null
        sleep 0.5
        tmux send-keys -t %7 Enter 2>/dev/null
    fi

    # === 1 HOUR AFTER OPENING: Run queries ===
    if [ "$MINS_SINCE_OPEN" -ge 60 ] && [ "$QUERIED_ROUND" != "$ACTIVE" ]; then
        TOKEN=$(cat "$SCRIPT_DIR/.token" 2>/dev/null | tr -d '\n')
        BUDGET=$(curl -s -H "Authorization: Bearer $TOKEN" https://api.ainm.no/astar-island/budget 2>/dev/null)
        QUERIES_USED=$(echo "$BUDGET" | python3 -c "import json,sys; print(json.load(sys.stdin).get('queries_used', 0))" 2>/dev/null)

        if [ "$QUERIES_USED" = "0" ]; then
            echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM: ${MINS_SINCE_OPEN}min since open — starting queries" >> "$LOG_FILE"
            say "Astar Island round $ROUND_NUM: 1 hour since opening, starting queries now." 2>/dev/null
            cd /home/me/ht/nmiai
            uv run python3 tasks/astar-island/query_runner.py >> "$LOG_FILE" 2>&1
            echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM: queries done" >> "$LOG_FILE"
            say "Astar Island round $ROUND_NUM queries complete." 2>/dev/null
        fi
        QUERIED_ROUND="$ACTIVE"
    fi

    # === 30 MIN BEFORE CLOSE: Submit predictions ===
    if [ "$MINS_UNTIL_CLOSE" -le 30 ] && [ "$SUBMITTED_ROUND" != "$ACTIVE" ]; then
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM: ${MINS_UNTIL_CLOSE}min until close — submitting predictions" >> "$LOG_FILE"
        say "Astar Island round $ROUND_NUM: 30 minutes remaining, submitting predictions now." 2>/dev/null
        cd /home/me/ht/nmiai
        uv run python3 tasks/astar-island/predictor.py >> "$LOG_FILE" 2>&1
        echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ROUND_NUM: predictions submitted" >> "$LOG_FILE"
        say "Astar Island round $ROUND_NUM predictions submitted." 2>/dev/null
        SUBMITTED_ROUND="$ACTIVE"
        tmux-tool send %7 "<agent id=\"auto-watcher\" role=\"astar-island-watcher\" pane=\"bg\">Round $ROUND_NUM: predictions auto-submitted with ${MINS_UNTIL_CLOSE}min remaining.</agent>" 2>/dev/null
        sleep 0.5
        tmux send-keys -t %7 Enter 2>/dev/null
    fi

    sleep 120  # Check every 2 min
done

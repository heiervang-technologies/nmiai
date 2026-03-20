#!/bin/bash
# Auto-submit fallback: checks for active rounds and runs solver
# Designed to be run via cron every 5 minutes

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCK_FILE="/tmp/astar-island-autosubmit.lock"
LOG_FILE="$SCRIPT_DIR/logs/auto_submit.log"

# Prevent concurrent runs
if [ -f "$LOCK_FILE" ]; then
    pid=$(cat "$LOCK_FILE")
    if kill -0 "$pid" 2>/dev/null; then
        exit 0  # Already running
    fi
    rm -f "$LOCK_FILE"
fi
echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

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
    echo "$(date -u +%Y-%m-%dT%H:%M:%S) No active round" >> "$LOG_FILE"
    exit 0
fi

# Check if we already submitted for this round
TOKEN=$(cat "$SCRIPT_DIR/.token" 2>/dev/null | tr -d '\n')
if [ -z "$TOKEN" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%S) No auth token" >> "$LOG_FILE"
    exit 1
fi

BUDGET=$(curl -s -H "Authorization: Bearer $TOKEN" https://api.ainm.no/astar-island/budget 2>/dev/null)
QUERIES_USED=$(echo "$BUDGET" | python3 -c "import json,sys; print(json.load(sys.stdin).get('queries_used', 0))" 2>/dev/null)

if [ "$QUERIES_USED" != "0" ]; then
    echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ACTIVE: already has $QUERIES_USED queries, skipping" >> "$LOG_FILE"
    exit 0
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%S) Round $ACTIVE: NEW ROUND DETECTED, running solver" >> "$LOG_FILE"
cd /home/me/ht/nmiai
uv run python3 tasks/astar-island/solver.py >> "$LOG_FILE" 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%S) Solver finished" >> "$LOG_FILE"

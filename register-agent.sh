#!/bin/bash
# Register an agent in the registry
# Usage: ./register-agent.sh <name> <role> <description>
# Each agent gets its own file (no race conditions)

PANE="${TMUX_PANE:-unknown}"
NAME="${1:-unnamed}"
ROLE="${2:-unknown}"
DESC="${3:-no description}"
SESSION=$(tmux display-message -p '#{session_name}' 2>/dev/null || echo "unknown")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Write to registry/<pane_id>.json (one file per pane = no race condition)
PANE_CLEAN=$(echo "$PANE" | tr -d '%')
cat > /home/me/ht/nmiai/registry/pane_${PANE_CLEAN}.json << EOF
{
  "pane": "$PANE",
  "name": "$NAME",
  "role": "$ROLE",
  "description": "$DESC",
  "session": "$SESSION",
  "registered_at": "$TIMESTAMP",
  "pid": $$
}
EOF

echo "Registered: $NAME ($ROLE) in $PANE"

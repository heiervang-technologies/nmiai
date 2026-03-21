#!/bin/bash
# Show all registered agents as a table
# Usage: ./show-registry.sh

echo "=== AGENT REGISTRY ($(date -u +"%H:%M UTC")) ==="
echo ""
printf "%-6s %-25s %-20s %s\n" "PANE" "NAME" "ROLE" "DESCRIPTION"
printf "%-6s %-25s %-20s %s\n" "----" "----" "----" "-----------"

for f in /home/me/ht/nmiai/registry/pane_*.json; do
  [ -f "$f" ] || continue
  pane=$(python3 -c "import json; print(json.load(open('$f'))['pane'])" 2>/dev/null)
  name=$(python3 -c "import json; print(json.load(open('$f'))['name'])" 2>/dev/null)
  role=$(python3 -c "import json; print(json.load(open('$f'))['role'])" 2>/dev/null)
  desc=$(python3 -c "import json; print(json.load(open('$f'))['description'][:60])" 2>/dev/null)

  # Check if pane still exists
  if tmux display-message -t "$pane" -p '#{pane_id}' &>/dev/null; then
    status="ALIVE"
  else
    status="DEAD"
  fi

  [ "$status" = "DEAD" ] && continue
  printf "%-6s %-25s %-20s %s\n" "$pane" "$name" "$role" "$desc"
done

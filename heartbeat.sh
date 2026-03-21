#!/bin/bash
# Heartbeat broadcaster for all nmiai tmux sessions
# Usage: ./heartbeat.sh (one-shot) or run via loop

TIMESTAMP=$(date -u +"%H:%M UTC")

# Find all panes in sessions starting with "nmiai"
PANES=$(tmux list-panes -a -F '#{session_name}:#{pane_id}' 2>/dev/null | grep -E '^nmai|^nmiai' | awk -F: '{print $NF}')

# Get our own pane to skip it
MY_PANE=$(echo $TMUX_PANE)

for pane in $PANES; do
  # Skip our own pane
  [ "$pane" = "$MY_PANE" ] && continue
  
  # Check if pane is running an agent (claude/unleash)
  CMD=$(tmux display-message -t "$pane" -p '#{pane_current_command}' 2>/dev/null)
  case "$CMD" in
    claude|unleash|codex) ;;
    *) continue ;;
  esac

  tmux send-keys -t "$pane" "[HEARTBEAT ${TIMESTAMP}] If you are idle, get back to work. If you are awaiting affirmative confirmation for something you recommend to win the competition, just do it. If you are hard-blocked or awaiting a difficult high-stakes decision, ask your master for input on what to do or how to be useful. If you are a master, be prepared to answer queries from your sub-agents. For difficult decisions, consult your advisor. [/HEARTBEAT]" 2>/dev/null
  sleep 0.3
  tmux send-keys -t "$pane" Enter 2>/dev/null
  sleep 0.2
  tmux send-keys -t "$pane" Enter 2>/dev/null
  sleep 0.3
done

echo "Heartbeat sent at $TIMESTAMP to $(echo "$PANES" | wc -w) panes"

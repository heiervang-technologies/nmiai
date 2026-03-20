#!/bin/bash
# Overnight daemon for object detection - monitors MarkusNet training
# Pokes the master-object-detection agent every 30 minutes
MY_PANE="%2"
LOG="/home/me/ht/nmiai/tasks/object-detection/overnight_daemon.log"

echo "$(date -u +%Y-%m-%dT%H:%M:%S) OD Daemon started (PID $$)" >> "$LOG"

while true; do
    NOW=$(date -u +%Y-%m-%dT%H:%M:%S)
    
    # Check if training is running
    TRAINING=$(ps aux | grep -E "train|yolo|unsloth" | grep -v grep | wc -l)
    GPU_USAGE=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "N/A")
    
    MSG="OD DAEMON ($NOW): Training processes: $TRAINING. GPU: $GPU_USAGE. Check training logs, evaluate latest checkpoint if available, prepare submission if model improved."
    
    echo "$NOW $MSG" >> "$LOG"
    tmux-tool send "$MY_PANE" "$MSG"
    sleep 0.5
    tmux send-keys -t "$MY_PANE" Enter
    sleep 0.3
    tmux send-keys -t "$MY_PANE" Enter
    
    sleep 1800
done

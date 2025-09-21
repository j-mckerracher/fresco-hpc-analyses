#!/bin/bash
# Monitoring script for 2015 processing

LOG_FILE=$(ls processing_2015_*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "No log file found"
    exit 1
fi

echo "=== FRESCO 2015 Processing Monitor ==="
echo "Log file: $LOG_FILE"
echo "Started at: $(head -1 "$LOG_FILE" | grep -o '\[.*\]' | tr -d '[]')"
echo "Last update: $(tail -1 "$LOG_FILE" | grep -o '\[.*\]' | tr -d '[]')"
echo ""

echo "=== Current Progress ==="
tail -3 "$LOG_FILE" | grep -E "(Reading files|Progress:|Completed)"

echo ""
echo "=== Resource Usage ==="
echo "Disk usage in artifacts/:"
du -sh artifacts/ 2>/dev/null || echo "No artifacts yet"

echo ""
echo "=== Tmux Session Status ==="
tmux list-sessions | grep fresco-2015 || echo "Session not found"

echo ""
echo "=== Process Status ==="
ps aux | grep -E "(fresco-fd|python)" | grep -v grep | head -5

echo ""
echo "Commands to manage:"
echo "  Monitor live: tail -f $LOG_FILE"
echo "  Attach tmux:  tmux attach-session -t fresco-2015-processing"  
echo "  Check status: ./monitor_2015_processing.sh"

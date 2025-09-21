#!/bin/bash
# Processing script for 2015 FRESCO dataset
# Modified to process only 2015 with enhanced logging and proper Python path

set -e  # Exit on any error

# Set up environment
export PYTHONPATH="/home/dynamo/a/jmckerra/projects/fresco-analysis/early-warning-failure-detection:$PYTHONPATH"
cd /home/dynamo/a/jmckerra/projects/fresco-analysis/early-warning-failure-detection

YEARS=(2015)
HORIZONS="5,15"
MAX_WORKERS=8
DATA_ROOT="../data"
LOG_FILE="processing_2015_$(date +%Y%m%d_%H%M%S).log"

echo "Starting processing of 2015 FRESCO dataset"
echo "Years to process: ${YEARS[*]}"
echo "Horizons: $HORIZONS"
echo "Max workers: $MAX_WORKERS"
echo "Data root: $DATA_ROOT"
echo "Log file: $LOG_FILE"
echo "Working directory: $(pwd)"
echo "Python path: $PYTHONPATH"
echo ""

# Function to log with timestamp
log_msg() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to process a single year
process_year() {
    local year=$1
    log_msg "==========================================="
    log_msg "Processing year: $year"
    log_msg "Started at: $(date)"
    log_msg "==========================================="
    
    # Check available disk space before starting
    available_gb=$(df -BG /home/dynamo/a/jmckerra/projects/fresco-analysis/early-warning-failure-detection/artifacts/ | tail -1 | awk '{print $4}' | sed 's/G//')
    log_msg "Available disk space: ${available_gb}GB"
    
    if [ "$available_gb" -lt 10 ]; then
        log_msg "ERROR: Less than 10GB available space. Stopping."
        return 1
    fi
    
    # Test the command first
    log_msg "Testing fresco-fd command availability..."
    python -m fresco_failure_detector.fresco_fd.cli --help > /dev/null 2>&1
    if [ $? -ne 0 ]; then
        log_msg "ERROR: fresco-fd command not working"
        return 1
    fi
    log_msg "Command test successful"
    
    python -m fresco_failure_detector.fresco_fd.cli prepare \
        --data-root "$DATA_ROOT" \
        --years "$year" \
        --horizons "$HORIZONS" \
        --max-workers "$MAX_WORKERS" 2>&1 | tee -a "$LOG_FILE"
    
    local exit_code=${PIPESTATUS[0]}
    if [ $exit_code -eq 0 ]; then
        log_msg "✓ Year $year completed successfully at $(date)"
        # Show output size
        du -sh artifacts/datasets/ 2>/dev/null | tee -a "$LOG_FILE" || true
    else
        log_msg "✗ Year $year failed at $(date) with exit code $exit_code"
        return 1
    fi
    log_msg ""
}

# Process 2015
total_years=${#YEARS[@]}
completed=0

log_msg "Starting processing with PID: $$"
log_msg "Current working directory: $(pwd)"

for year in "${YEARS[@]}"; do
    if process_year "$year"; then
        ((completed++))
        log_msg "Progress: $completed/$total_years years completed"
    else
        log_msg "ERROR: Processing failed for year $year"
        log_msg "Completed years: $completed/$total_years"
        exit 1
    fi
done

log_msg "==========================================="
log_msg "All years processed successfully!"
log_msg "Completed at: $(date)"
log_msg "Total years: $total_years"
log_msg "Final dataset sizes:"
du -sh artifacts/datasets/* 2>/dev/null | tee -a "$LOG_FILE" || true
log_msg "==========================================="

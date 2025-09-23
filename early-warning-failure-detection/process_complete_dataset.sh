#!/bin/bash
# Sequential processing script for complete FRESCO dataset
# Processes each year individually to avoid memory exhaustion

set -e  # Exit on any error

YEARS=(2013 2014 2015 2016 2017 2018 2022 2023)
HORIZONS="5,15"
MAX_WORKERS=3
DATA_ROOT="../data"

echo "Starting sequential processing of complete FRESCO dataset"
echo "Years to process: ${YEARS[*]}"
echo "Horizons: $HORIZONS"
echo "Max workers: $MAX_WORKERS"
echo "Data root: $DATA_ROOT"
echo ""

# Function to process a single year
process_year() {
    local year=$1
    echo "==========================================="
    echo "Processing year: $year"
    echo "Started at: $(date)"
    echo "==========================================="
    
    fresco-fd prepare \
        --data-root "$DATA_ROOT" \
        --years "$year" \
        --horizons "$HORIZONS" \
        --max-workers "$MAX_WORKERS"
    
    if [ $? -eq 0 ]; then
        echo "✓ Year $year completed successfully at $(date)"
    else
        echo "✗ Year $year failed at $(date)"
        return 1
    fi
    echo ""
}

# Process each year sequentially
total_years=${#YEARS[@]}
completed=0

for year in "${YEARS[@]}"; do
    if process_year "$year"; then
        ((completed++))
        echo "Progress: $completed/$total_years years completed"
    else
        echo "ERROR: Processing failed for year $year"
        echo "Completed years: $completed/$total_years"
        exit 1
    fi
done

echo "==========================================="
echo "All years processed successfully!"
echo "Completed at: $(date)"
echo "Total years: $total_years"
echo "==========================================="

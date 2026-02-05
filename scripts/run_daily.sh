#!/bin/bash
# ABOUTME: Example script for daily cron processing of multiple stations
# ABOUTME: Customize CONDA_PATH, PROJECT_DIR, and STATIONS for your setup

# Configuration - UPDATE THESE FOR YOUR ENVIRONMENT
CONDA_PATH="${CONDA_PATH:-$HOME/miniconda3}"
PROJECT_DIR="${PROJECT_DIR:-$(dirname "$0")/..}"
STATIONS="${STATIONS:-GLBX FORA VALR}"
LOG_DIR="${PROJECT_DIR}/logs"

# Setup
mkdir -p "$LOG_DIR"
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate py39
cd "$PROJECT_DIR"

DATE=$(date +%Y-%m-%d)
echo "=== Daily GNSS-IR Processing: $DATE ===" >> "$LOG_DIR/daily_$DATE.log"

# Process each station
for station in $STATIONS; do
    echo "Processing $station at $(date)" >> "$LOG_DIR/daily_$DATE.log"
    python scripts/process_station.py --station "$station" --today >> "$LOG_DIR/daily_$DATE.log" 2>&1

    if [ $? -eq 0 ]; then
        echo "  $station: SUCCESS" >> "$LOG_DIR/daily_$DATE.log"
    else
        echo "  $station: FAILED" >> "$LOG_DIR/daily_$DATE.log"
    fi
done

echo "=== Completed at $(date) ===" >> "$LOG_DIR/daily_$DATE.log"

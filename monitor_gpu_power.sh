#!/bin/bash

# GPU Power Monitoring Script for LeRobot Training
# Usage: ./monitor_gpu_power.sh [log_file]

LOG_FILE=${1:-"gpu_power_log.csv"}
echo "Monitoring GPU power usage..."
echo "Logging to: $LOG_FILE"
echo "Press Ctrl+C to stop monitoring"

# Create CSV header
echo "timestamp,name,power_draw_w,power_limit_w,gpu_util_percent,memory_util_percent,temp_c,memory_used_mib,memory_total_mib" > "$LOG_FILE"

# Monitor GPU every second
while true; do
    nvidia-smi --query-gpu=timestamp,name,power.draw,power.limit,utilization.gpu,utilization.memory,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits >> "$LOG_FILE"
    sleep 1
done



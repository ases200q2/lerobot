#!/bin/bash

# Simple script to train a policy for gym_hil/PandaPickCubeSpacemouse-v0 environment
# using the ases200q2/PandaPickCubeSpacemouseRandom2_v30 dataset with lerobot-train command

# Generate date suffix
date_suffix=$(date +"%Y%m%d_%H%M%S")

# Training configuration
policy_type="act"
dataset_id="ases200q2/PandaPickCubeSpacemouse_v30.5"
output_dir="outputs/train/PandaPickCubeSpacemouse_v30.5_ACT_test_$date_suffix"
job_name="PandaPickCubeSpacemouse_v30.5_ACT_test_$date_suffix"
steps=20000
batch_size=64  # Increased from 32
num_workers=8  # Increased from 4 to improve dataloading and GPU utilization
save_freq=5000   # Save every 5000 steps (4 checkpoints total)
log_freq=500    # Log every 500 steps (reduced from default 200)

echo "Training $policy_type policy..."
echo "Dataset: $dataset_id"
echo "Output: $output_dir"
echo "Job name: $job_name"
echo "Steps: $steps"
echo "Batch size: $batch_size"
echo "Num workers: $num_workers"
echo ""
echo "GPU Status Before Training:"
nvidia-smi --query-gpu=name,power.draw,power.limit,utilization.gpu,memory.used,memory.total --format=csv
echo ""
echo "Starting GPU power monitoring in background..."
./monitor_gpu_power.sh "gpu_power_${job_name}.csv" &
MONITOR_PID=$!
echo "GPU monitor PID: $MONITOR_PID"

# Run lerobot-train command with optimizations
lerobot-train \
    --dataset.repo_id="$dataset_id" \
    --policy.type="$policy_type" \
    --output_dir="$output_dir" \
    --job_name="$job_name" \
    --policy.device=cuda \
    --wandb.enable=true \
    --policy.repo_id="$job_name" \
    --batch_size="$batch_size" \
    --steps="$steps" \
    --num_workers="$num_workers" \
    --dataset.video_backend=pyav \
    --policy.use_amp=true \
    --policy.optimizer_lr=2e-5 \
    --policy.optimizer_lr_backbone=2e-5 \
    --save_freq="$save_freq" \
    --log_freq="$log_freq"

echo "Training completed!"
echo ""
echo "Stopping GPU power monitoring..."
kill $MONITOR_PID 2>/dev/null
echo "GPU Status After Training:"
nvidia-smi --query-gpu=name,power.draw,power.limit,utilization.gpu,memory.used,memory.total --format=csv
echo ""
echo "GPU power log saved to: gpu_power_${job_name}.csv"
echo "You can analyze the power usage with:"
echo "  python -c \"import pandas as pd; df=pd.read_csv('gpu_power_${job_name}.csv'); print(f'Avg Power: {df[\"power_draw_w\"].mean():.1f}W, Max Power: {df[\"power_draw_w\"].max():.1f}W, Avg GPU Util: {df[\"gpu_util_percent\"].mean():.1f}%')\""

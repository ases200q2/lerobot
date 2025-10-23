#!/bin/bash

# Simple script to train a policy for gym_hil/PandaPickCubeSpacemouse-v0 environment
# using the ases200q2/PandaPickCubeSpacemouseRandom2_v30 dataset with lerobot-train command

# Generate date suffix
date_suffix=$(date +"%Y%m%d_%H%M%S")

# Training configuration
dataset_id="ases200q2/PandaPickCubeSpacemouseRandom2_v30"
policy_type="act"
output_dir="outputs/train/PandaPickCubeSpacemouseRandom2_ACT_test_$date_suffix"
job_name="PandaPickCubeSpacemouseRandom2_ACT_test_$date_suffix"
steps=20000
batch_size=32

echo "Training $policy_type policy..."
echo "Dataset: $dataset_id"
echo "Output: $output_dir"
echo "Job name: $job_name"
echo "Steps: $steps"
echo "Batch size: $batch_size"

# Run lerobot-train command
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
    --dataset.video_backend=pyav

echo "Training completed!"

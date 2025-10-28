#!/usr/bin/env bash
set -euo pipefail

# Resolve repository root (parent of this script's directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Activate conda environment
if command -v conda >/dev/null 2>&1; then
  echo "[info] Activating lerobot-org conda environment..."
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate lerobot-org
else
  echo "[error] conda not found. Please install conda or activate the environment manually." >&2
  exit 1
fi


# Set model repo ID
MODEL_REPO_ID="${HF_USER:-ases200q2}/panda_pick_cube_act"

# Train using settings from docs/source/il_sim.mdx (lines 127-133)
# Note: Disabled streaming due to corrupted parquet files, using minimal memory settings
lerobot-train \
  --dataset.repo_id="ases200q2/PandaPickCubeGamepad-v0" \
  --dataset.streaming=false \
  --dataset.video_backend=pyav \
  --policy.type=act \
  --output_dir="outputs/train/${MODEL_REPO_ID//\//_}" \
  --job_name="panda_pick_cube_act" \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.push_to_hub=true \
  --policy.repo_id="${MODEL_REPO_ID}" \
  --num_workers=4 \
  --batch_size=32 \
  --steps=20000 \
  --save_freq=5000



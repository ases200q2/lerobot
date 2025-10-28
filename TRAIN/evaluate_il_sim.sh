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

# Set model repo ID (should match training script)
MODEL_REPO_ID="${HF_USER:-ases200q2}/panda_pick_cube_act"

# Check if trained model exists
MODEL_PATH="outputs/train/${MODEL_REPO_ID//\//_}"
if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "[error] Trained model not found at ${MODEL_PATH}" >&2
  echo "[info] Available models:" >&2
  find outputs/train -name "*.safetensors" -o -name "config.json" | head -10 >&2
  exit 1
fi

echo "[info] Evaluating trained model from: ${MODEL_PATH}"

# Evaluate the trained policy
lerobot-eval \
  --policy.type=act \
  --policy.pretrained_path="${MODEL_PATH}" \
  --policy.device=cuda \
  --env.type=gym_manipulator \
  --env.task=PandaPickCubeGamepad-v0 \
  --env.fps=10 \
  --env.processor.control_mode=gamepad \
  --env.processor.gripper.use_gripper=true \
  --env.processor.gripper.gripper_penalty=-0.02 \
  --env.processor.reset.fixed_reset_joint_positions="[0.0,0.195,0.0,-2.43,0.0,2.62,0.785]" \
  --env.processor.reset.reset_time_s=2.0 \
  --env.processor.reset.control_time_s=30.0 \
  --env.processor.reset.terminate_on_success=false \
  --eval.n_episodes=5 \
  --eval.batch_size=1 \
  --eval.use_async_envs=false \
  --output_dir="outputs/eval/${MODEL_REPO_ID//\//_}" \
  --job_name="eval_panda_pick_cube_act"

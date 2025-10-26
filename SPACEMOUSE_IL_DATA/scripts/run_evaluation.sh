#!/bin/bash

################################################################################
# ACT Policy Evaluation Runner
#
# This script provides an easy way to evaluate a trained ACT policy on the
# PandaPickCube simulation environment.
#
# Usage:
#   ./scripts/run_evaluation.sh username/policy_path
#   ./scripts/run_evaluation.sh username/policy username/dataset
#   ./scripts/run_evaluation.sh username/policy username/dataset 20  # 20 episodes
################################################################################

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
POLICY_PATH="${1:-}"
DATASET_ID="${2:-}"
EPISODES="${3:-10}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "=============================================================================="
echo "  ACT Policy Evaluation for PandaPickCube"
echo "=============================================================================="
echo ""

# Check if policy path is provided
if [ -z "$POLICY_PATH" ]; then
    echo -e "${RED}Error: Policy path not provided${NC}"
    echo ""
    echo "Usage:"
    echo "  ./scripts/run_evaluation.sh <policy_path> [dataset_id] [episodes]"
    echo ""
    echo "Examples:"
    echo "  ./scripts/run_evaluation.sh username/panda_spacemouse_act_policy"
    echo "  ./scripts/run_evaluation.sh username/policy username/dataset 20"
    echo "  ./scripts/run_evaluation.sh outputs/train/model/checkpoints/last/pretrained_model"
    echo ""
    exit 1
fi

# Display parameters
echo "Evaluation Configuration:"
echo "  Policy: $POLICY_PATH"
if [ -n "$DATASET_ID" ]; then
    echo "  Dataset (for metadata): $DATASET_ID"
else
    echo "  Dataset: Will attempt to load from policy config"
fi
echo "  Episodes: $EPISODES"
echo ""

# Check Python environment
echo "Checking environment..."
if ! python -c "import lerobot" 2>/dev/null; then
    echo -e "${RED}Error: LeRobot not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ LeRobot installed${NC}"

# Check gym_hil
if ! python -c "import gym_hil" 2>/dev/null; then
    echo -e "${RED}Error: gym_hil not installed${NC}"
    echo "Please install with: pip install -e '.[hilserl]'"
    exit 1
fi
echo -e "${GREEN}✓ gym_hil installed${NC}"

# Check PyTorch
if ! python -c "import torch" 2>/dev/null; then
    echo -e "${RED}Error: PyTorch not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓ PyTorch installed${NC}"

# Check GPU
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
    echo -e "${GREEN}✓ GPU detected: $GPU_NAME${NC}"
    DEVICE="cuda"
else
    echo -e "${YELLOW}⚠ No GPU detected, using CPU${NC}"
    DEVICE="cpu"
fi
echo ""

# Check if policy exists (for local paths)
if [[ "$POLICY_PATH" != *"/"* ]] || [[ "$POLICY_PATH" == *"outputs/"* ]]; then
    # Looks like a local path
    if [ ! -d "$POLICY_PATH" ] && [ ! -f "$POLICY_PATH/config.json" ]; then
        echo -e "${YELLOW}Warning: Local policy path may not exist: $POLICY_PATH${NC}"
        echo "Will attempt to load anyway..."
        echo ""
    fi
fi

# Build command
CMD="python $SCRIPT_DIR/eval_policy.py --policy_path $POLICY_PATH --episodes $EPISODES --device $DEVICE"

if [ -n "$DATASET_ID" ]; then
    CMD="$CMD --dataset_id $DATASET_ID"
fi

echo "GUI visualization will be enabled (use --no-gui to disable)"
echo ""
echo "Controls during evaluation:"
echo "  Press 'q' in the image window to quit"
echo ""
echo "Starting evaluation..."
echo ""

# Run evaluation
cd "$PROJECT_DIR"
eval $CMD

# Check exit status
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=============================================================================="
    echo "  Evaluation Completed Successfully!"
    echo -e "==============================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Review the success rate and average reward above"
    echo "  2. If performance is poor, consider:"
    echo "     - Collecting more demonstration data"
    echo "     - Training for more steps"
    echo "     - Adjusting hyperparameters"
    echo "  3. If performance is good, test on real robot or Isaac Sim"
else
    echo ""
    echo -e "${RED}=============================================================================="
    echo "  Evaluation Failed"
    echo -e "==============================================================================${NC}"
    echo ""
    echo "Common issues:"
    echo "  - Policy not found: Check the policy path"
    echo "  - Dataset not found: Provide --dataset_id if needed"
    echo "  - Environment errors: Check gym_hil installation"
    exit $EXIT_CODE
fi

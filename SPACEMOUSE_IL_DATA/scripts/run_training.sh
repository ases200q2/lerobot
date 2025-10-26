#!/bin/bash

################################################################################
# ACT Policy Training Runner
#
# This script provides an easy way to train an ACT policy using imitation
# learning on the collected SpaceMouse demonstration data.
#
# Usage:
#   ./scripts/run_training.sh username/dataset_id
#   ./scripts/run_training.sh username/dataset_id 50000        # 50k steps
#   ./scripts/run_training.sh username/dataset_id 50000 16    # Custom batch size
################################################################################

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments
DATASET_ID="${1:-}"
STEPS="${2:-100000}"
BATCH_SIZE="${3:-32}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================================================="
echo "  ACT Policy Training for PandaPickCubeSpacemouse-v0"
echo "=============================================================================="
echo ""

# Check if dataset ID is provided
if [ -z "$DATASET_ID" ]; then
    echo -e "${RED}Error: Dataset ID not provided${NC}"
    echo ""
    echo "Usage:"
    echo "  ./scripts/run_training.sh <dataset_id> [steps] [batch_size]"
    echo ""
    echo "Example:"
    echo "  ./scripts/run_training.sh username/panda_spacemouse_il_data"
    echo "  ./scripts/run_training.sh username/dataset 50000 16"
    echo ""
    exit 1
fi

# Check for placeholder
if [[ "$DATASET_ID" == *"YOUR_HF_USERNAME"* ]]; then
    echo -e "${RED}Error: Dataset ID contains placeholder 'YOUR_HF_USERNAME'${NC}"
    echo "Please provide your actual HuggingFace username and dataset name"
    echo ""
    exit 1
fi

# Display parameters
echo "Training Configuration:"
echo "  Dataset: $DATASET_ID"
echo "  Training steps: $STEPS"
echo "  Batch size: $BATCH_SIZE"
echo "  Policy type: ACT (Action Chunking Transformer)"
echo ""

# Check Python environment
echo "Checking environment..."
if ! python -c "import lerobot" 2>/dev/null; then
    echo -e "${RED}Error: LeRobot not installed${NC}"
    echo "Please install LeRobot first:"
    echo "  pip install -e ."
    exit 1
fi
echo -e "${GREEN}✓ LeRobot installed${NC}"

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
    echo -e "${YELLOW}⚠ No GPU detected, training will use CPU (slow)${NC}"
    DEVICE="cpu"
fi
echo ""

# Ask for confirmation
echo "This will start training and may take 30-60 minutes depending on your GPU."
echo ""
read -p "Continue? [y/N]: " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Training cancelled"
    exit 0
fi

# Run training
echo ""
echo "Starting training..."
echo ""

cd "$PROJECT_DIR"

python "$SCRIPT_DIR/train_policy.py" \
    --dataset_id "$DATASET_ID" \
    --steps "$STEPS" \
    --batch_size "$BATCH_SIZE" \
    --device "$DEVICE" \
    --push_to_hub

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=============================================================================="
    echo "  Training Completed Successfully!"
    echo -e "==============================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Evaluate your policy: ./scripts/run_evaluation.sh <policy_path>"
    echo "  2. View checkpoints: ls outputs/train/panda_spacemouse_act_*/checkpoints/"
    echo "  3. Check your HuggingFace Hub for uploaded model"
else
    echo ""
    echo -e "${RED}=============================================================================="
    echo "  Training Failed"
    echo -e "==============================================================================${NC}"
    echo ""
    echo "Common issues:"
    echo "  - Out of memory: Try reducing batch size"
    echo "  - Dataset not found: Check dataset ID and HuggingFace authentication"
    echo "  - CUDA errors: Check GPU drivers and PyTorch installation"
    exit 1
fi

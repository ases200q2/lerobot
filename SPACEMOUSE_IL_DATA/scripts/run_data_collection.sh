#!/bin/bash

################################################################################
# SpaceMouse Data Collection Runner
#
# This script provides an easy way to run data collection with SpaceMouse
# teleoperation for the PandaPickCubeSpacemouse-v0 simulation environment.
#
# Usage:
#   ./scripts/run_data_collection.sh
#   ./scripts/run_data_collection.sh 50                    # Collect 50 episodes
#   ./scripts/run_data_collection.sh 50 username/dataset   # Custom repo
################################################################################

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Default values
CONFIG_FILE="$PROJECT_DIR/configs/collect_data_config.json"
NUM_EPISODES="${1:-30}"  # Default to 30 episodes
REPO_ID="${2:-}"         # Optional HuggingFace repo ID

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================================================="
echo "  SpaceMouse Data Collection for PandaPickCubeSpacemouse-v0"
echo "=============================================================================="
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file not found: $CONFIG_FILE${NC}"
    exit 1
fi

# Display parameters
echo "Configuration:"
echo "  Config file: $CONFIG_FILE"
echo "  Episodes: $NUM_EPISODES"
if [ -n "$REPO_ID" ]; then
    echo "  Repository: $REPO_ID"
fi
echo ""

# Check Python environment
echo "Checking Python environment..."
if ! python -c "import lerobot" 2>/dev/null; then
    echo -e "${RED}Error: LeRobot not installed${NC}"
    echo "Please install LeRobot first:"
    echo "  pip install -e ."
    exit 1
fi
echo -e "${GREEN}✓ LeRobot installed${NC}"

# Check gym_hil
if ! python -c "import gym_hil" 2>/dev/null; then
    echo -e "${RED}Error: gym_hil not installed${NC}"
    echo "Please install gym_hil:"
    echo "  pip install -e '.[hilserl]'"
    exit 1
fi
echo -e "${GREEN}✓ gym_hil installed${NC}"
echo ""

# Build command
CMD="python $SCRIPT_DIR/collect_data.py --config $CONFIG_FILE --episodes $NUM_EPISODES"

if [ -n "$REPO_ID" ]; then
    CMD="$CMD --repo_id $REPO_ID"
fi

echo "Running data collection..."
echo "Command: $CMD"
echo ""

# Run the collection
cd "$PROJECT_DIR"
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}=============================================================================="
    echo "  Data Collection Completed Successfully!"
    echo -e "==============================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Verify your dataset on HuggingFace Hub"
    echo "  2. Visualize your data: lerobot-dataset-viz --repo-id <your-repo-id>"
    echo "  3. Train a policy: ./scripts/run_training.sh"
else
    echo ""
    echo -e "${RED}=============================================================================="
    echo "  Data Collection Failed"
    echo -e "==============================================================================${NC}"
    exit 1
fi

#!/bin/bash

# Simple script to run ACT model evaluation

# Default values
CHECKPOINT="ases200q2/PandaPickCubeSpacemouseRandom2_ACT_test_20251023_134850"
DATASET="ases200q2/PandaPickCubeSpacemouseRandom2_v30"
EPISODES=5
DEVICE="cuda:0"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --episodes)
            EPISODES="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --checkpoint CHECKPOINT  Model checkpoint or HuggingFace repo (default: $CHECKPOINT)"
            echo "  --dataset DATASET        Dataset to evaluate on (default: $DATASET)"
            echo "  --episodes EPISODES      Number of episodes to test (default: $EPISODES)"
            echo "  --device DEVICE          Device to use (default: $DEVICE)"
            echo "  -h, --help              Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "🚀 Running ACT Model Evaluation"
echo "Checkpoint: $CHECKPOINT"
echo "Dataset: $DATASET"
echo "Episodes: $EPISODES"
echo "Device: $DEVICE"
echo ""

# Run evaluation
python EVALUATE/evaluate_act_simple.py \
    --checkpoint "$CHECKPOINT" \
    --dataset "$DATASET" \
    --num-episodes "$EPISODES" \
    --device "$DEVICE"

echo ""
echo "✅ Evaluation completed!"

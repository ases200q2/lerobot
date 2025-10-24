#!/bin/bash

# Simple script to run dataset action replay (avoiding video issues)

# Default values
DATASET="ases200q2/PandaPickCubeSpacemouseRandom2_v30"
EPISODE=0
MAX_FRAMES=50
FPS=10.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --episode)
            EPISODE="$2"
            shift 2
            ;;
        --max-frames)
            MAX_FRAMES="$2"
            shift 2
            ;;
        --fps)
            FPS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --dataset DATASET      Dataset to replay (default: $DATASET)"
            echo "  --episode EPISODE      Episode index to replay (default: $EPISODE)"
            echo "  --max-frames FRAMES    Maximum frames to replay (default: $MAX_FRAMES)"
            echo "  --fps FPS              Replay FPS (default: $FPS)"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

echo "📺 Running Dataset Action Replay"
echo "Dataset: $DATASET"
echo "Episode: $EPISODE"
echo "Max frames: $MAX_FRAMES"
echo "FPS: $FPS"
echo ""

# Run action replay
python EVALUATE/replay_dataset_actions_only.py \
    --dataset "$DATASET" \
    --episode "$EPISODE" \
    --max-frames "$MAX_FRAMES" \
    --fps "$FPS"

echo ""
echo "✅ Action replay completed!"

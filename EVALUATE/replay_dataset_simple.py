#!/usr/bin/env python3
"""
Simple script to replay dataset episodes and check timing/behavior
"""

import argparse
import time

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Simple dataset replay")
    parser.add_argument(
        "--dataset",
        type=str,
        default="ases200q2/PandaPickCubeSpacemouseRandom2_v30",
        help="Dataset to replay",
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to replay")
    parser.add_argument(
        "--fps", type=float, default=10.0, help="Replay FPS (default: 10 to match data collection)"
    )

    args = parser.parse_args()

    print("📺 Simple Dataset Replay")
    print("=" * 40)

    # Load dataset
    print(f"📊 Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(args.dataset)
    print(f"✅ Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # Check episode bounds
    if args.episode >= dataset.num_episodes:
        print(f"❌ Episode {args.episode} not found. Dataset has {dataset.num_episodes} episodes.")
        return

    # Get episode boundaries
    from_idx = dataset.meta.episodes[args.episode]["dataset_from_index"]
    to_idx = dataset.meta.episodes[args.episode]["dataset_to_index"]
    episode_length = to_idx - from_idx

    print(f"\n📋 Episode {args.episode} Details:")
    print(f"  Frames: {from_idx} to {to_idx - 1} (length: {episode_length})")
    print(f"  Replay FPS: {args.fps}")
    print(f"  Max frames to replay: {args.max_frames}")

    # Calculate replay timing
    frame_delay = 1.0 / args.fps
    frames_to_replay = min(args.max_frames, episode_length)
    estimated_duration = frames_to_replay * frame_delay

    print(f"  Estimated duration: {estimated_duration:.1f} seconds")
    print("\n🎬 Starting replay...")
    print("Press Ctrl+C to stop early")

    try:
        # Replay frames
        for frame_idx in range(frames_to_replay):
            data_idx = from_idx + frame_idx
            frame = dataset[data_idx]

            print(f"\nFrame {frame_idx + 1}/{frames_to_replay} (data_idx: {data_idx})")

            # Display action information
            if "action" in frame:
                action = frame["action"]
                if isinstance(action, np.ndarray):
                    action_str = f"[{', '.join([f'{x:.3f}' for x in action])}]"
                else:
                    action_str = str(action)
                print(f"  Action: {action_str}")

            # Display observation information
            if "observation.state" in frame:
                state = frame["observation.state"]
                if isinstance(state, np.ndarray):
                    print(f"  State shape: {state.shape}")
                    print(f"  State range: [{state.min():.3f}, {state.max():.3f}]")

            # Display image information
            for key in ["observation.images.front", "observation.images.wrist"]:
                if key in frame:
                    img = frame[key]
                    if isinstance(img, (torch.Tensor, np.ndarray)):
                        print(f"  {key}: {img.shape}, range: [{img.min():.3f}, {img.max():.3f}]")

            # Wait for next frame (matching data collection FPS)
            time.sleep(frame_delay)

    except KeyboardInterrupt:
        print(f"\n⚠️  Replay interrupted by user at frame {frame_idx + 1}")

    print("\n✅ Replay completed!")
    print(f"  Frames replayed: {frame_idx + 1}/{frames_to_replay}")
    print(f"  Total time: {(frame_idx + 1) * frame_delay:.1f} seconds")


if __name__ == "__main__":
    main()

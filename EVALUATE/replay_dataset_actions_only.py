#!/usr/bin/env python3
"""
Simple script to replay dataset actions only (avoiding video decoding issues)
"""

import argparse
import time

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Simple dataset action replay")
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

    print("📺 Simple Dataset Action Replay")
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
    print("\n🎬 Starting action replay...")
    print("Press Ctrl+C to stop early")

    # Collect action statistics
    all_actions = []
    action_magnitudes = []

    try:
        # Replay frames
        for frame_idx in range(frames_to_replay):
            data_idx = from_idx + frame_idx

            # Try to get just the action data without video
            try:
                # Access the underlying HuggingFace dataset directly to avoid video decoding
                hf_item = dataset.hf_dataset[data_idx]

                # Extract action
                if "action" in hf_item:
                    action = hf_item["action"]
                    if isinstance(action, list):
                        action = np.array(action)
                    all_actions.append(action)

                    # Calculate action magnitude
                    action_mag = np.linalg.norm(action)
                    action_magnitudes.append(action_mag)

                    print(f"\nFrame {frame_idx + 1}/{frames_to_replay} (data_idx: {data_idx})")
                    print(f"  Action: [{', '.join([f'{x:.3f}' for x in action])}]")
                    print(f"  Action magnitude: {action_mag:.3f}")

                    # Show action components
                    if len(action) >= 4:
                        print(
                            f"  X: {action[0]:.3f}, Y: {action[1]:.3f}, Z: {action[2]:.3f}, Gripper: {action[3]:.3f}"
                        )

                else:
                    print(f"Frame {frame_idx + 1}: No action data found")

            except Exception as e:
                print(f"Frame {frame_idx + 1}: Error accessing data - {e}")
                continue

            # Wait for next frame (matching data collection FPS)
            time.sleep(frame_delay)

    except KeyboardInterrupt:
        print(f"\n⚠️  Replay interrupted by user at frame {frame_idx + 1}")

    # Summary statistics
    if all_actions:
        all_actions = np.array(all_actions)
        action_magnitudes = np.array(action_magnitudes)

        print("\n📊 Action Statistics:")
        print(f"  Frames replayed: {len(all_actions)}")
        print(f"  Total time: {len(all_actions) * frame_delay:.1f} seconds")
        print(f"  Action shape: {all_actions.shape}")
        print(f"  Mean action magnitude: {np.mean(action_magnitudes):.3f}")
        print(f"  Std action magnitude: {np.std(action_magnitudes):.3f}")
        print(
            f"  Min/Max action magnitude: {np.min(action_magnitudes):.3f} / {np.max(action_magnitudes):.3f}"
        )

        # Per-component statistics
        if all_actions.shape[1] >= 4:
            print("\n📈 Per-Component Statistics:")
            for i, name in enumerate(["X", "Y", "Z", "Gripper"]):
                component = all_actions[:, i]
                print(
                    f"  {name}: mean={np.mean(component):.3f}, std={np.std(component):.3f}, range=[{np.min(component):.3f}, {np.max(component):.3f}]"
                )

        # Action velocity (difference between consecutive actions)
        if len(all_actions) > 1:
            action_diffs = np.diff(all_actions, axis=0)
            action_velocities = np.linalg.norm(action_diffs, axis=1)
            print("\n🚀 Action Velocity Statistics:")
            print(f"  Mean velocity: {np.mean(action_velocities):.3f}")
            print(f"  Std velocity: {np.std(action_velocities):.3f}")
            print(f"  Max velocity: {np.max(action_velocities):.3f}")

    print("\n✅ Action replay completed!")


if __name__ == "__main__":
    main()

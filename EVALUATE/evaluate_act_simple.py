#!/usr/bin/env python3
"""
Simple evaluation script for trained ACT policy
"""

import argparse

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    parser = argparse.ArgumentParser(description="Simple ACT model evaluation")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint or HuggingFace repo"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ases200q2/PandaPickCubeSpacemouseRandom2_v30",
        help="Dataset to evaluate on",
    )
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to test")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    args = parser.parse_args()

    print("🤖 Simple ACT Model Evaluation")
    print("=" * 40)

    # Load policy
    print(f"📂 Loading policy: {args.checkpoint}")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(args.device)
    policy.eval()
    print("✅ Policy loaded")

    # Load dataset
    print(f"📊 Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(args.dataset)
    print(f"✅ Dataset loaded: {dataset.num_episodes} episodes")

    # Evaluate
    print(f"\n🧪 Testing {args.num_episodes} episodes...")

    total_errors = []
    successful_predictions = 0
    total_predictions = 0

    for episode_idx in range(min(args.num_episodes, dataset.num_episodes)):
        print(f"\nEpisode {episode_idx + 1}/{args.num_episodes}")

        # Get episode boundaries
        from_idx = dataset.episode_data_index["from"][episode_idx].item()
        to_idx = dataset.episode_data_index["to"][episode_idx].item()
        episode_length = to_idx - from_idx

        print(f"  Frames: {episode_length}")

        # Reset policy
        policy.reset()

        episode_errors = []

        # Test first 50 frames of episode
        for frame_idx in range(min(50, episode_length)):
            try:
                data_idx = from_idx + frame_idx
                frame = dataset[data_idx]

                # Prepare batch
                batch = {}
                for key in ["observation.images.front", "observation.images.wrist", "observation.state"]:
                    if key in frame:
                        batch[key] = frame[key].unsqueeze(0).to(args.device)

                # Get prediction
                with torch.no_grad():
                    action_pred = policy.select_action(batch)

                # Get ground truth
                gt_action = frame["action"]
                if isinstance(gt_action, np.ndarray):
                    gt_action = torch.from_numpy(gt_action)
                gt_action = gt_action.to(args.device)

                # Handle action chunking
                if action_pred.dim() > 1:
                    action_pred = action_pred[0]

                # Calculate error
                error = torch.nn.functional.mse_loss(action_pred.cpu(), gt_action.cpu()).item()
                episode_errors.append(error)
                total_errors.append(error)
                successful_predictions += 1

            except Exception as e:
                print(f"  ⚠️  Error on frame {frame_idx}: {e}")

            total_predictions += 1

        # Episode summary
        if episode_errors:
            mean_error = np.mean(episode_errors)
            print(f"  Mean MSE: {mean_error:.6f}")
        else:
            print("  ❌ No successful predictions")

    # Overall results
    print("\n" + "=" * 40)
    print("📊 RESULTS")
    print("=" * 40)

    if total_errors:
        mean_error = np.mean(total_errors)
        std_error = np.std(total_errors)

        print(f"Episodes tested: {args.num_episodes}")
        print(f"Successful predictions: {successful_predictions}/{total_predictions}")
        print(f"Mean MSE: {mean_error:.6f} ± {std_error:.6f}")
        print(f"Min/Max MSE: {min(total_errors):.6f} / {max(total_errors):.6f}")

        # Simple assessment
        if mean_error < 0.01:
            print("🏆 EXCELLENT - Very low error")
        elif mean_error < 0.1:
            print("✅ GOOD - Low error")
        elif mean_error < 1.0:
            print("⚠️  FAIR - Moderate error")
        else:
            print("❌ POOR - High error")
    else:
        print("❌ No successful evaluations")

    print("=" * 40)


if __name__ == "__main__":
    main()

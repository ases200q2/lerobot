#!/usr/bin/env python3
"""
Dataset evaluation script for ACT models
"""

import argparse

import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument(
        "--dataset", type=str, default="ases200q2/test_spacemouse_grasp", help="Dataset to evaluate on"
    )
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to test")
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Device to use (use cuda:0 instead of cuda)"
    )

    args = parser.parse_args()

    # Force single GPU usage to avoid device mismatch
    if args.device == "cuda":
        args.device = "cuda:0"

    print("🤖 ACT Model Evaluation")
    print("=" * 50)

    # Set CUDA device visibility to avoid multi-GPU issues
    if "cuda" in args.device:
        device_id = args.device.split(":")[-1]
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        # Now use cuda:0 since we've limited visibility to one GPU
        args.device = "cuda:0"

    # 1. Load the policy
    print(f"📂 Loading policy from: {args.checkpoint}")
    policy = ACTPolicy.from_pretrained(args.checkpoint)

    # Move all model components to the same device
    print(f"🔧 Moving model to device: {args.device}")
    policy = policy.to(args.device)

    policy.eval()
    print("✅ Policy loaded successfully")

    # 2. Load dataset
    print(f"📊 Loading dataset: {args.dataset}")
    dataset = LeRobotDataset(args.dataset)
    print(f"✅ Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # 3. Test on episodes
    print(f"\n🧪 Testing on {args.num_episodes} episodes...")

    episode_errors = []
    all_frame_errors = []

    eval_episodes = min(args.num_episodes, dataset.num_episodes)

    for episode_idx in range(eval_episodes):
        print(f"\n📋 Episode {episode_idx + 1}/{eval_episodes}")

        # Get episode boundaries
        from_idx = dataset.episode_data_index["from"][episode_idx].item()
        to_idx = dataset.episode_data_index["to"][episode_idx].item()
        episode_length = to_idx - from_idx

        print(f"   Frames: {from_idx} to {to_idx - 1} (length: {episode_length})")

        # Reset policy for new episode
        policy.reset()

        frame_errors = []
        successful_predictions = 0

        # Process each frame in the episode
        for frame_idx, data_idx in enumerate(
            range(from_idx, min(to_idx, from_idx + 100))
        ):  # Limit to 100 frames per episode
            try:
                # Get frame data
                frame = dataset[data_idx]

                # Prepare batch for policy
                batch = {}

                # Add images if they exist
                for key in ["observation.images.front", "observation.images.wrist"]:
                    if key in frame:
                        # Ensure tensor is on the correct device
                        tensor = frame[key].unsqueeze(0)
                        batch[key] = tensor.to(args.device)

                # Add state
                if "observation.state" in frame:
                    # Ensure tensor is on the correct device
                    tensor = frame["observation.state"].unsqueeze(0)
                    batch["observation.state"] = tensor.to(args.device)

                # Get prediction
                with torch.no_grad():
                    action_pred = policy.select_action(batch)

                # Get ground truth
                gt_action = frame["action"]
                if isinstance(gt_action, np.ndarray):
                    gt_action = torch.from_numpy(gt_action)

                # Ensure both tensors are on the same device
                gt_action = gt_action.to(args.device)
                action_pred = action_pred.to(args.device)

                # Handle action chunking - take first action if multiple predicted
                if action_pred.dim() > 1:
                    action_pred = action_pred[0]  # Take first action from chunk

                # Calculate MSE error
                error = torch.nn.functional.mse_loss(action_pred.cpu(), gt_action.cpu()).item()
                frame_errors.append(error)
                all_frame_errors.append(error)
                successful_predictions += 1

                if frame_idx % 10 == 0:
                    print(f"   Frame {frame_idx}: MSE = {error:.6f}")

            except Exception as e:
                print(f"   ⚠️  Error on frame {frame_idx}: {e}")
                continue

        # Episode summary
        if frame_errors:
            episode_mean_error = np.mean(frame_errors)
            episode_errors.append(episode_mean_error)
            print(f"   📈 Episode {episode_idx + 1} Results:")
            print(f"      - Successful predictions: {successful_predictions}/{episode_length}")
            print(f"      - Mean MSE: {episode_mean_error:.6f}")
            print(f"      - Min/Max MSE: {min(frame_errors):.6f} / {max(frame_errors):.6f}")
        else:
            print(f"   ❌ No successful predictions in episode {episode_idx + 1}")

    # Overall results
    print("\n" + "=" * 50)
    print("📊 EVALUATION RESULTS")
    print("=" * 50)

    if episode_errors:
        mean_episode_error = np.mean(episode_errors)
        std_episode_error = np.std(episode_errors)
        mean_frame_error = np.mean(all_frame_errors)
        std_frame_error = np.std(all_frame_errors)

        print(f"Episodes evaluated: {len(episode_errors)}/{args.num_episodes}")
        print(f"Total frames tested: {len(all_frame_errors)}")
        print("")
        print("📈 Episode-level metrics:")
        print(f"   Mean MSE: {mean_episode_error:.6f} ± {std_episode_error:.6f}")
        print("")
        print("🎯 Frame-level metrics:")
        print(f"   Mean MSE: {mean_frame_error:.6f} ± {std_frame_error:.6f}")
        print(f"   Min MSE:  {min(all_frame_errors):.6f}")
        print(f"   Max MSE:  {max(all_frame_errors):.6f}")

        # Performance assessment
        print("")
        print("🏆 Performance Assessment:")
        if mean_frame_error < 0.001:
            print("   ✅ EXCELLENT - Near perfect action reproduction")
        elif mean_frame_error < 0.01:
            print("   ✅ VERY GOOD - Low prediction error")
        elif mean_frame_error < 0.1:
            print("   ⚠️  GOOD - Acceptable prediction error")
        elif mean_frame_error < 1.0:
            print("   ⚠️  FAIR - Model is learning but needs improvement")
        else:
            print("   ❌ POOR - High prediction error, needs more training")

    else:
        print("❌ No successful evaluations completed")

    print("=" * 50)


if __name__ == "__main__":
    main()

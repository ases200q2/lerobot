#!/usr/bin/env python3
"""
Simple simulation evaluation script for trained ACT policy
"""

import argparse
import time

import gym_hil  # noqa: F401
import numpy as np
import torch
from gym_hil.wrappers.factory import make_env

from lerobot.policies.act.modeling_act import ACTPolicy


def main():
    parser = argparse.ArgumentParser(description="Simple ACT simulation evaluation")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to checkpoint or HuggingFace repo"
    )
    parser.add_argument("--num-episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")

    args = parser.parse_args()

    print("🤖 Simple ACT Simulation Evaluation")
    print("=" * 40)

    # Load policy
    print(f"📂 Loading policy: {args.checkpoint}")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy = policy.to(args.device)
    policy.eval()
    print("✅ Policy loaded")

    # Create environment with matching data collection config
    print("🎮 Creating simulation environment...")
    env = make_env(
        env_id="gym_hil/PandaPickCubeBase-v0",
        render_mode="human",
        image_obs=True,
        use_viewer=True,
        use_gamepad=False,
        use_gripper=True,
        auto_reset=False,
        reset_delay_seconds=20.0,  # Match data collection reset_time_s
        random_block_position=True,
    )
    print("✅ Environment created")

    # Run evaluation
    print(f"\n🎮 Running {args.num_episodes} episodes...")

    successful_episodes = 0
    total_reward = 0.0
    episode_lengths = []

    for episode in range(args.num_episodes):
        print(f"\nEpisode {episode + 1}/{args.num_episodes}")

        # Reset environment and policy
        obs, info = env.reset()
        policy.reset()

        episode_reward = 0.0
        episode_length = 0
        success = False

        # Run episode (max 100 steps)
        for step in range(100):
            try:
                # Prepare observation for policy
                batch = {}

                # Add state if available
                if "agent_pos" in obs:
                    batch["observation.state"] = (
                        torch.from_numpy(obs["agent_pos"]).unsqueeze(0).float().to(args.device)
                    )
                else:
                    batch["observation.state"] = torch.zeros(1, 18).float().to(args.device)

                # Add images if available
                if "pixels" in obs and obs["pixels"]:
                    pixel_keys = list(obs["pixels"].keys())

                    # Front camera
                    if pixel_keys:
                        img = obs["pixels"][pixel_keys[0]]
                        if img.ndim == 3:
                            img = img.transpose(2, 0, 1)  # HWC to CHW
                        img = torch.from_numpy(img).float() / 255.0
                        batch["observation.images.front"] = img.unsqueeze(0).to(args.device)

                    # Wrist camera (duplicate front if only one camera)
                    if len(pixel_keys) > 1:
                        img = obs["pixels"][pixel_keys[1]]
                    else:
                        img = (
                            obs["pixels"][pixel_keys[0]]
                            if pixel_keys
                            else np.zeros((128, 128, 3), dtype=np.uint8)
                        )

                    if img.ndim == 3:
                        img = img.transpose(2, 0, 1)  # HWC to CHW
                    img = torch.from_numpy(img).float() / 255.0
                    batch["observation.images.wrist"] = img.unsqueeze(0).to(args.device)

                # Get action from policy
                with torch.no_grad():
                    action = policy.select_action(batch)

                # Convert to numpy
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()

                # Handle action chunking
                if action.ndim > 1:
                    action = action[0]

                # Ensure action is 4D (x, y, z, gripper)
                if len(action) > 4:
                    action = action[:4]
                elif len(action) < 4:
                    action = np.pad(action, (0, 4 - len(action)), constant_values=1.0)

                # Execute action
                obs, reward, terminated, truncated, info = env.step(action)

                episode_reward += reward
                episode_length += 1

                # Check for success
                if info.get("succeed", False):
                    success = True
                    print(f"  ✅ Success at step {step}!")
                    break

                # Check for termination
                if terminated or truncated:
                    break

                # Match data collection timing (10 FPS = 0.1s delay)
                # This controls the action execution rate to match training data
                time.sleep(0.1)

            except Exception as e:
                print(f"  ⚠️  Error on step {step}: {e}")
                break

        # Episode summary
        if success:
            successful_episodes += 1

        total_reward += episode_reward
        episode_lengths.append(episode_length)

        print(f"  Success: {'✅ Yes' if success else '❌ No'}")
        print(f"  Length: {episode_length} steps")
        print(f"  Reward: {episode_reward:.3f}")

        # Wait between episodes
        time.sleep(2.0)

    # Close environment
    env.close()

    # Overall results
    print("\n" + "=" * 40)
    print("📊 SIMULATION RESULTS")
    print("=" * 40)

    success_rate = successful_episodes / args.num_episodes
    avg_reward = total_reward / args.num_episodes
    avg_length = np.mean(episode_lengths)

    print(f"Episodes completed: {args.num_episodes}")
    print(f"Success rate: {success_rate:.1%}")
    print(f"Average reward: {avg_reward:.3f}")
    print(f"Average length: {avg_length:.1f} steps")

    # Simple assessment
    if success_rate >= 0.8:
        print("🏆 EXCELLENT - High success rate!")
    elif success_rate >= 0.5:
        print("✅ GOOD - Decent success rate")
    elif success_rate >= 0.2:
        print("⚠️  FAIR - Low success rate")
    else:
        print("❌ POOR - Very low success rate")

    print("=" * 40)


if __name__ == "__main__":
    main()

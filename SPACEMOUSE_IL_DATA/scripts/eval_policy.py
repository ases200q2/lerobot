#!/usr/bin/env python3

"""
ACT Policy Evaluation Script for PandaPickCube Simulation

This script evaluates a trained ACT policy on the PandaPickCube simulation environment.
It handles proper observation preprocessing and action transformation to match the
training configuration.

Usage:
    python scripts/eval_policy.py --policy_path username/panda_spacemouse_act_policy
    python scripts/eval_policy.py --policy_path username/model --dataset_id username/dataset
    python scripts/eval_policy.py --policy_path username/model --episodes 20 --no-gui
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy, make_policy_config
from lerobot.utils.random_utils import set_seed
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import get_safe_torch_device, init_logging


class ObservationWrapper(gym.Wrapper):
    """
    Wrapper to transform gym_hil observations to match ACT policy expectations.

    Handles:
    - Image preprocessing (crop and resize to 128x128)
    - State vector construction (18-dim: joint pos + vel + EE pose)
    - Observation formatting to match LeRobot dataset structure
    - Optional GUI visualization using OpenCV
    """

    def __init__(self, env, device="cpu", show_gui=True):
        super().__init__(env)
        self.device = device
        self.show_gui = show_gui

        if show_gui:
            logging.info("GUI visualization enabled")

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_obs(obs), info

    def step(self, action):
        # Transform action from policy format (4-dim) to gym_hil format (7-dim)
        transformed_action = self._transform_action(action)
        obs, reward, terminated, truncated, info = self.env.step(transformed_action)

        # Show GUI if enabled
        if self.show_gui and 'pixels' in obs:
            self._show_images(obs['pixels'])

        return self._transform_obs(obs), reward, terminated, truncated, info

    def _transform_obs(self, obs):
        """Transform gym_hil observations to match policy expectations."""
        transformed = {}

        # Process image observations
        if 'pixels' in obs:
            # Preprocess front camera image
            front_img = self._preprocess_image(obs['pixels']['front'])
            front_tensor = torch.from_numpy(front_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            transformed['observation.images.front'] = front_tensor

            # Preprocess wrist camera image
            wrist_img = self._preprocess_image(obs['pixels']['wrist'])
            wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            transformed['observation.images.wrist'] = wrist_tensor

        # Build 18-dimensional state vector matching training configuration
        state_components = []

        # Add joint positions (7 dims)
        if 'agent_pos' in obs:
            state_components.append(obs['agent_pos'])

        # Add joint velocities (7 dims)
        if 'joint_velocities' in obs:
            state_components.append(obs['joint_velocities'])
        else:
            # Pad with zeros if not available
            state_components.append(np.zeros(7))

        # Add end-effector pose (4 dims: xyz position + gripper state)
        if 'ee_pose' in obs:
            # Take first 4 elements (position + gripper)
            state_components.append(obs['ee_pose'][:4])
        else:
            # Pad with zeros if not available
            state_components.append(np.zeros(4))

        # Concatenate to create 18-dim state vector
        complete_state = np.concatenate(state_components)

        # Ensure exactly 18 dimensions
        if len(complete_state) < 18:
            padding = np.zeros(18 - len(complete_state))
            complete_state = np.concatenate([complete_state, padding])
        elif len(complete_state) > 18:
            complete_state = complete_state[:18]

        # Convert to tensor with batch dimension
        state_tensor = torch.from_numpy(complete_state).float().unsqueeze(0).to(self.device)
        transformed['observation.state'] = state_tensor

        return transformed

    def _preprocess_image(self, image):
        """Apply crop and resize to match training configuration."""
        # Crop from top-left corner to 128x128
        if image.shape[0] > 128 or image.shape[1] > 128:
            cropped = image[:128, :128]
        else:
            cropped = image

        # Resize to 128x128
        resized = cv2.resize(cropped, (128, 128))

        return resized

    def _show_images(self, pixels):
        """Display camera images using OpenCV."""
        if not self.show_gui:
            return

        try:
            # Display front camera
            if 'front' in pixels:
                front_resized = cv2.resize(pixels['front'], (512, 512))
                cv2.imshow('Front Camera', front_resized)

            # Display wrist camera
            if 'wrist' in pixels:
                wrist_resized = cv2.resize(pixels['wrist'], (512, 512))
                cv2.imshow('Wrist Camera', wrist_resized)

            # Non-blocking wait
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info("Quit requested by user")
                cv2.destroyAllWindows()
                sys.exit(0)
        except Exception as e:
            logging.warning(f"Error displaying images: {e}")

    def _transform_action(self, action):
        """
        Transform policy action (4-dim) to gym_hil action (7-dim).

        Policy outputs: [delta_x, delta_y, delta_z, gripper]
        Env expects: [delta_x, delta_y, delta_z, delta_rx, delta_ry, delta_rz, gripper]
        """
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        # Remove batch dimension if present
        if action.ndim > 1:
            action = action.squeeze()

        # Pad with zeros for rotation components
        if len(action) == 4:
            transformed = np.array([
                action[0],  # delta_x
                action[1],  # delta_y
                action[2],  # delta_z
                0.0,        # delta_rx (not used)
                0.0,        # delta_ry (not used)
                0.0,        # delta_rz (not used)
                action[3]   # gripper
            ])
        else:
            transformed = action

        return transformed


def evaluate_policy(env, policy, n_episodes=10, fps=10):
    """
    Evaluate policy with FPS timing control matching training.

    Args:
        env: Wrapped gym environment
        policy: Trained ACT policy
        n_episodes: Number of episodes to evaluate
        fps: Frame rate for evaluation (should match training)

    Returns:
        List of episode rewards
    """
    dt = 1.0 / fps
    episode_rewards = []
    episode_successes = []

    logging.info(f"Starting evaluation for {n_episodes} episodes at {fps} FPS")

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        step_count = 0

        while True:
            step_start_time = __import__('time').perf_counter()

            # Get action from policy
            with torch.no_grad():
                action = policy.select_action(obs)

            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1

            # Maintain FPS timing
            elapsed = __import__('time').perf_counter() - step_start_time
            if elapsed < dt:
                busy_wait(dt - elapsed)

            if terminated or truncated:
                break

        # Episode is successful if reward > 0
        success = episode_reward > 0
        episode_rewards.append(episode_reward)
        episode_successes.append(success)

        status = colored("✓ SUCCESS", "green") if success else colored("✗ FAILURE", "red")
        logging.info(f"Episode {episode + 1}/{n_episodes}: {step_count} steps, "
                    f"reward: {episode_reward:.3f} {status}")

    # Calculate success rate
    success_rate = sum(episode_successes) / len(episode_successes)
    avg_reward = np.mean(episode_rewards)

    print("\n" + "="*70)
    print(colored("  EVALUATION RESULTS", "cyan", attrs=["bold"]))
    print("="*70)
    print(f"  Episodes: {n_episodes}")
    print(f"  Success Rate: {success_rate:.1%} ({sum(episode_successes)}/{n_episodes})")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"  Reward Range: [{min(episode_rewards):.3f}, {max(episode_rewards):.3f}]")
    print("="*70 + "\n")

    return episode_rewards


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate ACT policy on PandaPickCube simulation")

    parser.add_argument(
        "--policy_path",
        type=str,
        required=True,
        help="HuggingFace Hub model ID or local path to trained policy",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help="HuggingFace dataset ID (for loading metadata, optional if policy has it)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Evaluation FPS (default: 10, matching training)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=['cuda', 'cpu', 'mps'],
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        default=True,
        help="Enable GUI visualization (default: True)",
    )
    parser.add_argument(
        "--no-gui",
        dest="gui",
        action="store_false",
        help="Disable GUI visualization",
    )

    args = parser.parse_args()

    # Initialize logging
    init_logging()

    print("\n" + "="*70)
    print("  ACT Policy Evaluation for PandaPickCube")
    print("="*70 + "\n")

    # Set random seed
    set_seed(args.seed)
    logging.info(f"Random seed: {args.seed}")

    # Get device
    device = get_safe_torch_device(args.device, log=True)

    # Create environment
    logging.info("Creating gym_hil environment...")
    try:
        import gym_hil  # noqa: F401
    except ImportError:
        logging.error("gym_hil not installed. Please install with: pip install -e '.[hilserl]'")
        sys.exit(1)

    # Create base environment
    base_env = gym.make(
        "gym_hil/PandaPickCubeBase-v0",
        image_obs=True,
    )

    # Wrap environment
    env = ObservationWrapper(base_env, device=str(device), show_gui=args.gui)
    logging.info("Environment created and wrapped")

    # Load dataset metadata for policy configuration
    logging.info("Loading dataset metadata...")
    if args.dataset_id:
        dataset_id = args.dataset_id
    else:
        # Try to infer from config
        logging.warning("Dataset ID not provided, attempting to load policy without it")
        dataset_id = None

    if dataset_id:
        try:
            dataset = LeRobotDataset(repo_id=dataset_id)
            dataset_meta = dataset.meta
            logging.info(f"Dataset metadata loaded from: {dataset_id}")
        except Exception as e:
            logging.error(f"Failed to load dataset: {e}")
            sys.exit(1)
    else:
        dataset_meta = None

    # Load policy
    logging.info(f"Loading policy from: {args.policy_path}")
    try:
        policy_cfg = make_policy_config("act", pretrained_path=args.policy_path, device=str(device))
        policy = make_policy(cfg=policy_cfg, ds_meta=dataset_meta)
        policy.eval()
        logging.info("Policy loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load policy: {e}")
        logging.error("Make sure the policy path is correct and accessible")
        sys.exit(1)

    # Run evaluation
    logging.info(f"\nStarting evaluation: {args.episodes} episodes at {args.fps} FPS\n")

    try:
        with torch.no_grad():
            evaluate_policy(env, policy, n_episodes=args.episodes, fps=args.fps)
    except KeyboardInterrupt:
        logging.warning("\nEvaluation interrupted by user")
    finally:
        env.close()
        if args.gui:
            cv2.destroyAllWindows()

    logging.info("Evaluation completed")


if __name__ == "__main__":
    main()

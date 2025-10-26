#!/usr/bin/env python3

"""
Simple evaluation script for testing trained ACT policy on gym_hil simulation.

This script loads a trained policy from HuggingFace Hub and evaluates it on the
PandaPickCubeBase-v0 gym_hil simulation environment with GUI visualization.

The script attempts to match the training configuration as closely as possible:
- Uses PandaPickCubeBase-v0 (avoids SpaceMouse requirement)
- Applies image preprocessing (crop and resize to 128x128)
- Handles 18-dimensional state vector (joint positions + velocities + EE pose)
- Transforms 4D policy actions to 7D environment actions
- Provides OpenCV-based GUI visualization

Note: Some training configuration features (gripper settings, exact timing) are not
fully supported by the base environment, but the core observation/action spaces match.

Usage:
    python EVAL/eval_gym_hil_simple.py
    python EVAL/eval_gym_hil_simple.py --checkpoint ases200q2/other_model --episodes 20
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import torch
import gymnasium as gym
from termcolor import colored

from lerobot.configs.eval import EvalConfig
from lerobot.envs.configs import HILSerlRobotEnvConfig
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.rl.gym_manipulator import make_robot_env
from lerobot.rl.eval_policy import eval_policy
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import get_safe_torch_device, init_logging
from lerobot.utils.robot_utils import busy_wait


def eval_policy_with_fps(env, policy, n_episodes, fps=10):
    """Evaluate policy with proper FPS timing control to match training."""
    import time
    import logging
    
    dt = 1.0 / fps
    sum_reward_episode = []
    
    logging.info(f"Starting evaluation with FPS timing control at {fps} FPS")
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        step_count = 0
        
        while True:
            step_start_time = time.perf_counter()
            
            # Get action from policy
            action = policy.select_action(obs)
            
            # Step environment
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            step_count += 1
            
            # Maintain FPS timing (same as training)
            elapsed_time = time.perf_counter() - step_start_time
            if elapsed_time < dt:
                busy_wait(dt - elapsed_time)
            
            if terminated or truncated:
                break
        
        sum_reward_episode.append(episode_reward)
        logging.info(f"Episode {episode + 1}: {step_count} steps, reward: {episode_reward:.3f}")

    logging.info(f"Success after 20 steps {sum_reward_episode}")
    logging.info(f"success rate {sum(sum_reward_episode) / len(sum_reward_episode)}")
    
    return sum_reward_episode


class ObservationWrapper(gym.Wrapper):
    """Wrapper to transform observations and show GUI using OpenCV."""
    
    def __init__(self, env, device="cpu", show_gui=False):
        super().__init__(env)
        self.device = device
        self.show_gui = show_gui
        if show_gui:
            import cv2
            self.cv2 = cv2
            print("GUI enabled - images will be displayed using OpenCV")
            print("Press 'q' to quit, any other key to continue")
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._transform_obs(obs), info
        
    def step(self, action):
        # Transform action from policy format to gym_hil format
        transformed_action = self._transform_action(action)
        obs, reward, terminated, truncated, info = self.env.step(transformed_action)
        
        # Show images using OpenCV if GUI is enabled
        if self.show_gui and 'pixels' in obs:
            self._show_images(obs['pixels'])
            
        return self._transform_obs(obs), reward, terminated, truncated, info
        
    def _transform_obs(self, obs):
        """Transform gym_hil observations to match policy expectations."""
        import torch
        import numpy as np
        transformed = {}
        
        # Transform pixels to observation.images format with proper preprocessing
        if 'pixels' in obs:
            # Apply crop and resize to match training config (128x128)
            front_img = self._preprocess_image(obs['pixels']['front'])
            wrist_img = self._preprocess_image(obs['pixels']['wrist'])
            
            # Convert to torch tensors, add batch dimension, and move to device
            front_tensor = torch.from_numpy(front_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            wrist_tensor = torch.from_numpy(wrist_img).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
            
            transformed['observation.images.front'] = front_tensor
            transformed['observation.images.wrist'] = wrist_tensor
        
        # Build the complete state vector to match training (18 dimensions)
        state_components = []
        
        # Add joint positions (agent_pos)
        if 'agent_pos' in obs:
            state_components.append(obs['agent_pos'])
        
        # Add joint velocities if available (from training config)
        if 'joint_velocities' in obs:
            state_components.append(obs['joint_velocities'])
        else:
            # If not available, add zeros to match expected state size
            joint_vel_zeros = np.zeros(7)  # 7 joint velocities
            state_components.append(joint_vel_zeros)
        
        # Add end-effector pose if available (from training config)
        if 'ee_pose' in obs:
            state_components.append(obs['ee_pose'])
        else:
            # If not available, add zeros to match expected state size
            ee_pose_zeros = np.zeros(7)  # 7 DOF end-effector pose (position + quaternion)
            state_components.append(ee_pose_zeros)
        
        # Concatenate all state components
        if state_components:
            complete_state = np.concatenate(state_components)
            # Ensure we have exactly 18 dimensions as expected by the policy
            if len(complete_state) < 18:
                # Pad with zeros if needed
                padding = np.zeros(18 - len(complete_state))
                complete_state = np.concatenate([complete_state, padding])
            elif len(complete_state) > 18:
                # Truncate if too long
                complete_state = complete_state[:18]
            
            # Convert to torch tensor with batch dimension
            state_tensor = torch.from_numpy(complete_state).float().unsqueeze(0).to(self.device)
            transformed['observation.state'] = state_tensor
                
        return transformed
    
    def _preprocess_image(self, image):
        """Apply crop and resize to match training configuration."""
        import cv2
        
        # Apply crop (0, 0, 128, 128) as specified in training config
        if image.shape[0] > 128 or image.shape[1] > 128:
            # Crop from top-left corner
            cropped = image[:128, :128]
        else:
            cropped = image
        
        # Resize to 128x128 as specified in training config
        resized = cv2.resize(cropped, (128, 128))
        
        return resized
        
    def _show_images(self, pixels):
        """Show images using OpenCV."""
        if not self.show_gui:
            return
            
        try:
            # Show front camera
            if 'front' in pixels:
                front_img = pixels['front']
                # Resize for better visibility
                front_img_resized = self.cv2.resize(front_img, (512, 512))
                self.cv2.imshow('Front Camera', front_img_resized)
            
            # Show wrist camera
            if 'wrist' in pixels:
                wrist_img = pixels['wrist']
                # Resize for better visibility
                wrist_img_resized = self.cv2.resize(wrist_img, (512, 512))
                self.cv2.imshow('Wrist Camera', wrist_img_resized)
            
            # Wait for key press (non-blocking)
            key = self.cv2.waitKey(100) & 0xFF  # Increased delay to 100ms
            if key == ord('q'):
                print("Quit requested by user")
                self.cv2.destroyAllWindows()
                exit(0)
                
        except Exception as e:
            print(f"Error showing images: {e}")
        
    def _transform_action(self, action):
        """Transform policy action to gym_hil format."""
        import torch
        import numpy as np
        
        # Convert tensor to numpy if needed
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        
        # Remove batch dimension if present
        if action.ndim > 1:
            action = action.squeeze()
        
        # The policy outputs 4 values (x, y, z, grasp), but gym_hil expects 7 (x, y, z, rx, ry, rz, grasp)
        # We'll pad with zeros for the rotation components
        if len(action) == 4:
            # Pad with zeros for rx, ry, rz
            transformed = np.array([action[0], action[1], action[2], 0.0, 0.0, 0.0, action[3]])
        else:
            # If action is already 7 values, use as is
            transformed = action
            
        return transformed


def create_env_config():
    """Create environment configuration for gym_hil PandaPickCubeBase-v0."""
    from lerobot.envs.configs import HILSerlProcessorConfig
    
    # Create minimal processor config that works with base environment
    # The base environment doesn't support all the advanced features
    processor = HILSerlProcessorConfig()
    
    return HILSerlRobotEnvConfig(
        name="gym_hil",
        task="PandaPickCubeBase-v0",  # Use base environment to avoid SpaceMouse requirement
        fps=10,  # Default FPS, can be overridden by command line
        robot=None,  # No robot needed for gym_hil
        teleop=None,  # No teleop needed for gym_hil
        processor=processor,
    )


def create_eval_config(n_episodes=10, batch_size=5):
    """Create evaluation configuration."""
    # Ensure batch_size doesn't exceed n_episodes
    batch_size = min(batch_size, n_episodes)
    return EvalConfig(
        n_episodes=n_episodes,
        batch_size=batch_size,
        use_async_envs=True,
    )


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Simple gym_hil policy evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="ases200q2/PandaPickCubeSpacemouseRandom2_ACT_test_20251023_134850",
        help="HuggingFace Hub checkpoint path or local path",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=5,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda, cpu, mps)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/eval/timestamp)",
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
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="FPS for evaluation timing control (default: 10, matches training)",
    )

    args = parser.parse_args()

    # Initialize logging
    init_logging()
    logging.info("Starting gym_hil policy evaluation")

    # Set random seed
    set_seed(args.seed)

    # Check device availability
    device = get_safe_torch_device(args.device, log=True)

    # Create output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/eval/gym_hil_eval_{timestamp}")
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {output_dir}")

    # Create environment configuration
    env_config = create_env_config()
    logging.info("Environment config created")

    # Create evaluation configuration
    eval_config = create_eval_config(n_episodes=args.episodes, batch_size=args.batch_size)
    logging.info(f"Evaluation config: {args.episodes} episodes, batch size {args.batch_size}")

    # Create environment directly to avoid LeRobot's hardcoded gripper parameters
    logging.info("Creating gym_hil environment with training configuration...")
    import gym_hil  # noqa: F401
    import gymnasium as gym
    
    # Create environment directly with minimal parameters that work
    base_env = gym.make(
        "gym_hil/PandaPickCubeBase-v0",
        image_obs=True,
    )
    
    # Try to enable rendering manually if GUI is requested
    if args.gui:
        logging.info("Attempting to enable GUI visualization...")
        try:
            # Try to set render mode after creation
            base_env.render_mode = "human"
            logging.info("Set render_mode to human")
        except Exception as e:
            logging.warning(f"Could not set render_mode: {e}")
        
        # Try to call render to initialize the viewer
        try:
            base_env.render()
            logging.info("Called render() to initialize viewer")
        except Exception as e:
            logging.warning(f"Could not call render(): {e}")
    
    # Wrap environment to transform observations to match policy expectations
    env = ObservationWrapper(base_env, device=str(device), show_gui=args.gui)
    logging.info("Created gym_hil environment with observation wrapper")

    # Load dataset metadata for policy configuration
    logging.info("Loading dataset metadata...")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    
    dataset = LeRobotDataset(repo_id="ases200q2/PandaPickCubeSpacemouseRandom2_v30")
    dataset_meta = dataset.meta
    logging.info("Dataset metadata loaded")

    # Create policy config and load policy
    logging.info("Loading policy from checkpoint...")
    from lerobot.policies.factory import make_policy_config
    
    # Create a basic ACT policy config
    policy_cfg = make_policy_config("act", pretrained_path=args.checkpoint, device=str(device))
    
    # Load the policy using dataset metadata
    policy = make_policy(
        cfg=policy_cfg,
        ds_meta=dataset_meta,
    )
    policy.eval()
    logging.info(f"Policy loaded from: {args.checkpoint}")

    # Run evaluation with proper FPS timing control
    logging.info(f"Starting evaluation with FPS timing control at {args.fps} FPS...")
    with torch.no_grad():
        # Use our custom evaluation function with FPS control matching training
        eval_policy_with_fps(env, policy=policy, n_episodes=eval_config.n_episodes, fps=args.fps)

    # Print results
    print("\n" + "="*60)
    print(colored("EVALUATION COMPLETED", "green", attrs=["bold"]))
    print("="*60)
    print(f"Total Episodes: {eval_config.n_episodes}")
    print("Check the logs above for success rate and reward information.")
    if args.gui:
        print("GUI: OpenCV windows were displayed during evaluation")
        print("Note: If you didn't see the windows, they might be behind other windows or on a different desktop")
    print(f"Results saved to: {output_dir}")

    # Close environment and cleanup
    env.close()
    if args.gui:
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
    logging.info("Evaluation completed successfully!")

    return None


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Simulation evaluation script for ACT models in gym_hil environment
"""

import argparse
import time
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

# Import gym_hil
import gym_hil  # noqa: F401
from gym_hil.wrappers.factory import make_env

from lerobot.policies.act.modeling_act import ACTPolicy


def create_observation_mapping(env_obs, policy_expected_keys):
    """
    Map environment observations to policy expected format for ACT.
    
    Args:
        env_obs: Observation from gym_hil environment
        policy_expected_keys: Keys that the policy expects
    
    Returns:
        Mapped observation dictionary
    """
    mapped_obs = {}
    
    # Handle state observation
    if "observation.state" in policy_expected_keys:
        if "agent_pos" in env_obs:
            mapped_obs["observation.state"] = torch.from_numpy(env_obs["agent_pos"]).unsqueeze(0).float()
        elif "state" in env_obs:
            mapped_obs["observation.state"] = torch.from_numpy(env_obs["state"]).unsqueeze(0).float()
        else:
            # Create dummy state if none available
            mapped_obs["observation.state"] = torch.zeros(1, 18).float()
    
    # Handle image observations
    if "pixels" in env_obs:
        pixel_keys = list(env_obs["pixels"].keys())
        
        # Map front camera
        if "observation.images.front" in policy_expected_keys:
            if "front" in pixel_keys:
                img = env_obs["pixels"]["front"]
            elif len(pixel_keys) > 0:
                img = env_obs["pixels"][pixel_keys[0]]  # Use first available camera
            else:
                # Create dummy image if none available
                img = np.zeros((128, 128, 3), dtype=np.uint8)
            
            # Convert from HWC to CHW and normalize to [0,1]
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).float() / 255.0
            mapped_obs["observation.images.front"] = img.unsqueeze(0)
        
        # Map wrist camera
        if "observation.images.wrist" in policy_expected_keys:
            if "wrist" in pixel_keys:
                img = env_obs["pixels"]["wrist"]
            elif len(pixel_keys) > 1:
                img = env_obs["pixels"][pixel_keys[1]]  # Use second available camera
            elif len(pixel_keys) > 0:
                img = env_obs["pixels"][pixel_keys[0]]  # Duplicate first camera
            else:
                # Create dummy image if none available
                img = np.zeros((128, 128, 3), dtype=np.uint8)
                
            # Convert from HWC to CHW and normalize to [0,1]
            if img.ndim == 3:
                img = img.transpose(2, 0, 1)  # HWC to CHW
            img = torch.from_numpy(img).float() / 255.0
            mapped_obs["observation.images.wrist"] = img.unsqueeze(0)
    
    return mapped_obs


def evaluate_policy_in_simulation(policy, device, num_episodes=5):
    """
    Evaluate the ACT policy in the gym_hil simulation environment.
    
    Args:
        policy: Trained ACT policy
        device: Device to run on
        num_episodes: Number of episodes to run
    
    Returns:
        Dictionary with evaluation results
    """
    
    # Create the gym_hil environment (matching your training setup)
    env = make_env(
        env_id="gym_hil/PandaPickCubeBase-v0",
        render_mode="human",  # Set to "human" to see the simulation
        image_obs=True,
        use_viewer=True,
        use_gamepad=False,
        use_gripper=True,
        auto_reset=False,
        reset_delay_seconds=1.0,
        random_block_position=True  # Enable random cube position
    )
    
    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    print(f"  🎲 Random cube positioning: ENABLED")
    
    # Get policy expected input keys
    policy_expected_keys = set()
    if hasattr(policy, 'config') and hasattr(policy.config, 'input_shapes'):
        policy_expected_keys = set(policy.config.input_shapes.keys())
    else:
        # Default expected keys for ACT
        policy_expected_keys = {"observation.state", "observation.images.front", "observation.images.wrist"}
    
    print(f"  Policy expects: {policy_expected_keys}")
    
    results = {
        "episodes": [],
        "success_rate": 0.0,
        "average_episode_length": 0.0,
        "average_reward": 0.0
    }
    
    successful_episodes = 0
    total_episode_length = 0
    total_reward = 0.0
    
    try:
        for episode in range(num_episodes):
            print(f"\n🎮 Episode {episode + 1}/{num_episodes}")
            
            # Reset environment and policy
            obs, info = env.reset()
            policy.reset()
            
            episode_reward = 0.0
            episode_length = 0
            success = False
            
            # Run episode
            for step in range(100):  # Max 100 steps per episode
                # Map observation to policy format
                try:
                    mapped_obs = create_observation_mapping(obs, policy_expected_keys)
                    
                    # Move observations to device
                    for key in mapped_obs:
                        if isinstance(mapped_obs[key], torch.Tensor):
                            mapped_obs[key] = mapped_obs[key].to(device)
                    
                    # Get action from policy
                    with torch.no_grad():
                        action = policy.select_action(mapped_obs)
                    
                    # Convert action to numpy
                    if isinstance(action, torch.Tensor):
                        action = action.cpu().numpy()
                    
                    # ACT may return action chunks, take first action
                    if action.ndim > 1:
                        action = action[0]  # Take first action if batch or chunk
                    
                    # Ensure action is 4D for gym_hil (x, y, z, gripper)
                    if len(action) > 4:
                        action = action[:4]  # Take first 4 dimensions
                    elif len(action) < 4:
                        # Pad with neutral gripper command
                        action = np.pad(action, (0, 4 - len(action)), constant_values=1.0)
                    
                    # Debug: Print action every 10 steps
                    if step % 10 == 0:
                        print(f"      Step {step}: Action = [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]")
                    
                    # Step environment
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    episode_reward += reward
                    episode_length += 1
                    
                    # Check for success
                    if info.get("succeed", False):
                        success = True
                        print(f"   ✅ Success at step {step}!")
                        break
                    
                    # Check for termination
                    if terminated or truncated:
                        break
                        
                    # Small delay for visualization
                    time.sleep(0.1)
                    
                except Exception as e:
                    print(f"   ⚠️  Error during step {step}: {e}")
                    break
            
            # Episode summary
            if success:
                successful_episodes += 1
            
            total_episode_length += episode_length
            total_reward += episode_reward
            
            episode_result = {
                "episode": episode + 1,
                "success": success,
                "length": episode_length,
                "reward": episode_reward
            }
            results["episodes"].append(episode_result)
            
            print(f"   📊 Episode {episode + 1} Results:")
            print(f"      Success: {'✅ Yes' if success else '❌ No'}")
            print(f"      Length: {episode_length} steps")
            print(f"      Reward: {episode_reward:.3f}")
            
            # Wait a bit between episodes
            time.sleep(2.0)
    
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    
    finally:
        env.close()
    
    # Calculate overall statistics
    results["success_rate"] = successful_episodes / len(results["episodes"]) if results["episodes"] else 0.0
    results["average_episode_length"] = total_episode_length / len(results["episodes"]) if results["episodes"] else 0.0
    results["average_reward"] = total_reward / len(results["episodes"]) if results["episodes"] else 0.0
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate ACT model in simulation")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to ACT policy checkpoint")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use (use cuda:0 instead of cuda)")
    
    args = parser.parse_args()
    
    # Force single GPU usage to avoid device mismatch
    if args.device == "cuda":
        args.device = "cuda:0"
    
    print("🤖 ACT Simulation Evaluation")
    print("=" * 60)
    
    # Set CUDA device visibility to avoid multi-GPU issues
    if "cuda" in args.device:
        device_id = args.device.split(":")[-1]
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = device_id
        # Now use cuda:0 since we've limited visibility to one GPU
        args.device = "cuda:0"
    
    # 1. Load the ACT policy
    print(f"📂 Loading ACT policy from: {args.checkpoint}")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    
    # Move all model components to the same device
    print(f"🔧 Moving model to device: {args.device}")
    policy = policy.to(args.device)
    
    policy.eval()
    print("✅ Policy loaded successfully")
    
    # 2. Run simulation evaluation
    print(f"\n🎮 Starting simulation evaluation with {args.num_episodes} episodes...")
    print("Press Ctrl+C to stop early")
    
    results = evaluate_policy_in_simulation(policy, args.device, args.num_episodes)
    
    # 3. Print results
    print("\n" + "=" * 60)
    print("🏆 SIMULATION EVALUATION RESULTS")
    print("=" * 60)
    
    print(f"Episodes completed: {len(results['episodes'])}/{args.num_episodes}")
    print(f"Success rate: {results['success_rate']:.1%}")
    print(f"Average episode length: {results['average_episode_length']:.1f} steps")
    print(f"Average reward: {results['average_reward']:.3f}")
    
    print(f"\n📋 Episode Details:")
    for ep in results["episodes"]:
        status = "✅ SUCCESS" if ep["success"] else "❌ FAILED"
        print(f"  Episode {ep['episode']}: {status} | {ep['length']} steps | {ep['reward']:.3f} reward")
    
    # Performance assessment
    print(f"\n🎯 Performance Assessment:")
    if results['success_rate'] >= 0.8:
        print("   🏆 EXCELLENT - High success rate!")
    elif results['success_rate'] >= 0.5:
        print("   ✅ GOOD - Decent success rate")
    elif results['success_rate'] >= 0.2:
        print("   ⚠️  FAIR - Low success rate, needs improvement")
    else:
        print("   ❌ POOR - Very low success rate, model needs more training")
    
    print("=" * 60)


if __name__ == "__main__":
    main()






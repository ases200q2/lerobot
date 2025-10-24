---
noteId: "d8c26290b09c11f08bd0898d85b76daa"
tags: []
---

# LeRobot Simulation Environments Guide

This comprehensive guide covers how to use simulation environments with LeRobot, including MuJoCo-based environments, Isaac Sim integration, and custom simulation setups for robot learning.

## Table of Contents

1. [Overview](#overview)
2. [MuJoCo Environments](#mujoco-environments)
3. [Isaac Sim Integration](#isaac-sim-integration)
4. [Gym-HIL Environments](#gym-hil-environments)
5. [Custom Simulation Setup](#custom-simulation-setup)
6. [Data Collection in Simulation](#data-collection-in-simulation)
7. [Training in Simulation](#training-in-simulation)
8. [Troubleshooting](#troubleshooting)

## Overview

LeRobot supports various simulation environments for robot learning:

- **MuJoCo**: Physics-based simulation for manipulation tasks
- **Isaac Sim**: NVIDIA's high-fidelity robotics simulation
- **Gym-HIL**: Human-in-the-loop simulation environments
- **Custom Environments**: Integration with other simulators

### Benefits of Simulation

- **Cost-Effective**: No need for expensive hardware
- **Scalable**: Run multiple environments in parallel
- **Safe**: Test policies without risk to real robots
- **Reproducible**: Consistent environment conditions
- **Fast Iteration**: Rapid prototyping and testing

## MuJoCo Environments

### Installation

```bash
# Install MuJoCo
pip install mujoco

# Install LeRobot with MuJoCo support
pip install -e ".[mujoco]"
```

### Basic MuJoCo Environment

```python
import mujoco
import numpy as np

def create_mujoco_env():
    """Create a basic MuJoCo environment."""

    # Load MuJoCo model
    model = mujoco.MjModel.from_xml_path("robot_model.xml")
    data = mujoco.MjData(model)

    # Simulation loop
    for step in range(1000):
        # Set control inputs
        data.ctrl[:] = np.random.uniform(-1, 1, model.nu)

        # Step simulation
        mujoco.mj_step(model, data)

        # Get observations
        joint_positions = data.qpos.copy()
        joint_velocities = data.qvel.copy()

        # Render (optional)
        if step % 10 == 0:
            mujoco.mj_render(model, data)

    return model, data
```

### MuJoCo with LeRobot Integration

```python
import gymnasium as gym
from lerobot.envs.mujoco import MuJoCoEnv

class CustomMuJoCoEnv(MuJoCoEnv):
    """Custom MuJoCo environment for LeRobot."""

    def __init__(self, model_path, **kwargs):
        super().__init__(model_path, **kwargs)

        # Define action and observation spaces
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Dict({
            "joint_positions": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            ),
            "joint_velocities": gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
            ),
            "images": gym.spaces.Dict({
                "front": gym.spaces.Box(
                    low=0, high=255, shape=(480, 640, 3), dtype=np.uint8
                )
            })
        })

    def step(self, action):
        """Execute one simulation step."""
        # Apply action
        self.data.ctrl[:] = action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Get observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        terminated = self._check_termination()
        truncated = self._check_truncation()

        # Get info
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current observation."""
        return {
            "joint_positions": self.data.qpos[:7].copy(),
            "joint_velocities": self.data.qvel[:7].copy(),
            "images": {
                "front": self._render_camera("front_camera")
            }
        }

    def _render_camera(self, camera_name):
        """Render camera image."""
        # Set camera
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)

        # Render
        renderer = mujoco.Renderer(self.model, height=480, width=640)
        renderer.update_scene(self.data, camera=camera_id)
        image = renderer.render()

        return image
```

## Isaac Sim Integration

### Installation

```bash
# Install Isaac Sim (follow NVIDIA's installation guide)
# Download from: https://developer.nvidia.com/isaac-sim

# Install LeRobot with Isaac Sim support
pip install -e ".[isaac]"
```

### Isaac Sim Environment Setup

```python
import numpy as np
from isaac_sim import IsaacSim

class IsaacSimRobotEnv:
    """Isaac Sim environment for robot learning."""

    def __init__(self, scene_path, robot_config):
        # Initialize Isaac Sim
        self.sim = IsaacSim()
        self.sim.load_scene(scene_path)

        # Setup robot
        self.robot = self.sim.get_robot(robot_config["name"])
        self.robot.setup_controllers(robot_config["controllers"])

        # Setup cameras
        self.cameras = {}
        for cam_config in robot_config["cameras"]:
            self.cameras[cam_config["name"]] = self.sim.get_camera(cam_config["name"])

        # Setup task
        self.task = self.sim.get_task(robot_config["task"])

    def reset(self):
        """Reset environment to initial state."""
        self.sim.reset()
        self.task.reset()

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one simulation step."""
        # Apply action to robot
        self.robot.set_joint_targets(action)

        # Step simulation
        self.sim.step()

        # Get observation
        observation = self._get_observation()

        # Calculate reward
        reward = self.task.calculate_reward()

        # Check termination
        terminated = self.task.is_terminated()
        truncated = self.task.is_truncated()

        # Get info
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current observation."""
        observation = {
            "joint_positions": self.robot.get_joint_positions(),
            "joint_velocities": self.robot.get_joint_velocities(),
            "end_effector_pose": self.robot.get_end_effector_pose(),
            "images": {}
        }

        # Get camera images
        for cam_name, camera in self.cameras.items():
            observation["images"][cam_name] = camera.get_image()

        return observation

    def _get_info(self):
        """Get additional information."""
        return {
            "task_info": self.task.get_info(),
            "robot_info": self.robot.get_info(),
            "simulation_time": self.sim.get_time()
        }
```

### Isaac Sim Configuration

```json
{
  "isaac_sim_config": {
    "scene_path": "/path/to/scene.usd",
    "robot_config": {
      "name": "franka_panda",
      "task": "pick_and_place",
      "controllers": {
        "joint_controller": {
          "type": "position",
          "gains": {
            "kp": 100.0,
            "kd": 10.0
          }
        }
      },
      "cameras": [
        {
          "name": "front_camera",
          "position": [0.5, 0.0, 1.0],
          "target": [0.0, 0.0, 0.0],
          "resolution": [640, 480]
        },
        {
          "name": "wrist_camera",
          "position": [0.0, 0.0, 0.1],
          "target": [0.0, 0.0, 0.0],
          "resolution": [320, 240]
        }
      ]
    }
  }
}
```

## Gym-HIL Environments

### Installation

```bash
# Install gym-hil
pip install -e ".[hilserl]"
```

### Gym-HIL Environment Usage

```python
import gymnasium as gym
import gym_hil

def create_gym_hil_env():
    """Create Gym-HIL environment."""

    # Create environment
    env = gym.make(
        "gym_hil/PandaPickCubeSpacemouse-v0",
        image_obs=True,
        render_mode="human",
        use_gripper=True,
        gripper_penalty=-0.05
    )

    return env

def run_gym_hil_episode():
    """Run a complete episode in Gym-HIL."""

    env = create_gym_hil_env()
    obs, info = env.reset()

    episode_reward = 0
    step_count = 0

    while True:
        # Get action from teleoperator (SpaceMouse)
        action = env.get_teleop_action()

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        step_count += 1

        # Check termination
        if terminated or truncated:
            break

    env.close()

    print(f"Episode completed:")
    print(f"  Steps: {step_count}")
    print(f"  Reward: {episode_reward}")
    print(f"  Success: {info.get('success', False)}")

    return episode_reward, step_count
```

### Gym-HIL Configuration

```json
{
  "gym_hil_config": {
    "env": {
      "type": "gym_manipulator",
      "name": "gym_hil",
      "task": "PandaPickCubeSpacemouse-v0",
      "fps": 10
    },
    "processor": {
      "control_mode": "spacemouse",
      "max_gripper_pos": 30.0,
      "use_gripper": true,
      "gripper_penalty": -0.05,
      "reset": {
        "reset_time_s": 5.0,
        "control_time_s": 20.0
      }
    },
    "device": "cuda"
  }
}
```

### Gym-HIL Data Collection

```python
def collect_gym_hil_data(num_episodes=50):
    """Collect demonstration data in Gym-HIL."""

    # Create environment
    env = gym.make(
        "gym_hil/PandaPickCubeSpacemouse-v0",
        image_obs=True,
        render_mode="human",
        use_gripper=True
    )

    # Initialize dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id="user/gym_hil_dataset",
        root="./data/gym_hil",
        video=True
    )

    for episode in range(num_episodes):
        print(f"Recording episode {episode + 1}/{num_episodes}")

        obs, info = env.reset()
        episode_frames = []

        while True:
            # Get teleoperator action
            action = env.get_teleop_action()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Record frame
            frame = {
                "observation": obs,
                "action": action,
                "reward": reward,
                "done": terminated or truncated,
                "info": info
            }
            episode_frames.append(frame)

            # Check termination
            if terminated or truncated:
                break

        # Save episode
        dataset.add_episode(episode_frames)
        print(f"Episode {episode + 1} completed with {len(episode_frames)} frames")

    # Finalize dataset
    dataset.finalize()
    dataset.push_to_hub()

    env.close()
    print(f"Dataset collection completed: {dataset.num_episodes} episodes")

    return dataset
```

## Custom Simulation Setup

### Custom Environment Framework

```python
import gymnasium as gym
from abc import ABC, abstractmethod

class CustomSimulationEnv(gym.Env, ABC):
    """Base class for custom simulation environments."""

    def __init__(self, config):
        self.config = config
        self.simulator = self._initialize_simulator()
        self.robot = self._initialize_robot()
        self.task = self._initialize_task()
        self.cameras = self._initialize_cameras()

        # Define spaces
        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

    @abstractmethod
    def _initialize_simulator(self):
        """Initialize the simulation engine."""
        pass

    @abstractmethod
    def _initialize_robot(self):
        """Initialize the robot model."""
        pass

    @abstractmethod
    def _initialize_task(self):
        """Initialize the task environment."""
        pass

    @abstractmethod
    def _initialize_cameras(self):
        """Initialize cameras."""
        pass

    @abstractmethod
    def _define_action_space(self):
        """Define action space."""
        pass

    @abstractmethod
    def _define_observation_space(self):
        """Define observation space."""
        pass

    def reset(self, seed=None, options=None):
        """Reset environment."""
        super().reset(seed=seed)

        # Reset simulator
        self.simulator.reset()

        # Reset robot
        self.robot.reset()

        # Reset task
        self.task.reset()

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """Execute one step."""
        # Apply action
        self.robot.set_action(action)

        # Step simulation
        self.simulator.step()

        # Get observation
        observation = self._get_observation()

        # Calculate reward
        reward = self.task.calculate_reward()

        # Check termination
        terminated = self.task.is_terminated()
        truncated = self.task.is_truncated()

        # Get info
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        """Get current observation."""
        observation = {
            "joint_positions": self.robot.get_joint_positions(),
            "joint_velocities": self.robot.get_joint_velocities(),
            "end_effector_pose": self.robot.get_end_effector_pose(),
            "images": {}
        }

        # Get camera images
        for cam_name, camera in self.cameras.items():
            observation["images"][cam_name] = camera.get_image()

        return observation

    def _get_info(self):
        """Get additional information."""
        return {
            "task_info": self.task.get_info(),
            "robot_info": self.robot.get_info(),
            "simulation_time": self.simulator.get_time()
        }

    def render(self, mode="human"):
        """Render environment."""
        if mode == "human":
            return self.simulator.render()
        elif mode == "rgb_array":
            return self.simulator.render_rgb()
        else:
            raise ValueError(f"Unsupported render mode: {mode}")

    def close(self):
        """Close environment."""
        self.simulator.close()
```

### Custom MuJoCo Environment

```python
import mujoco
import numpy as np

class CustomMuJoCoEnv(CustomSimulationEnv):
    """Custom MuJoCo environment implementation."""

    def _initialize_simulator(self):
        """Initialize MuJoCo simulator."""
        model_path = self.config["model_path"]
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # Setup renderer
        self.renderer = mujoco.Renderer(self.model, height=480, width=640)

        return {
            "model": self.model,
            "data": self.data,
            "renderer": self.renderer
        }

    def _initialize_robot(self):
        """Initialize robot in MuJoCo."""
        return {
            "joint_names": [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                           for i in range(self.model.njnt)],
            "actuator_names": [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                              for i in range(self.model.nu)]
        }

    def _initialize_task(self):
        """Initialize task environment."""
        return {
            "target_position": np.array([0.5, 0.0, 0.1]),
            "success_threshold": 0.05
        }

    def _initialize_cameras(self):
        """Initialize cameras."""
        cameras = {}
        for cam_config in self.config["cameras"]:
            camera_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_config["name"]
            )
            cameras[cam_config["name"]] = {
                "id": camera_id,
                "config": cam_config
            }
        return cameras

    def _define_action_space(self):
        """Define action space."""
        return gym.spaces.Box(
            low=-1.0, high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32
        )

    def _define_observation_space(self):
        """Define observation space."""
        return gym.spaces.Dict({
            "joint_positions": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.model.nq,),
                dtype=np.float32
            ),
            "joint_velocities": gym.spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.model.nv,),
                dtype=np.float32
            ),
            "images": gym.spaces.Dict({
                name: gym.spaces.Box(
                    low=0, high=255,
                    shape=(480, 640, 3),
                    dtype=np.uint8
                ) for name in self.cameras.keys()
            })
        })

    def _get_observation(self):
        """Get current observation."""
        observation = {
            "joint_positions": self.data.qpos.copy(),
            "joint_velocities": self.data.qvel.copy(),
            "images": {}
        }

        # Get camera images
        for cam_name, camera in self.cameras.items():
            self.renderer.update_scene(self.data, camera=camera["id"])
            image = self.renderer.render()
            observation["images"][cam_name] = image

        return observation

    def step(self, action):
        """Execute one step."""
        # Apply action
        self.data.ctrl[:] = action

        # Step simulation
        mujoco.mj_step(self.model, self.data)

        # Get observation
        observation = self._get_observation()

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        terminated = self._check_termination()
        truncated = self._check_truncation()

        # Get info
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self):
        """Calculate reward."""
        # Get end-effector position
        ee_pos = self.data.site_xpos[0]  # Assuming first site is end-effector

        # Distance to target
        target_pos = self.task["target_position"]
        distance = np.linalg.norm(ee_pos - target_pos)

        # Reward based on distance
        reward = -distance

        # Success bonus
        if distance < self.task["success_threshold"]:
            reward += 10.0

        return reward

    def _check_termination(self):
        """Check if episode is terminated."""
        # Check if target reached
        ee_pos = self.data.site_xpos[0]
        target_pos = self.task["target_position"]
        distance = np.linalg.norm(ee_pos - target_pos)

        return distance < self.task["success_threshold"]

    def _check_truncation(self):
        """Check if episode is truncated."""
        # Truncate after maximum steps
        return self.data.time > 10.0  # 10 seconds max
```

## Data Collection in Simulation

### Simulation Data Collection Script

```python
#!/usr/bin/env python3
"""
Data collection script for simulation environments
"""

import argparse
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def collect_simulation_data(
    env_type,
    num_episodes=50,
    episode_length=100,
    output_dir="./data/simulation"
):
    """Collect demonstration data in simulation."""

    print(f"🎮 Collecting simulation data")
    print(f"Environment: {env_type}")
    print(f"Episodes: {num_episodes}")
    print(f"Episode length: {episode_length}")

    # Create environment
    if env_type == "mujoco":
        env = create_mujoco_env()
    elif env_type == "isaac":
        env = create_isaac_env()
    elif env_type == "gym_hil":
        env = create_gym_hil_env()
    else:
        raise ValueError(f"Unknown environment type: {env_type}")

    # Initialize dataset
    dataset = LeRobotDataset(
        repo_id="user/simulation_dataset",
        root=output_dir,
        video=True
    )

    # Collect episodes
    for episode in range(num_episodes):
        print(f"Recording episode {episode + 1}/{num_episodes}")

        obs, info = env.reset()
        episode_frames = []

        for step in range(episode_length):
            # Get action (random for demonstration)
            action = env.action_space.sample()

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Record frame
            frame = {
                "observation": obs,
                "action": action,
                "reward": reward,
                "done": terminated or truncated,
                "info": info
            }
            episode_frames.append(frame)

            # Check termination
            if terminated or truncated:
                break

        # Save episode
        dataset.add_episode(episode_frames)
        print(f"Episode {episode + 1} completed with {len(episode_frames)} frames")

    # Finalize dataset
    dataset.finalize()
    dataset.push_to_hub()

    env.close()
    print(f"Dataset collection completed: {dataset.num_episodes} episodes")

    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect simulation data")
    parser.add_argument("--env-type", type=str, required=True,
                       choices=["mujoco", "isaac", "gym_hil"],
                       help="Simulation environment type")
    parser.add_argument("--num-episodes", type=int, default=50,
                       help="Number of episodes to collect")
    parser.add_argument("--episode-length", type=int, default=100,
                       help="Maximum episode length")
    parser.add_argument("--output-dir", type=str, default="./data/simulation",
                       help="Output directory for dataset")

    args = parser.parse_args()

    collect_simulation_data(
        env_type=args.env_type,
        num_episodes=args.num_episodes,
        episode_length=args.episode_length,
        output_dir=args.output_dir
    )
```

### Teleoperation in Simulation

```python
def collect_teleop_data_simulation(env_type, num_episodes=20):
    """Collect teleoperation data in simulation."""

    # Create environment
    env = create_simulation_env(env_type)

    # Setup teleoperator
    if env_type == "gym_hil":
        teleop = env.get_teleop_device()
    else:
        teleop = setup_teleoperator()

    # Initialize dataset
    dataset = LeRobotDataset(
        repo_id="user/teleop_simulation_dataset",
        root="./data/teleop_simulation",
        video=True
    )

    print("🎮 Teleoperation data collection")
    print("Use your input device to control the robot")
    print("Press 'q' to quit, 'r' to reset episode")

    for episode in range(num_episodes):
        print(f"\nEpisode {episode + 1}/{num_episodes}")
        print("Reset environment and start teleoperation...")

        obs, info = env.reset()
        episode_frames = []

        while True:
            # Get teleoperator action
            action = teleop.get_action()

            # Check for quit/reset commands
            if teleop.get_quit_signal():
                print("Quitting data collection...")
                env.close()
                return dataset

            if teleop.get_reset_signal():
                print("Resetting episode...")
                break

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            # Record frame
            frame = {
                "observation": obs,
                "action": action,
                "reward": reward,
                "done": terminated or truncated,
                "info": info
            }
            episode_frames.append(frame)

            # Check termination
            if terminated or truncated:
                print(f"Episode completed: {len(episode_frames)} frames")
                break

        # Save episode
        if episode_frames:
            dataset.add_episode(episode_frames)

    # Finalize dataset
    dataset.finalize()
    dataset.push_to_hub()

    env.close()
    print(f"Teleoperation data collection completed: {dataset.num_episodes} episodes")

    return dataset
```

## Training in Simulation

### Simulation Training Script

```python
#!/usr/bin/env python3
"""
Training script for simulation environments
"""

import argparse
import torch
from lerobot.scripts.lerobot_train import train

def train_in_simulation(
    dataset_id,
    policy_type="act",
    output_dir="./outputs/simulation_training",
    steps=10000,
    batch_size=32
):
    """Train policy in simulation environment."""

    print(f"🤖 Training in simulation")
    print(f"Dataset: {dataset_id}")
    print(f"Policy: {policy_type}")
    print(f"Steps: {steps}")
    print(f"Batch size: {batch_size}")

    # Training arguments
    args = [
        f"--dataset.repo_id={dataset_id}",
        f"--policy.type={policy_type}",
        f"--output_dir={output_dir}",
        f"--job_name=simulation_{policy_type}",
        f"--policy.device=cuda",
        f"--wandb.enable=true",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--dataset.video_backend=pyav"
    ]

    # Start training
    try:
        train(args)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train in simulation")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset ID")
    parser.add_argument("--policy", type=str, default="act", help="Policy type")
    parser.add_argument("--output-dir", type=str, default="./outputs/simulation_training",
                       help="Output directory")
    parser.add_argument("--steps", type=int, default=10000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    train_in_simulation(
        dataset_id=args.dataset,
        policy_type=args.policy,
        output_dir=args.output_dir,
        steps=args.steps,
        batch_size=args.batch_size
    )
```

### Simulation Evaluation

```python
def evaluate_in_simulation(policy_path, env_type, num_episodes=10):
    """Evaluate trained policy in simulation."""

    print(f"🔍 Evaluating policy in simulation")
    print(f"Policy: {policy_path}")
    print(f"Environment: {env_type}")
    print(f"Episodes: {num_episodes}")

    # Load policy
    from lerobot.policies.act.policy import ACTPolicy
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.eval()

    # Create environment
    env = create_simulation_env(env_type)

    # Evaluation metrics
    total_rewards = []
    success_rates = []

    for episode in range(num_episodes):
        print(f"Evaluating episode {episode + 1}/{num_episodes}")

        obs, info = env.reset()
        episode_reward = 0
        episode_success = False

        while True:
            # Get policy action
            with torch.no_grad():
                action = policy.select_action(obs)

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward

            # Check termination
            if terminated or truncated:
                episode_success = info.get("success", False)
                break

        total_rewards.append(episode_reward)
        success_rates.append(episode_success)

        print(f"  Episode {episode + 1}: Reward={episode_reward:.3f}, Success={episode_success}")

    # Calculate final metrics
    avg_reward = sum(total_rewards) / len(total_rewards)
    success_rate = sum(success_rates) / len(success_rates)

    print(f"\n📊 Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"  Success Rate: {success_rate:.3f}")
    print(f"  Episodes Evaluated: {len(total_rewards)}")

    env.close()

    return {
        "avg_reward": avg_reward,
        "success_rate": success_rate,
        "episode_rewards": total_rewards,
        "episode_successes": success_rates
    }
```

## Troubleshooting

### Common Simulation Issues

#### MuJoCo Issues

```bash
# Check MuJoCo installation
python -c "import mujoco; print('MuJoCo installed successfully')"

# Check model loading
python -c "import mujoco; model = mujoco.MjModel.from_xml_path('model.xml'); print('Model loaded successfully')"
```

#### Isaac Sim Issues

```bash
# Check Isaac Sim installation
python -c "from isaac_sim import IsaacSim; print('Isaac Sim installed successfully')"

# Check scene loading
python -c "from isaac_sim import IsaacSim; sim = IsaacSim(); sim.load_scene('scene.usd'); print('Scene loaded successfully')"
```

#### Gym-HIL Issues

```bash
# Check gym-hil installation
python -c "import gym_hil; print('Gym-HIL installed successfully')"

# Check environment creation
python -c "import gymnasium as gym; import gym_hil; env = gym.make('gym_hil/PandaPickCubeSpacemouse-v0'); print('Environment created successfully')"
```

### Performance Optimization

#### Parallel Environments

```python
from multiprocessing import Pool

def run_parallel_simulation(num_processes=4, num_episodes=100):
    """Run simulation in parallel processes."""

    episodes_per_process = num_episodes // num_processes

    def collect_episodes(process_id):
        """Collect episodes in single process."""
        env = create_simulation_env()
        episodes = []

        for episode in range(episodes_per_process):
            obs, info = env.reset()
            episode_frames = []

            while True:
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)

                episode_frames.append({
                    "observation": obs,
                    "action": action,
                    "reward": reward,
                    "done": terminated or truncated
                })

                if terminated or truncated:
                    break

            episodes.append(episode_frames)

        env.close()
        return episodes

    # Run in parallel
    with Pool(num_processes) as pool:
        results = pool.map(collect_episodes, range(num_processes))

    # Combine results
    all_episodes = []
    for process_episodes in results:
        all_episodes.extend(process_episodes)

    return all_episodes
```

#### Memory Optimization

```python
def optimize_simulation_memory():
    """Optimize memory usage in simulation."""

    # Use smaller batch sizes
    batch_size = 16

    # Process data in chunks
    chunk_size = 1000

    # Clear unused variables
    import gc
    gc.collect()

    # Use mixed precision
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
```

### Debug Mode

```python
def debug_simulation():
    """Debug simulation environment."""

    # Enable debug logging
    import logging
    logging.basicConfig(level=logging.DEBUG)

    # Create environment with debug info
    env = create_simulation_env(debug=True)

    # Run single episode with detailed output
    obs, info = env.reset()
    print(f"Initial observation: {obs}")
    print(f"Initial info: {info}")

    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(f"Step {step}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Info: {info}")

        if terminated or truncated:
            break

    env.close()
```

## Conclusion

This guide provides comprehensive coverage of LeRobot's simulation environment capabilities. Key takeaways:

- Use MuJoCo for physics-based manipulation tasks
- Leverage Isaac Sim for high-fidelity robotics simulation
- Utilize Gym-HIL for human-in-the-loop training
- Implement custom environments for specific needs
- Collect data efficiently in simulation
- Train and evaluate policies in simulation before real-world deployment

For specific simulation setups or advanced configurations, refer to the individual documentation pages in the LeRobot documentation.
noteId: "8e738540b09011f08bd0898d85b76daa"
tags: []

---

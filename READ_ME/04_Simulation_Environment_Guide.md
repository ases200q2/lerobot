---
noteId: "a0f46520e2ba11f08bd0898d85b76daa"
tags: []
---

# LeRobot Simulation Environment Guide

## Overview

LeRobot provides comprehensive integration with multiple simulation environments, enabling robust testing, data collection, and policy training in virtual environments. This guide covers integration with MuJoCo, NVIDIA Isaac, MetaWorld, LIBERO, and custom simulation frameworks through the Gym-HIL (Human-in-the-Loop) system.

## Supported Simulation Environments

### 1. Gym-HIL (Human-in-the-Loop)

**Primary simulation framework** designed for robotics learning with built-in teleoperation support.

#### Key Features

- **MuJoCo Physics**: High-fidelity physics simulation
- **Multi-Robot Support**: Franka Panda, SO100, ALOHA configurations
- **Teleoperation Integration**: Native support for SpaceMouse, gamepad, keyboard
- **Camera Rendering**: Multiple camera viewpoints with configurable resolution
- **Task Variety**: Pick-and-place, assembly, manipulation tasks

#### Installation

```bash
# Install Gym-HIL (included with LeRobot)
cd gym-hil
pip install -e .

# Install MuJoCo if not already installed
pip install mujoco

# Install additional dependencies
pip install opencv-python imageio ffmpeg-python
```

### 2. MetaWorld

Benchmark suite for diverse manipulation tasks with 50 different tasks.

#### Task Categories

- **ML1**: Single task with multiple environments
- **ML10**: 10 different manipulation tasks
- **ML45**: 45 diverse manipulation challenges

#### Installation

```bash
# Install MetaWorld
pip install metaworld

# Install Gymnasium (required)
pip install gymnasium

# Test installation
python -c "import metaworld; print('MetaWorld installed successfully')"
```

### 3. LIBERO

Language-conditioned manipulation tasks with visual and natural language instructions.

#### Benchmark Suites

- **LIBERO-10**: 10 basic manipulation tasks
- **LIBERO-90**: 90 diverse manipulation challenges
- **LIBERO-Spatial**: Spatial reasoning tasks
- **LIBERO-Object**: Object-centric manipulation
- **LIBERO-Goal**: Goal-oriented tasks

#### Installation

```bash
# Install LIBERO dependencies
pip install huggingface-hub
pip install Pillow numpy torch torchvision

# Test installation
python -c "import libero; print('LIBERO installed successfully')"
```

### 4. NVIDIA Isaac Gym (Experimental)

High-performance parallel simulation for large-scale training.

#### Features

- **GPU Acceleration**: Parallel simulation on GPU
- **Massive Parallelism**: Thousands of environments simultaneously
- **Realistic Physics**: PhysX-based simulation
- **Domain Randomization**: Extensive randomization capabilities

#### Installation

```bash
# Install Isaac Gym (requires NVIDIA GPU)
# Download from: https://developer.nvidia.com/isaac-gym
# Follow official installation instructions

# Install Python bindings
cd isaac-gym/python
pip install -e .
```

## Gym-HIL Integration

### Core Architecture

```
Gym-HIL Environment
├── MuJoCo Physics Engine
├── Robot Models (Panda, SO100, etc.)
├── Camera System
├── Task Definitions
└── Teleoperation Interfaces
    ├── SpaceMouse
    ├── Gamepad
    ├── Keyboard
    └── Custom Controllers
```

### Environment Configuration

```python
# Environment configuration structure
@dataclass
class HILSerlRobotEnvConfig:
    # Basic environment settings
    env_type: str = "hil"
    task: str = "PandaPickCubeBase-v0"

    # Wrapper configuration
    wrapper:
        control_mode: str = "spacemouse"  # "gamepad", "keyboard", "spacemouse"
        use_gripper: bool = True
        resize_size: List[int] = [128, 128]
        add_joint_velocity_to_observation: bool = True
        gripper_penalty: float = -0.02
        display_cameras: bool = False
        control_time_s: float = 150.0
        reset_time_s: float = 150.0
        fixed_reset_joint_positions: List[float] = [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]

    # Observation and action spaces
    features:
        "observation.images.front": {"type": "VISUAL", "shape": [3, 128, 128]}
        "observation.images.wrist": {"type": "VISUAL", "shape": [3, 128, 128]}
        "observation.state": {"type": "STATE", "shape": [18]}
        "action": {"type": "ACTION", "shape": [4]}
```

### Available Tasks

#### Panda Tasks

- **PandaPickCubeBase-v0**: Basic cube picking
- **PandaPickCubeGamepad-v0**: Gamepad teleoperation
- **PandaPickCubeKeyboard-v0**: Keyboard control
- **PandaPickCubeSpacemouse-v0**: SpaceMouse control

#### Custom Task Creation

```python
# Create custom task in Gym-HIL
import gym_hil
from gym_hil.wrappers.factory import make_env

# Custom environment creation
def create_custom_task():
    env = make_env(
        env_id="gym_hil/PandaPickCubeBase-v0",
        render_mode="human",
        image_obs=True,
        use_gripper=True,
        random_block_position=True,
        wrapper_kwargs={
            "control_mode": "spacemouse",
            "resize_size": [224, 224],
            "control_time_s": 200.0,
            "reset_time_s": 100.0
        }
    )
    return env
```

### Environment Usage Examples

#### Basic Environment Creation

```python
import gymnasium as gym
import gym_hil

# Create basic environment
env = gym.make(
    "gym_hil/PandaPickCubeSpacemouse-v0",
    render_mode="human",
    image_obs=True,
    use_gripper=True,
    random_block_position=True
)

# Test environment
obs, info = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

#### Data Collection in Simulation

```bash
# Collect demonstration data in simulation
lerobot-record \
    --env.type=hill \
    --env.task=PandaPickCubeSpacemouse-v0 \
    --env.wrapper.control_mode=spacemouse \
    --env.wrapper.use_gripper=true \
    --env.wrapper.resize_size=[224,224] \
    --dataset.repo_id=${HF_USER}/sim_demonstrations \
    --dataset.num_episodes=100 \
    --teleop.type=spacemouse
```

#### Policy Training in Simulation

```bash
# Train policy purely in simulation
lerobot-train \
    --env.type=hill \
    --env.task=PandaPickCubeBase-v0 \
    --policy.type=act \
    --dataset.repo_id=sim_training_data \
    --output_dir=./outputs/sim_policy \
    --batch_size=32 \
    --steps=200000
```

## MetaWorld Integration

### Environment Setup

```python
# MetaWorld environment creation
import gymnasium as gym
import metaworld

# Create ML10 environment
env = gym.make("mw-ml10-v0")

# Create specific task
env.set_task(metaworld.ML10.get_train_tasks()["pick-place"])

# Reset environment
obs, info = env.reset()
```

### Task Configuration

```python
# MetaWorld task configuration
from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE

# Available tasks
tasks = [
    "button-press-topdown-v2-goal-observable",
    "button-press-v2-goal-observable",
    "button-press-wall-v2-goal-observable",
    "reach-v2-goal-observable",
    "push-v2-goal-observable",
    "pick-place-v2-goal-observable",
    "door-open-v2-goal-observable",
    "drawer-open-v2-goal-observable",
    "drawer-close-v2-goal-observable",
    "shelf-place-v2-goal-observable"
]

# Create task-specific environment
for task_name in tasks:
    env = gym.make(f"mw-{task_name}")
    obs, info = env.reset()
    # Task-specific training or evaluation
```

### Integration with LeRobot

```python
# LeRobot MetaWorld environment wrapper
from lerobot.envs.metaworld import MetaWorldEnv

# Create MetaWorld environment for LeRobot
env_config = {
    "task": "pick-place-v2-goal-observable",
    "observation_type": "pixels",
    "image_size": [224, 224],
    "frame_stack": 1,
    "action_repeat": 1,
    "seed": 42
}

env = MetaWorldEnv(env_config)
```

## LIBERO Integration

### Environment Creation

```python
# LIBERO environment setup
import libero

# Create LIBERO environment
env = libero.create_env(
    task_suite="libero_10",
    task_id=0,
    image_size=(224, 224),
    cameras=["agentview_image", "robot0_eye_in_hand_image"],
    init_states=True,
    seed=42
)
```

### Task Configuration

```python
# LIBERO task suite configuration
task_suites = {
    "libero_10": {
        "num_tasks": 10,
        "description": "Basic manipulation tasks",
        "cameras": ["agentview_image"]
    },
    "libero_90": {
        "num_tasks": 90,
        "description": "Comprehensive manipulation benchmark",
        "cameras": ["agentview_image", "robot0_eye_in_hand_image"]
    },
    "libero_spatial": {
        "num_tasks": 10,
        "description": "Spatial reasoning tasks",
        "cameras": ["agentview_image", "robot0_eye_in_hand_image"]
    },
    "libero_object": {
        "num_tasks": 12,
        "description": "Object-centric manipulation",
        "cameras": ["agentview_image"]
    },
    "libero_goal": {
        "num_tasks": 10,
        "description": "Goal-oriented manipulation",
        "cameras": ["agentview_image", "robot0_eye_in_hand_image"]
    }
}
```

### Language-Conditioned Training

```python
# Language-conditioned policy training
from libero.datasets import get_dataset
from libero.lifelong import get_task_groups

# Load LIBERO dataset
dataset = get_dataset(
    suite_name="libero_90",
    data_dir="./libero_data",
    split="train"
)

# Get task descriptions
task_groups = get_task_groups("libero_90")
for task_id, task_info in task_groups.items():
    print(f"Task {task_id}: {task_info['description']}")

# Create language-conditioned environment
env = libero.create_env(
    task_suite="libero_90",
    task_id=0,
    include_language_instruction=True
)
```

## Domain Randomization

### Visual Randomization

```python
# Visual domain randomization configuration
visual_randomization = {
    "camera_parameters": {
        "random_position": True,
        "position_range": [-0.1, 0.1],
        "random_rotation": True,
        "rotation_range": [-0.2, 0.2],
        "random_fov": True,
        "fov_range": [45, 75]
    },
    "lighting": {
        "random_intensity": True,
        "intensity_range": [0.7, 1.3],
        "random_color": True,
        "color_range": [0.8, 1.2]
    },
    "textures": {
        "random_objects": True,
        "texture_variations": 10,
        "random_colors": True,
        "color_variations": 20
    },
    "post_processing": {
        "gaussian_noise": {"mean": 0, "std": 0.01},
        "motion_blur": {"kernel_size": 3},
        "compression_artifacts": {"quality": 85}
    }
}
```

### Physics Randomization

```python
# Physics domain randomization
physics_randomization = {
    "object_properties": {
        "mass_range": [0.5, 2.0],
        "friction_range": [0.3, 0.9],
        "restitution_range": [0.0, 0.3],
        "inertia_scale": [0.8, 1.2]
    },
    "robot_parameters": {
        "joint_damping_range": [0.005, 0.015],
        "joint_friction_range": [0.001, 0.005],
        "actuator_delay_range": [0.01, 0.05]
    },
    "environment": {
        "gravity_variation": [-0.1, 0.1],
        "air_resistance": [0.0, 0.001]
    }
}
```

### Implementation in Gym-HIL

```python
# Domain randomization in Gym-HIL
from gym_hil.wrappers.domain_randomization import DomainRandomizationWrapper

# Create environment with domain randomization
env = make_env(
    env_id="gym_hil/PandaPickCubeBase-v0",
    wrapper_kwargs={
        "enable_domain_randomization": True,
        "visual_randomization": visual_randomization,
        "physics_randomization": physics_randomization,
        "randomization_frequency": "episode"  # "step", "episode", "session"
    }
)
```

## Real-to-Sim Transfer

### System Identification

```python
# Real robot system identification
from lerobot.calibration import SystemIdentifier

# Collect real robot data
identifier = SystemIdentifier(
    robot_type="so100",
    calibration_data_path="./real_robot_calibration"
)

# Estimate physical parameters
parameters = identifier.estimate_parameters()
print(f"Identified parameters: {parameters}")

# Create sim-to-real configuration
sim_config = {
    "robot_config": {
        "joint_damping": parameters["damping"],
        "joint_friction": parameters["friction"],
        "mass_matrix": parameters["mass_matrix"]
    }
}
```

### Sim-to-Real Adaptation

```python
# Sim-to-real domain adaptation
from lerobot.adaptation import Sim2RealAdapter

# Train policy in simulation
sim_policy = train_policy_in_simulation()

# Adapt to real robot
adapter = Sim2RealAdapter(
    sim_policy=sim_policy,
    real_robot_data_path="./real_robot_data",
    adaptation_method="daib"  # Domain Adversarial Invariant Bottleneck
)

# Train adaptation layers
adapted_policy = adapter.train_adaptation()

# Deploy on real robot
deploy_policy_on_robot(adapted_policy)
```

### Fine-tuning Pipeline

```bash
# Fine-tune sim-trained policy on real data
lerobot-train \
    --policy.type=act \
    --policy.pretrained_path=./outputs/sim_policy \
    --dataset.repo_id=${HF_USER}/real_world_fine_tuning \
    --training.fine_tuning=true \
    --training.learning_rate=1e-5 \
    --training.steps=10000 \
    --output_dir=./outputs/fine_tuned_policy
```

## Vectorized Environments

### Parallel Environment Creation

```python
# Create vectorized environments for parallel training
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv

def make_env():
    return gym.make(
        "gym_hil/PandaPickCubeBase-v0",
        image_obs=True,
        use_gripper=True
    )

# Synchronous vectorized environments
num_envs = 8
envs = SyncVectorEnv([make_env for _ in range(num_envs)])

# Asynchronous vectorized environments (faster)
envs_async = AsyncVectorEnv([make_env for _ in range(num_envs)])
```

### Parallel Training Configuration

```yaml
# Vectorized environment training
vectorized_training:
  num_envs: 16
  vectorization_type: "async" # "sync" or "async"
  in_series: 1 # Batch size per environment

  # Environment settings
  env_kwargs:
    image_obs: true
    use_gripper: true
    random_block_position: true

  # Training optimization
  shared_memory: true
  worker_timeout: 30
  prefetch_factor: 2
```

### Performance Optimization

```python
# High-performance environment configuration
def create_optimized_env():
    return make_env(
        env_id="gym_hil/PandaPickCubeBase-v0",
        render_mode=None,  # No rendering for speed
        image_obs=True,
        use_gripper=True,
        wrapper_kwargs={
            "control_mode": "policy",
            "resize_size": [128, 128],  # Smaller images
            "control_time_s": 100.0,    # Faster control
            "reset_time_s": 50.0
        }
    )

# Vectorized for parallel data collection
num_envs = 32
envs = AsyncVectorEnv([create_optimized_env for _ in range(num_envs)])
```

## Benchmarking and Evaluation

### Standard Evaluation Protocols

```python
# Standard evaluation for different benchmarks
evaluation_protocols = {
    "metaworld_ml10": {
        "num_tasks": 10,
        "episodes_per_task": 50,
        "success_threshold": 0.8,
        "metrics": ["success_rate", "episode_reward", "action_smoothness"]
    },
    "libero_90": {
        "num_tasks": 90,
        "episodes_per_task": 20,
        "success_threshold": 0.7,
        "metrics": ["success_rate", "goal_distance", "language_conditioning"]
    },
    "gym_hil": {
        "num_tasks": 1,
        "episodes_per_task": 100,
        "success_threshold": 0.9,
        "metrics": ["success_rate", "efficiency", "safety"]
    }
}
```

### Evaluation Script

```python
# Comprehensive evaluation script
def evaluate_policy(policy, env_config, num_episodes=50):
    env = create_environment(env_config)
    results = {
        "success_rate": 0,
        "average_reward": 0,
        "episode_lengths": [],
        "successes": 0,
        "failures": 0
    }

    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        episode_length = 0

        while True:
            action = policy.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            if terminated or truncated:
                if reward > 0:  # Success condition
                    results["successes"] += 1
                else:
                    results["failures"] += 1

                results["episode_lengths"].append(episode_length)
                break

    results["success_rate"] = results["successes"] / num_episodes
    results["average_reward"] = episode_reward / num_episodes

    return results
```

### Cross-Environment Evaluation

```bash
# Evaluate policy across multiple environments
for env in "gym_hil" "metaworld" "libero"; do
    lerobot-eval \
        --policy.path=./outputs/trained_policy \
        --env.type=$env \
        --num_episodes=100 \
        --output_dir=./evaluation_results/$env \
        --render=false
done
```

## Advanced Features

### Custom Environment Integration

```python
# Custom environment wrapper
class CustomEnvironment(gym.Env):
    def __init__(self, config):
        self.config = config
        self.physics_engine = self._initialize_physics()
        self.robot_model = self._load_robot_model()
        self.task_space = self._create_task_space()

    def step(self, action):
        # Custom physics simulation
        next_state, reward, done = self._simulate_step(action)

        # Process observations
        obs = self._get_observations()

        return obs, reward, done, {}

    def reset(self):
        # Reset simulation state
        self._reset_simulation()
        return self._get_observations()
```

### Real-time Visualization

```python
# Real-time visualization with Rerun
import rerun as rr

def visualize_episode(policy, env, episode_id):
    rr.init("simulation_visualization", spawn=True)
    rr.set_time_sequence("episode", episode_id)

    obs, info = env.reset()
    step = 0

    while True:
        # Log observations
        rr.log("observations/images", rr.Image(obs["images"]))
        rr.log("observations/state", rr.TimeSeriesScalar(obs["state"]))

        # Get and log action
        action = policy.select_action(obs)
        rr.log("actions", rr.TimeSeriesScalar(action))

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Log environment state
        rr.log("environment/robot_state", rr.Transform3D(env.robot.get_pose()))
        rr.log("environment/object_poses", rr.Points3D(env.get_object_positions()))

        step += 1
        if terminated or truncated:
            break
```

### Simulation-to-Deployment Pipeline

```python
# End-to-end sim-to-deployment pipeline
class Sim2RealPipeline:
    def __init__(self, config):
        self.config = config
        self.sim_env = self._create_simulation()
        self.real_robot = self._initialize_real_robot()

    def train_in_simulation(self):
        """Train policy entirely in simulation"""
        policy = self._create_policy()

        # Train with domain randomization
        for epoch in range(self.config.num_epochs):
            episode_data = self._collect_simulation_episodes(policy)
            policy.train_step(episode_data)

        return policy

    def adapt_to_real(self, sim_policy):
        """Adapt simulation policy to real robot"""
        adapter = Sim2RealAdapter(sim_policy, self.real_robot)
        real_policy = adapter.train_adaptation()
        return real_policy

    def deploy_and_fine_tune(self, policy):
        """Deploy and fine-tune on real robot"""
        # Collect real robot data
        real_data = self._collect_real_data(policy)

        # Fine-tune policy
        policy.fine_tune(real_data)

        return policy
```

## Performance Optimization

### Memory Management

```python
# Efficient memory management for large-scale simulation
class EfficientEnvironment:
    def __init__(self, config):
        self.config = config
        self.frame_buffer = []
        self.max_buffer_size = 1000

    def step(self, action):
        # Process step
        next_state, reward, done, info = self._physics_step(action)

        # Manage memory efficiently
        if len(self.frame_buffer) >= self.max_buffer_size:
            self.frame_buffer.pop(0)

        self.frame_buffer.append(next_state)

        return next_state, reward, done, info
```

### GPU Acceleration

```python
# GPU-accelerated simulation (Isaac Gym)
def create_gpu_environment():
    import isaacgym

    # Create parallelized GPU environment
    env = isaacgym.GymEnv(
        num_envs=256,  # 256 parallel environments
        sim_device="cuda:0",
        rl_device="cuda:0",
        graphics_device_id=0
    )

    return env
```

### Batch Processing

```python
# Batch processing for efficiency
def batch_process_observations(observations, batch_size=32):
    """Process observations in batches for efficiency"""
    processed = []

    for i in range(0, len(observations), batch_size):
        batch = observations[i:i+batch_size]
        processed_batch = process_batch(batch)
        processed.extend(processed_batch)

    return processed
```

This comprehensive simulation guide provides everything needed to work with LeRobot's simulation environments, from basic setup to advanced optimization and real-to-sim transfer techniques.

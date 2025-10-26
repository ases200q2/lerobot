# Isaac Sim Integration Guide

This document provides a comprehensive guide for extending the SpaceMouse IL pipeline to work with NVIDIA Isaac Sim environments.

## Table of Contents

- [Overview](#overview)
- [Why This Pipeline is Isaac Sim Ready](#why-this-pipeline-is-isaac-sim-ready)
- [Key Differences: gym-hil vs Isaac Sim](#key-differences-gym-hil-vs-isaac-sim)
- [Integration Strategy](#integration-strategy)
- [Step-by-Step Migration](#step-by-step-migration)
- [Code Modifications Required](#code-modifications-required)
- [Testing and Validation](#testing-and-validation)
- [Best Practices](#best-practices)

## Overview

NVIDIA Isaac Sim provides high-fidelity physics simulation with photorealistic rendering, making it ideal for:
- **Sim-to-Real Transfer**: More realistic simulation → better real-world performance
- **Complex Scenes**: Multi-object, multi-robot scenarios
- **Advanced Sensors**: LiDAR, depth cameras, force-torque sensors
- **Scalability**: Parallel environment simulation for faster training

This pipeline is **designed from the ground up** to support Isaac Sim integration with minimal code changes.

## Why This Pipeline is Isaac Sim Ready

### 1. **Modular Architecture**

Each component is cleanly separated:
```
Data Collection → Dataset → Training → Policy → Evaluation
```

You only need to modify the **evaluation environment creation** and **observation wrapper**.

### 2. **Standard Data Format**

Uses **LeRobotDataset v3.0** format:
- Compatible with any LeRobot environment
- HuggingFace Hub integration
- Standard observation/action keys

### 3. **Configurable Observation Processing**

The `ObservationWrapper` class is designed to be environment-agnostic:
```python
class ObservationWrapper:
    def _transform_obs(self, obs):
        # Transform environment obs → policy obs
        # Easy to adapt for Isaac Sim observations
```

### 4. **Flexible Action Transformation**

Action transformation is cleanly separated:
```python
def _transform_action(self, action):
    # Transform policy action → environment action
    # Configurable for different action spaces
```

### 5. **Pure Imitation Learning**

No dependency on RL frameworks (HIL-SERL, etc.):
- **Dataset-agnostic training**: Works with any LeRobotDataset
- **Policy-agnostic**: ACT policy doesn't know about environment details
- **Clean separation**: Training and evaluation are independent

## Key Differences: gym-hil vs Isaac Sim

| Aspect | gym-hil (MuJoCo) | Isaac Sim |
|--------|------------------|-----------|
| **Physics Engine** | MuJoCo | PhysX |
| **Rendering** | OpenGL (basic) | RTX ray tracing (photorealistic) |
| **Environment API** | Gymnasium | Isaac Sim API / Gymnasium wrapper |
| **Observation Format** | `{'pixels': {...}, 'agent_pos': ...}` | Custom format (configurable) |
| **Action Space** | Typically 7-dim (xyz + rot + gripper) | Configurable |
| **Camera Images** | NumPy arrays (H, W, 3) | Can be NumPy or tensors |
| **State Information** | Simple dict | Rich structured data |
| **Initialization** | `gym.make(...)` | Isaac Sim environment setup |

## Integration Strategy

### Three Approaches (from easiest to most complex):

#### Approach 1: **Isaac Sim Gymnasium Wrapper** ⭐ Recommended

Use Isaac Sim's Gymnasium-compatible wrapper if available:

```python
import isaac_sim_gym  # Hypothetical Isaac Sim gym wrapper
env = gym.make("IsaacSim-PandaPickCube-v0")
```

**Pros:**
- Minimal code changes
- Familiar Gymnasium API
- Easy to integrate

**Cons:**
- Depends on Isaac Sim providing Gymnasium wrapper
- May have limitations in customization

#### Approach 2: **Custom Isaac Sim Wrapper**

Create a Gymnasium wrapper around Isaac Sim:

```python
class IsaacSimWrapper(gym.Env):
    def __init__(self, isaac_env):
        self.isaac_env = isaac_env
        # Define observation_space, action_space

    def step(self, action):
        # Call Isaac Sim environment
        # Transform observations to Gymnasium format
        pass
```

**Pros:**
- Full control over interface
- Can optimize for performance
- Standardizes Isaac Sim interface

**Cons:**
- Requires more implementation
- Need to handle Isaac Sim API details

#### Approach 3: **Direct Isaac Sim Integration**

Modify evaluation script to work directly with Isaac Sim API:

**Pros:**
- Maximum performance
- Access to all Isaac Sim features

**Cons:**
- Significant code changes
- Breaks compatibility with gym-hil

## Step-by-Step Migration

### Phase 1: Environment Setup

#### 1.1 Install Isaac Sim

```bash
# Download and install Isaac Sim from NVIDIA
# https://developer.nvidia.com/isaac-sim

# Install IsaacSim Python packages
pip install isaacsim-python
```

#### 1.2 Create Isaac Sim Environment

Follow Isaac Sim documentation to create a Panda pick-and-place environment similar to gym-hil's PandaPickCube.

### Phase 2: Data Collection (Optional)

You have two options:

**Option A: Use Existing gym-hil Data** ✅ Recommended
- Train on gym-hil data
- Evaluate on Isaac Sim
- Fine-tune if needed

**Option B: Collect New Isaac Sim Data**
- Modify `scripts/collect_data.py` to use Isaac Sim environment
- Collect demonstrations in Isaac Sim (more realistic)
- Train on Isaac Sim data

### Phase 3: Training (No Changes Needed)

The training pipeline **requires no modifications**:

```bash
# Works exactly the same
./scripts/run_training.sh username/panda_spacemouse_il_data
```

The trained policy is **environment-agnostic** and will work with Isaac Sim.

### Phase 4: Evaluation Modification

This is where Isaac Sim integration happens.

#### 4.1 Create Isaac Sim Environment Wrapper

Create `scripts/isaac_sim_env.py`:

```python
import gymnasium as gym
import numpy as np
from omni.isaac.kit import SimulationApp


class IsaacSimPandaEnv(gym.Env):
    """Gymnasium wrapper for Isaac Sim Panda environment."""

    def __init__(self, headless=False):
        # Initialize Isaac Sim
        self.simulation_app = SimulationApp({
            "headless": headless,
            "width": 1280,
            "height": 720,
        })

        # Import Isaac Sim modules (must be after SimulationApp)
        from omni.isaac.core import World
        from omni.isaac.franka import Franka
        # ... other imports

        # Create world and robot
        self.world = World(stage_units_in_meters=1.0)
        self.robot = Franka(prim_path="/World/Franka")
        # ... setup cameras, objects, etc.

        # Define spaces
        self.observation_space = gym.spaces.Dict({
            'pixels': gym.spaces.Dict({
                'front': gym.spaces.Box(0, 255, (480, 640, 3), np.uint8),
                'wrist': gym.spaces.Box(0, 255, (480, 640, 3), np.uint8),
            }),
            'agent_pos': gym.spaces.Box(-np.inf, np.inf, (7,), np.float32),
        })

        self.action_space = gym.spaces.Box(-1, 1, (7,), np.float32)

    def reset(self, seed=None, options=None):
        # Reset Isaac Sim environment
        self.world.reset()
        # ... reset robot, objects, etc.

        obs = self._get_observation()
        info = {}
        return obs, info

    def step(self, action):
        # Apply action to robot
        self.robot.apply_action(action)

        # Step simulation
        self.world.step(render=True)

        obs = self._get_observation()
        reward = self._compute_reward()
        terminated = self._check_success()
        truncated = self._check_timeout()
        info = {}

        return obs, reward, terminated, truncated, info

    def _get_observation(self):
        # Get camera images
        front_img = self._get_camera_image("front_camera")
        wrist_img = self._get_camera_image("wrist_camera")

        # Get robot state
        agent_pos = self.robot.get_joint_positions()

        obs = {
            'pixels': {
                'front': front_img,
                'wrist': wrist_img,
            },
            'agent_pos': agent_pos,
            # Add joint_velocities, ee_pose if available
        }
        return obs

    def _get_camera_image(self, camera_name):
        # Use Isaac Sim camera API
        # Convert to NumPy array (H, W, 3) uint8
        pass

    def _compute_reward(self):
        # Implement reward function
        # E.g., check if cube is grasped and lifted
        pass

    def close(self):
        self.simulation_app.close()
```

#### 4.2 Modify Evaluation Script

Update `scripts/eval_policy.py` to support Isaac Sim:

```python
# Add Isaac Sim option
parser.add_argument(
    "--use_isaac_sim",
    action="store_true",
    help="Use Isaac Sim instead of gym-hil",
)

# In main():
if args.use_isaac_sim:
    from scripts.isaac_sim_env import IsaacSimPandaEnv
    base_env = IsaacSimPandaEnv(headless=not args.gui)
else:
    # Use gym-hil as before
    import gym_hil
    base_env = gym.make("gym_hil/PandaPickCubeBase-v0", image_obs=True)
```

#### 4.3 Update Observation Wrapper (if needed)

The existing `ObservationWrapper` should work if Isaac Sim returns observations in similar format. If not, add Isaac Sim-specific transformations:

```python
class ObservationWrapper(gym.Wrapper):
    def __init__(self, env, device="cpu", show_gui=True, is_isaac_sim=False):
        super().__init__(env)
        self.device = device
        self.show_gui = show_gui
        self.is_isaac_sim = is_isaac_sim

    def _transform_obs(self, obs):
        if self.is_isaac_sim:
            # Isaac Sim-specific transformations
            return self._transform_isaac_obs(obs)
        else:
            # gym-hil transformations (existing code)
            return self._transform_gym_obs(obs)
```

### Phase 5: Testing

```bash
# Test Isaac Sim evaluation
python scripts/eval_policy.py \
    --policy_path username/panda_spacemouse_act_policy \
    --dataset_id username/panda_spacemouse_il_data \
    --use_isaac_sim \
    --episodes 10
```

## Code Modifications Required

### Minimal Changes (Approach 1: Gymnasium Wrapper)

**Files to Modify:**
1. `scripts/eval_policy.py`: Add Isaac Sim environment option (~20 lines)
2. `scripts/isaac_sim_env.py`: Create Gymnasium wrapper (~200 lines)

**Files Unchanged:**
- All training scripts ✅
- All data collection scripts ✅
- Configuration files ✅
- Policy implementations ✅

### Summary of Changes

| Component | Changes Required | Effort |
|-----------|-----------------|--------|
| **Data Collection** | Optional (can reuse gym-hil data) | 0-2 hours |
| **Training** | None | 0 hours |
| **Policy** | None | 0 hours |
| **Evaluation Environment** | Create Isaac Sim wrapper | 2-4 hours |
| **Observation Processing** | Minimal (if format similar) | 0-1 hour |
| **Action Transformation** | Minimal (if action space same) | 0-1 hour |
| **Testing & Tuning** | Validate on Isaac Sim | 2-4 hours |
| **Total** | | **4-12 hours** |

## Testing and Validation

### 1. Environment Compatibility Test

```python
# Test that Isaac Sim environment matches expected interface
env = IsaacSimPandaEnv()
obs, info = env.reset()

# Check observation format
assert 'pixels' in obs
assert 'front' in obs['pixels']
assert obs['pixels']['front'].shape == (480, 640, 3)  # or (128, 128, 3) after preprocessing

# Check action space
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print("Environment compatible!")
```

### 2. Observation Wrapper Test

```python
# Test observation wrapper
wrapped_env = ObservationWrapper(env, device="cuda", is_isaac_sim=True)
obs, _ = wrapped_env.reset()

# Check transformed observations
assert 'observation.images.front' in obs
assert obs['observation.images.front'].shape == (1, 3, 128, 128)  # (B, C, H, W)
assert 'observation.state' in obs
assert obs['observation.state'].shape == (1, 18)  # (B, state_dim)
print("Observation wrapper compatible!")
```

### 3. Policy Inference Test

```python
# Test policy on Isaac Sim observations
from lerobot.policies.factory import make_policy, make_policy_config

policy_cfg = make_policy_config("act", pretrained_path="username/model", device="cuda")
policy = make_policy(cfg=policy_cfg, ds_meta=dataset_meta)

obs, _ = wrapped_env.reset()
action = policy.select_action(obs)

assert action.shape == (4,) or action.shape == (1, 4)
print("Policy inference successful!")
```

### 4. Full Episode Test

```python
# Run complete episode
obs, _ = wrapped_env.reset()
for step in range(500):
    action = policy.select_action(obs)
    obs, reward, terminated, truncated, info = wrapped_env.step(action)
    if terminated or truncated:
        print(f"Episode finished at step {step}, reward: {reward}")
        break
```

## Best Practices

### 1. **Observation Space Consistency**

Ensure Isaac Sim observations match gym-hil format:
- **Image shape**: (H, W, 3) NumPy uint8
- **State vector**: Same dimensionality and ordering
- **Key names**: Use same keys for compatibility

### 2. **Action Space Compatibility**

If Isaac Sim uses different action space:
```python
def _transform_action_for_isaac(self, action):
    # Remap from gym-hil action space to Isaac Sim action space
    # Example: Different scaling or ordering
    return transformed_action
```

### 3. **Gradual Migration**

1. **Start with gym-hil**: Collect data and train
2. **Add Isaac Sim evaluation**: Test policy transfer
3. **Fine-tune if needed**: Collect small Isaac Sim dataset for fine-tuning
4. **Full migration**: Eventually collect all data in Isaac Sim

### 4. **Sim-to-Real Gap Mitigation**

Isaac Sim is more realistic than gym-hil, but still has sim-to-real gap:
- **Domain randomization**: Vary lighting, textures, dynamics
- **Realistic sensors**: Use depth cameras, noise models
- **Physics parameters**: Tune friction, damping to match real robot

### 5. **Performance Optimization**

Isaac Sim is more computationally intensive:
- **Headless mode**: Disable rendering for faster evaluation
- **Parallel environments**: Use Isaac Sim's multi-environment support
- **GPU utilization**: Ensure both simulation and policy use GPU efficiently

## Future Enhancements

### 1. **Multi-Task Learning**

Extend to multiple Isaac Sim tasks:
```python
tasks = [
    "IsaacSim-PandaPickCube-v0",
    "IsaacSim-PandaStackBlocks-v0",
    "IsaacSim-PandaPourWater-v0",
]

for task in tasks:
    collect_data(task)
    train_policy(task)
    evaluate_policy(task)
```

### 2. **Sim-to-Real Transfer**

Train in Isaac Sim → Deploy on real Panda robot:
```python
# Evaluation on real robot (similar API)
real_env = RealPandaEnv(robot_ip="192.168.1.10")
wrapped_env = ObservationWrapper(real_env)
evaluate_policy(wrapped_env, policy)
```

### 3. **Advanced Sensors**

Leverage Isaac Sim's advanced sensors:
- Depth cameras
- Force-torque sensors
- Tactile sensors

Update observation space accordingly:
```python
obs = {
    'pixels': {'front': rgb, 'wrist': rgb},
    'depth': {'front': depth, 'wrist': depth},
    'force_torque': ft_reading,
    'agent_pos': joint_pos,
}
```

## Conclusion

This SpaceMouse IL pipeline is **architected for Isaac Sim compatibility**:

✅ **Minimal code changes** (4-12 hours of work)
✅ **Modular design** (swap environments easily)
✅ **Standard interfaces** (Gymnasium API)
✅ **Clean separation** (training unchanged)
✅ **Proven approach** (gym-hil → Isaac Sim → Real Robot)

The path from gym-hil to Isaac Sim is **straightforward and well-defined**. Follow this guide, and you'll have Isaac Sim integration running within a day!

---

**Ready to integrate Isaac Sim? Start with Phase 1! 🚀**

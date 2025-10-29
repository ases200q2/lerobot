# Environment Analysis: PandaPickCubeSpacemouse-v0 Data Collection

This document provides a complete breakdown of the environment, reward function, end conditions, and reset logic used by `python scripts/collect_data.py`.

## Environment Overview

**Base Environment**: `PandaPickCubeGymEnv` (MuJoCo-based)  
**Registered Name**: `gym_hil/PandaPickCubeSpacemouse-v0`  
**File Location**: `gym-hil/gym_hil/envs/panda_pick_gym_env.py`  
**XML Model**: `gym-hil/gym_hil/assets/scene.xml`

## 1. Environment Reset Logic

### Robot Reset
- **Home Position**: `[0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785]` (7-DOF joint positions)
- **Reset Method**: `reset_robot()` - resets to home position

### Cube Reset
- **Fixed Position Mode** (`random_block_position=False`):
  - Cube XY position: `[0.5, 0.0]` (middle of table)
- **Random Position Mode** (`random_block_position=True`):
  - Sampling bounds: `[[0.35, -0.1], [0.45, 0.1]]`
  - Cube XY sampled uniformly within bounds

### Success Metrics Cache
After reset, the environment caches:
- `_z_init`: Initial block Z position (height)
- `_z_success`: Success threshold = `_z_init + 0.1` meters

### Configuration
From `collect_data_config.json`:
```json
"reset": {
  "fixed_reset_joint_positions": [0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785],
  "reset_time_s": 5.0,
  "control_time_s": 20.0,
  "terminate_on_success": false
}
```

## 2. Reward Function

The reward depends on the `reward_type` configuration:

### A. Sparse Reward (Default)
```python
lift = block_pos[2] - self._z_init
return float(lift > 0.1)
```
- Returns `1.0` if block is lifted >0.1m above initial height
- Returns `0.0` otherwise

### B. Dense Reward
```python
tcp_pos = self._data.sensor("2f85/pinch_pos").data
dist = np.linalg.norm(block_pos - tcp_pos)
r_close = np.exp(-20 * dist)              # Distance to block
r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)  # Lift height
r_lift = np.clip(r_lift, 0.0, 1.0)
return 0.3 * r_close + 0.7 * r_lift
```
- 30% weight on proximity to block
- 70% weight on lift height
- Continuous signal between 0.0 and 1.0

### Additional Rewards from Wrappers

**Gripper Penalty** (from `GripperPenaltyWrapper`):
- Applied when gripper state changes (open ↔ close)
- Penalty: `-0.02` (added to info dict, not reward directly)
- Affects training signal for gripper usage

**SpaceMouse Success Reward** (manual intervention):
- When SpaceMouse button pressed for SUCCESS → `reward = 1.0`

## 3. End Conditions (Episode Termination)

### A. Success Condition
```python
def _is_success(self) -> bool:
    block_pos = self._data.sensor("block_pos").data
    tcp_pos = self._data.sensor("2f85/pinch_pos").data
    dist = np.linalg.norm(block_pos - tcp_pos)
    lift = block_pos[2] - self._z_init
    return dist < 0.05 and lift > 0.1
```
- Gripper within **0.05m** of block
- Block lifted **>0.1m** above initial height

### B. Failure Condition
```python
block_pos = self._data.sensor("block_pos").data
exceeded_bounds = np.any(block_pos[:2] < (0.3, -0.15)) or \
                  np.any(block_pos[:2] > (0.5, 0.15))
```
- Block moves outside XY bounds: `[0.3, -0.15]` to `[0.5, 0.15]`

### C. Episode Termination Logic
```python
if self.terminate_on_success:
    terminated = bool(success or exceeded_bounds)
else:
    terminated = bool(exceeded_bounds)
```

**Configuration**: `terminate_on_success=False` in data collection config
- Episodes **do NOT** auto-terminate on success
- Human operator manually ends episodes using SpaceMouse buttons:
  - **SPACE**: Mark as SUCCESS (terminate with reward=1.0)
  - **C**: Mark as FAILURE (terminate with reward=0.0)
  - **R**: Re-record current episode

### D. Time Limit
- **Maximum Episode Steps**: 200 (from gym registration)
- At 10 FPS: ~20 seconds per episode
- Configuration: `"control_time_s": 20.0`

## 4. Environment Wrappers (Applied in Order)

### 1. GripperPenaltyWrapper
- **Purpose**: Penalize gripper usage
- **Penalty**: `-0.02` for state changes
- **Location**: `gym-hil/gym_hil/wrappers/hil_wrappers.py`

### 2. EEActionWrapper
- **Purpose**: Convert action to end-effector space
- **Step Sizes**: Default `{x: 0.025, y: 0.025, z: 0.025}` meters
- **Gripper Normalization**: `[0, 2]` → `[-1, 1]`
- **Location**: `gym-hil/gym_hil/wrappers/hil_wrappers.py`

### 3. SpaceMouseControlWrapper
- **Purpose**: Control robot via SpaceMouse device
- **Mode**: 5DOF (X, Y, Z, gripper)
- **Sensitivity**: 0.5
- **Deadzone**: 0.02
- **Controls**:
  - Movement: 3D translation
  - Gripper: Open/close buttons
  - Episode control: SPACE (success), C (failure), R (re-record)
- **Location**: `gym-hil/gym_hil/wrappers/spacemouse_wrapper.py`

### 4. PassiveViewerWrapper
- **Purpose**: Display camera views with UI panels
- **Show UI**: Enabled
- **Cameras**: Front view and wrist view
- **Location**: `gym-hil/gym_hil/wrappers/viewer_wrapper.py`

### 5. ResetDelayWrapper
- **Purpose**: Add delay between episodes
- **Delay**: 1.0 second
- **Location**: `gym-hil/gym_hil/wrappers/hil_wrappers.py`

## 5. Observation Space

### Images
- **Front camera**: 128×128×3 (RGB)
- **Wrist camera**: 128×128×3 (RGB)
- **Resize**: Both cropped and resized to 128×128

### State Vector
- **Agent position**: 7-DOF joint positions
- **Joint velocities**: Added via config (`add_joint_velocity_to_observation: true`)
- **Total state dim**: 14 (7 positions + 7 velocities)
- **Additional features**: May include forward kinematics (not in current config)

### Observation Structure
```python
{
    "pixels": {
        "front": np.array[128, 128, 3],  # uint8
        "wrist": np.array[128, 128, 3]   # uint8
    },
    "agent_pos": np.array[7],            # float32
    # + joint velocities if enabled
}
```

## 6. Action Space

**Dimensions**: 4 (X, Y, Z, gripper)  
**Bounds**:
- X, Y, Z: `[-1.0, 1.0]`
- Gripper: `[0.0, 2.0]` (0=close, 1=hold, 2=open)

**Action Processing**:
1. Scale XYZ by step sizes (0.025m per unit)
2. Convert gripper from `[0,2]` → `[-1,1]` for MuJoCo
3. Append zero rotations `[0, 0, 0]`
4. Final action to MuJoCo: `[x, y, z, rx, ry, rz, gripper]` (7D)

## 7. Configuration Summary

From `SPACEMOUSE_IL_DATA/configs/collect_data_config.json`:

```json
{
  "mode": "record",
  "device": "cuda",
  "env": {
    "task": "PandaPickCubeSpacemouse-v0",
    "fps": 10,
    "processor": {
      "gripper": {
        "use_gripper": true,
        "gripper_penalty": -0.02
      },
      "reset": {
        "fixed_reset_joint_positions": [0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785],
        "reset_time_s": 5.0,
        "control_time_s": 20.0,
        "terminate_on_success": false
      }
    }
  }
}
```

### Key Settings
- **FPS**: 10 Hz
- **Max episode duration**: 20 seconds (200 steps)
- **Manual episode control**: Enabled (no auto-termination)
- **Gripper penalty**: -0.02
- **Reset delay**: 5 seconds
- **Images**: 128×128×3 (front + wrist)
- **State**: 14D (7 joint positions + 7 joint velocities)

## 8. Flow Diagram

```
collect_data.py
    ↓
gym_manipulator.py
    ↓
make_robot_env() → gym.make("gym_hil/PandaPickCubeSpacemouse-v0")
    ↓
make_env() [factory.py]
    ├─ PandaPickCubeGymEnv (base env)
    └─ wrap_env() [apply wrappers in order]
        ├─ GripperPenaltyWrapper
        ├─ EEActionWrapper
        ├─ SpaceMouseControlWrapper
        ├─ PassiveViewerWrapper
        └─ ResetDelayWrapper
    ↓
control_loop()
    ↓
step_env_and_process_transition()
    ↓
Environment execution with SpaceMouse control
```

## 9. Data Collection Process

1. **Episode Start**: Robot resets to home, cube at fixed position
2. **Control Loop**: Human controls robot via SpaceMouse at 10 Hz
3. **Episode End Triggers**:
   - Manual button press (SPACE/C)
   - Block leaves bounds
   - Time limit (20s)
4. **Episode Marking**:
   - Operator presses SPACE → SUCCESS (reward=1.0)
   - Operator presses C → FAILURE (reward=0.0)
   - Operator presses R → Re-record episode
5. **Data Saving**: Observations, actions, rewards saved to dataset

## 10. References

- **Environment File**: `gym-hil/gym_hil/envs/panda_pick_gym_env.py`
- **Wrappers**: `gym-hil/gym_hil/wrappers/`
- **Registration**: `gym-hil/gym_hil/__init__.py`
- **Factory**: `gym-hil/gym_hil/wrappers/factory.py`
- **Config**: `SPACEMOUSE_IL_DATA/configs/collect_data_config.json`
- **Main Script**: `src/lerobot/rl/gym_manipulator.py`
noteId: "73f6b630b2d211f08eda6bb380d6cd3c"
tags: []

---


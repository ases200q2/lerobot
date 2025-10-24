---
noteId: "8a2d4b20c0a811f08bd0898d85b76daa"
tags: []
---

# LeRobot SpaceMouse Integration Guide

## Overview

The SpaceMouse is a 6-DOF (Degrees of Freedom) input device that provides intuitive and precise control for robot manipulation tasks. LeRobot offers comprehensive SpaceMouse integration with advanced features including configurable control modes, calibration procedures, and seamless integration with the data collection pipeline.

## SpaceMouse Hardware

### Supported Models

1. **3Dconnexion SpaceMouse Compact**
2. **3Dconnexion SpaceMouse Pro**
3. **3Dconnexion SpaceMouse Wireless**
4. **3Dconnexion SpaceMouse Enterprise**
5. **3Dconnexion SpacePilot Pro**

### Hardware Requirements

- USB 2.0 or higher connection
- Compatible operating system (Linux, macOS, Windows)
- 3Dconnexion drivers (optional, native support available)

### Installation and Setup

#### Linux Setup

```bash
# Install required dependencies
sudo apt update
sudo apt install build-essential

# Install SpaceMouse support libraries
pip install pyspacemouse
# Alternative: Install hidapi for broader compatibility
pip install hidapi

# Add user to input group for device access
sudo usermod -a -G input $USER
# Re-login required
```

#### macOS Setup

```bash
# Install required dependencies
brew install libusb

# Install SpaceMouse Python library
pip install pyspacemouse

# Test device recognition
python -c "import pyspacemouse; print(pyspacemouse.open())"
```

#### Windows Setup

```bash
# Install 3Dconnexion drivers (recommended)
# Download from: https://www.3dconnexion.com/service/drivers.html

# Install Python library
pip install pyspacemouse

# Test device connection
python -c "import pyspacemouse; print('SpaceMouse detected:', pyspacemouse.open())"
```

## Control Modes and Mapping

### 5-DOF Control Mode

The default control mode provides intuitive 5 degrees of freedom:

```python
# 5-DOF Control Mapping
control_mapping = {
    "translation_x": "move_left/right",      # Push/pull left-right
    "translation_y": "move_forward/back",    # Push/pull forward-backward
    "translation_z": "move_up/down",        # Push/pull up-down
    "rotation_x": "tilt_forward/backward",  # Rotate around X axis
    "rotation_y": "tilt_left/right"         # Rotate around Y axis
}
```

### 7-DOF Control Mode

Advanced control mode with full 7 degrees of freedom:

```python
# 7-DOF Control Mapping (with gripper)
control_mapping = {
    "translation_x": "move_left/right",
    "translation_y": "move_forward/back",
    "translation_z": "move_up/down",
    "rotation_x": "tilt_forward/backward",
    "rotation_y": "tilt_left/right",
    "rotation_z": "twist_cw/ccw",           # Rotate around Z axis
    "gripper": "button_1"                   # Gripper control via button
}
```

### Configuration Parameters

```python
@dataclass
class SpaceMouseConfig:
    # Control mode
    control_mode: str = "7dof"  # "5dof" or "7dof"

    # Sensitivity settings
    translation_sensitivity: float = 1.0
    rotation_sensitivity: float = 1.0

    # Deadzone settings
    translation_deadzone: float = 0.05
    rotation_deadzone: float = 0.05

    # Button mapping
    gripper_button: int = 1
    reset_button: int = 2
    mode_switch_button: int = 3

    # Advanced settings
    velocity_limit: float = 0.1
    acceleration_limit: float = 0.5
    filter_enabled: bool = True
    filter_strength: float = 0.8
```

## SpaceMouse Integration in LeRobot

### Core Implementation

The SpaceMouse integration is implemented in the `gym-hil` module:

```
gym-hil/
├── gym_hil/
│   ├── spacemouse/
│   │   ├── __init__.py
│   │   ├── spacemouse.py          # Core SpaceMouse wrapper
│   │   ├── spacemouse_teleop.py   # Teleoperator implementation
│   │   └── calibration.py         # Calibration utilities
│   └── ...
```

### Key Features

- **Dual Library Support**: Compatible with `pyspacemouse` and `hidapi`
- **Robust Error Handling**: Automatic reconnection and error recovery
- **Configurable Sensitivity**: Adjustable translation and rotation sensitivity
- **Button Mapping**: Customizable button assignments
- **Filtering**: Optional signal filtering for smooth control
- **Keyboard Overrides**: Keyboard backup control when SpaceMouse is unavailable

### Teleoperator Implementation

```python
class SpaceMouseTeleop(Teleoperator):
    def __init__(self, config: SpaceMouseConfig):
        self.config = config
        self.spacemouse = SpaceMouseWrapper()
        self.control_mode = config.control_mode
        self.sensitivity = {
            'translation': config.translation_sensitivity,
            'rotation': config.rotation_sensitivity
        }
        self.deadzone = {
            'translation': config.translation_deadzone,
            'rotation': config.rotation_deadzone
        }

    def get_action(self, observation) -> Dict[str, Any]:
        """Get action from SpaceMouse input"""
        spacemouse_state = self.spacemouse.get_state()

        # Apply sensitivity and deadzone
        processed_state = self._process_input(spacemouse_state)

        # Map to robot action space
        action = self._map_to_action(processed_state, observation)

        return action
```

## Usage Examples

### Basic SpaceMouse Teleoperation

```bash
# Basic SpaceMouse data collection
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --dataset.repo_id=${HF_USER}/spacemouse_basic \
    --dataset.num_episodes=10 \
    --teleop.type=spacemouse \
    --teleop.control_mode=5dof
```

### Advanced SpaceMouse Configuration

```bash
# Advanced SpaceMouse with custom settings
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 60}}" \
    --robot.id=arm_1 \
    --dataset.repo_id=${HF_USER}/spacemouse_advanced \
    --dataset.num_episodes=25 \
    --teleop.type=spacemouse \
    --teleop.control_mode=7dof \
    --teleop.translation_sensitivity=1.2 \
    --teleop.rotation_sensitivity=0.8 \
    --teleop.translation_deadzone=0.03 \
    --teleop.rotation_deadzone=0.05 \
    --teleop.velocity_limit=0.15 \
    --teleop.filter_enabled=true \
    --teleop.filter_strength=0.9
```

### Bimanual SpaceMouse Control

```bash
# Dual SpaceMouse for bimanual control
lerobot-record \
    --robot.type=bi_so100_follower \
    --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
    --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
    --robot.id=bimanual_robot \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=${HF_USER}/bimanual_spacemouse \
    --dataset.num_episodes=30 \
    --teleop.type=bi_spacemouse \
    --teleop.left_arm_spacemouse_device_id=/dev/input/event12 \
    --teleop.right_arm_spacemouse_device_id=/dev/input/event13 \
    --teleop.control_mode=7dof
```

### Configuration File Method

Create a custom configuration file:

```yaml
# configs/teleop/spacemouse_advanced.yaml
teleop:
  type: spacemouse
  control_mode: "7dof"

  # Sensitivity settings
  translation_sensitivity: 1.0
  rotation_sensitivity: 0.8

  # Deadzone settings
  translation_deadzone: 0.05
  rotation_deadzone: 0.03

  # Button mapping
  gripper_button: 1
  reset_button: 2
  emergency_stop_button: 3

  # Performance settings
  velocity_limit: 0.1
  acceleration_limit: 0.5
  filter_enabled: true
  filter_strength: 0.8

  # Advanced options
  auto_calibrate: true
  keyboard_override: true
  verbose_logging: false
```

Then use the configuration:

```bash
# Using configuration file
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.config_path=configs/teleop/spacemouse_advanced.yaml \
    --dataset.repo_id=${HF_USER}/config_based_spacemouse \
    --dataset.num_episodes=20
```

## Calibration Procedures

### Automatic Calibration

```bash
# Run automatic calibration
lerobot-calibrate \
    --teleop.type=spacemouse \
    --calibration.duration=10 \
    --calibration.output_file=spacemouse_calibration.json
```

### Manual Calibration

```python
# Manual calibration script
import numpy as np
from gym_hil.spacemouse.calibration import SpaceMouseCalibrator

# Create calibrator
calibrator = SpaceMouseCalibrator()

# Collect calibration data
print("Move SpaceMouse in all directions for 10 seconds...")
calibration_data = calibrator.collect_calibration_data(duration=10.0)

# Compute calibration parameters
calibration_params = calibrator.compute_calibration(calibration_data)

# Save calibration
calibrator.save_calibration(calibration_params, "spacemouse_calibration.json")
print("Calibration saved successfully!")
```

### Calibration Parameters

```json
{
  "translation_bias": [0.001, -0.002, 0.0005],
  "rotation_bias": [0.0001, -0.0003, 0.0002],
  "translation_scale": [1.05, 0.98, 1.02],
  "rotation_scale": [1.01, 0.99, 1.03],
  "deadzone_translation": 0.03,
  "deadzone_rotation": 0.04,
  "filter_coefficient": 0.85,
  "velocity_limits": {
    "translation": 0.15,
    "rotation": 1.0
  }
}
```

## Testing and Verification

### Basic Functionality Test

```bash
# Test SpaceMouse connection
python -c "
import pyspacemouse
try:
    spacemouse = pyspacemouse.open()
    print('✓ SpaceMouse connected successfully')
    print(f'  Device: {spacemouse.get_device_info()}')
except Exception as e:
    print(f'✗ SpaceMouse connection failed: {e}')
"
```

### Real-time Control Test

```bash
# Test real-time control
python -c "
from gym_hil.spacemouse.spacemouse_teleop import SpaceMouseTeleop
from gym_hil.spacemouse.config import SpaceMouseConfig

config = SpaceMouseConfig(control_mode='7dof')
teleop = SpaceMouseTeleop(config)

print('Testing SpaceMouse control... Press Ctrl+C to exit')
try:
    while True:
        action = teleop.get_action({})
        print(f'Action: {action}')
except KeyboardInterrupt:
    print('Test completed')
"
```

### Integration Test

```bash
# Full integration test with robot
lerobot-teleoperate \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=spacemouse \
    --teleop.control_mode=7dof \
    --test.duration=30
```

## Comparison with Other Teleoperation Devices

### SpaceMouse vs Gamepad

| Feature            | SpaceMouse             | Gamepad (Xbox/PS4)            |
| ------------------ | ---------------------- | ----------------------------- |
| Degrees of Freedom | 6-DOF + buttons        | 2-DOF analog sticks + buttons |
| Precision          | High (continuous)      | Medium (discrete levels)      |
| Learning Curve     | Moderate               | Low                           |
| Intuitive Control  | Excellent for 3D tasks | Good for navigation           |
| Cost               | High ($200-600)        | Low ($40-60)                  |
| Battery Life       | Excellent (wired)      | Good (20-40 hours)            |
| Multi-axis Control | Simultaneous           | Limited                       |

### SpaceMouse vs Keyboard

| Feature              | SpaceMouse      | Keyboard + Mouse     |
| -------------------- | --------------- | -------------------- |
| Degrees of Freedom   | 6-DOF + buttons | 3-DOF (WASD + mouse) |
| Precision            | High            | Medium               |
| Learning Curve       | Moderate        | Low                  |
| Simultaneous Control | Full 6-DOF      | Limited              |
| Fatigue              | Low             | High (extended use)  |
| Cost                 | High            | None (included)      |
| Portability          | Medium          | High                 |

### Use Case Recommendations

#### SpaceMouse is Best For:

- **Precise manipulation tasks**
- **3D positioning and orientation**
- **Fine motor control**
- **Professional robot operation**
- **Long teleoperation sessions**

#### Gamepad is Better For:

- **Navigation tasks**
- **Gripper control**
- **Quick teleoperation**
- **Training beginners**
- **Budget constraints**

#### Keyboard is Suitable For:

- **Simple positioning**
- **Emergency control**
- **Testing and debugging**
- **Backup control**
- **Very precise, slow movements**

## Advanced Features

### Custom Control Mapping

```python
# Custom control mapping for specific tasks
class CustomSpaceMouseMapping(SpaceMouseTeleop):
    def __init__(self, config):
        super().__init__(config)
        self.custom_mapping = {
            # Pick and place specific mapping
            "approach": {
                "translation": [0, 1, 0],      # Forward only
                "rotation": [0, 0, 0]          # No rotation
            },
            "grasp": {
                "translation": [0, 0, 0],      # No movement
                "rotation": [0, 0, 0],         # No rotation
                "gripper": "close"             # Close gripper
            },
            "lift": {
                "translation": [0, 0, 1],      # Up only
                "rotation": [0, 0, 0]          # No rotation
            }
        }

    def get_custom_action(self, mode: str, observation):
        """Get action for custom mode"""
        base_action = self.get_action(observation)
        mapping = self.custom_mapping.get(mode, {})

        # Apply custom mapping
        for key, value in mapping.items():
            if key in base_action:
                base_action[key] = value

        return base_action
```

### Multi-Device Support

```bash
# Configure multiple SpaceMouse devices
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=multi_spacemouse \
    --teleop.primary_device=/dev/input/event12 \
    --teleop.secondary_device=/dev/input/event13 \
    --teleop.control_mode=hybrid \
    --dataset.repo_id=${HF_USER}/multi_device_test
```

### Real-time Signal Processing

```python
# Advanced filtering for smooth control
class FilteredSpaceMouse(SpaceMouseWrapper):
    def __init__(self, config):
        super().__init__()
        self.filter_type = config.filter_type  # "lowpass", "kalman", "median"
        self.filter_params = config.filter_params

    def get_filtered_state(self):
        """Get filtered SpaceMouse state"""
        raw_state = self.get_state()

        if self.filter_type == "lowpass":
            return self._apply_lowpass_filter(raw_state)
        elif self.filter_type == "kalman":
            return self._apply_kalman_filter(raw_state)
        elif self.filter_type == "median":
            return self._apply_median_filter(raw_state)
        else:
            return raw_state
```

## Troubleshooting

### Common Issues and Solutions

#### Device Not Detected

```bash
# Check if device is recognized
ls /dev/input/by-id/ | grep -i 3dconnexion

# Check device permissions
sudo chmod 666 /dev/input/event*

# Test with hidapi (alternative library)
python -c "
import hid
devices = hid.enumerate()
for device in devices:
    if '3Dconnexion' in str(device.get('product_string', '')):
        print(f'Found: {device}')
"
```

#### Laggy Response

```python
# Optimize performance settings
config = SpaceMouseConfig(
    control_mode="7dof",
    translation_sensitivity=1.5,      # Increase for faster response
    rotation_sensitivity=1.2,
    translation_deadzone=0.02,       # Reduce for finer control
    rotation_deadzone=0.02,
    filter_enabled=False,           # Disable filter for minimum latency
    velocity_limit=0.2              # Increase speed limit
)
```

#### Drift Issues

```python
# Recalibrate to fix drift
calibrator = SpaceMouseCalibrator()
calibration_params = calibrator.auto_calibrate(
    duration=15.0,
    samples_per_second=100
)
calibrator.save_calibration(calibration_params, "drift_fix_calibration.json")
```

#### Button Not Working

```bash
# Check button mapping
python -c "
from gym_hil.spacemouse.spacemouse import SpaceMouseWrapper
sm = SpaceMouseWrapper()
print('Button states:', sm.get_button_states())
"

# Test button press detection
python -c "
import time
from gym_hil.spacemouse.spacemouse import SpaceMouseWrapper
sm = SpaceMouseWrapper()
print('Press buttons on SpaceMouse (Ctrl+C to exit)...')
try:
    while True:
        buttons = sm.get_button_states()
        if any(buttons.values()):
            print('Buttons pressed:', [k for k, v in buttons.items() if v])
        time.sleep(0.1)
except KeyboardInterrupt:
    pass
"
```

### Performance Optimization

#### Reduce Latency

```python
# High-performance configuration
high_perf_config = SpaceMouseConfig(
    # Increase sampling rate
    sampling_rate=200,              # Hz

    # Optimize sensitivity
    translation_sensitivity=2.0,
    rotation_sensitivity=1.5,

    # Minimize filtering
    filter_enabled=False,
    filter_strength=0.0,

    # Optimize deadzone
    translation_deadzone=0.01,
    rotation_deadzone=0.01,

    # Increase limits
    velocity_limit=0.3,
    acceleration_limit=1.0
)
```

#### Memory Optimization

```python
# Lightweight implementation for embedded systems
class LightweightSpaceMouse:
    def __init__(self):
        self.state_buffer = []
        self.buffer_size = 5  # Minimal buffer

    def get_action(self):
        """Optimized action computation"""
        current_state = self._read_device()

        # Simple moving average
        self.state_buffer.append(current_state)
        if len(self.state_buffer) > self.buffer_size:
            self.state_buffer.pop(0)

        return self._compute_average_action()
```

## Integration Examples

### Pick and Place Task

```bash
# Specialized configuration for pick and place
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=${HF_USER}/pick_place_spacemouse \
    --dataset.num_episodes=50 \
    --dataset.single_task="Pick up the block and place it in the target zone" \
    --teleop.type=spacemouse \
    --teleop.control_mode=7dof \
    --teleop.translation_sensitivity=0.8 \
    --teleop.rotation_sensitivity=0.6 \
    --teleop.gripper_button=1 \
    --teleop.approach_mode_button=2
```

### Assembly Task

```bash
# Fine manipulation for assembly tasks
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 60}}" \
    --dataset.repo_id=${HF_USER}/assembly_spacemouse \
    --dataset.num_episodes=30 \
    --dataset.single_task="Assemble the components with precise fit" \
    --teleop.type=spacemouse \
    --teleop.control_mode=7dof \
    --teleop.translation_sensitivity=0.5 \
    --teleop.rotation_sensitivity=0.3 \
    --teleop.translation_deadzone=0.01 \
    --teleop.rotation_deadzone=0.01 \
    --teleop.filter_enabled=true \
    --teleop.filter_strength=0.95
```

This comprehensive guide provides everything needed to effectively use SpaceMouse with LeRobot, from basic setup to advanced configuration and troubleshooting.

---
noteId: "d8c21470b09c11f08bd0898d85b76daa"
tags: []
---

# LeRobot Data Collection Guide

This comprehensive guide covers how to collect demonstration data using LeRobot, including general data collection workflows and specific SpaceMouse teleoperation setups.

## Table of Contents

1. [Overview](#overview)
2. [General Data Collection](#general-data-collection)
3. [SpaceMouse Data Collection](#spacemouse-data-collection)
4. [Configuration Files](#configuration-files)
5. [Data Collection Workflow](#data-collection-workflow)
6. [Best Practices](#best-practices)
7. [Troubleshooting](#troubleshooting)

## Overview

LeRobot provides a unified framework for collecting robot demonstration data through various teleoperation methods. The data collection process involves:

- **Teleoperation**: Human operators control robots using various input devices
- **Data Recording**: Simultaneous capture of robot states, actions, and observations
- **Dataset Management**: Automatic organization and storage in LeRobotDataset v3.0 format
- **Hub Integration**: Direct upload to Hugging Face Hub for sharing and collaboration

### Key Components

- **Robots**: Physical robots (SO-100, SO-101, ALOHA, etc.) or simulation environments
- **Teleoperators**: Input devices (SpaceMouse, gamepad, keyboard, phone, etc.)
- **Cameras**: Visual observation capture (OpenCV, Intel RealSense, etc.)
- **Processors**: Data transformation and preprocessing pipelines

## General Data Collection

### Basic Recording Command

The primary tool for data collection is the `lerobot-record` command:

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/tty.usbmodem585A0076841 \
  --robot.id=my_awesome_follower_arm \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 1920, height: 1080, fps: 30}}" \
  --teleop.type=so101_leader \
  --teleop.port=/dev/tty.usbmodem58760431551 \
  --teleop.id=my_awesome_leader_arm \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/record-test \
  --dataset.num_episodes=5 \
  --dataset.single_task="Grab the black cube"
```

### Key Parameters

#### Robot Configuration

- `--robot.type`: Robot type (so100_follower, so101_follower, aloha, etc.)
- `--robot.port`: Serial port for robot communication
- `--robot.id`: Unique identifier for the robot
- `--robot.cameras`: Camera configuration with type, resolution, and FPS

#### Teleoperator Configuration

- `--teleop.type`: Teleoperator type (spacemouse, gamepad, keyboard, phone, etc.)
- `--teleop.port`: Device port or connection details
- `--teleop.id`: Unique identifier for the teleoperator

#### Dataset Configuration

- `--dataset.repo_id`: Hugging Face repository ID for the dataset
- `--dataset.num_episodes`: Number of episodes to record
- `--dataset.single_task`: Task description for the demonstrations
- `--dataset.episode_time_s`: Duration of each episode (default: 60s)
- `--dataset.reset_time_s`: Time for environment reset between episodes (default: 60s)
- `--dataset.fps`: Recording frame rate (default: 30 FPS)

### Recording Process

The recording process follows this workflow:

1. **Initialization**: Connect to robot and teleoperator devices
2. **Episode Loop**: For each episode:
   - Reset environment to initial state
   - Record demonstration for specified duration
   - Save episode data
   - Reset environment for next episode
3. **Finalization**: Process and upload dataset to Hugging Face Hub

### Data Storage

Data is stored locally in `~/.cache/huggingface/lerobot/{repo-id}` and automatically uploaded to the Hugging Face Hub upon completion.

## SpaceMouse Data Collection

SpaceMouse is a 3D input device that provides intuitive 6-DOF control for robot teleoperation. It's particularly effective for precise manipulation tasks.

### SpaceMouse Setup

#### Hardware Requirements

- 3Dconnexion SpaceMouse device
- USB connection to computer
- Compatible drivers installed

#### Software Configuration

Create a configuration file for SpaceMouse teleoperation:

```json
{
  "type": "hil",
  "wrapper": {
    "gripper_penalty": -0.02,
    "display_cameras": false,
    "add_joint_velocity_to_observation": true,
    "add_ee_pose_to_observation": true,
    "crop_params_dict": {
      "observation.images.front": [0, 0, 128, 128],
      "observation.images.wrist": [0, 0, 128, 128]
    },
    "resize_size": [128, 128],
    "control_time_s": 150.0,
    "use_gripper": true,
    "fixed_reset_joint_positions": [0.0, 0.195, 0.0, -2.43, 0.0, 2.62, 0.785],
    "reset_time_s": 150.0,
    "control_mode": "spacemouse"
  },
  "name": "franka_sim",
  "mode": "record",
  "repo_id": "test_spacemouse_grasp",
  "task": "PandaPickCubeSpacemouse-v0",
  "num_episodes": 30,
  "fps": 10
}
```

### SpaceMouse Controls

#### Movement Controls

- **Translation**: Move SpaceMouse in X, Y, Z directions for end-effector position control
- **Rotation**: Rotate SpaceMouse for end-effector orientation control
- **Sensitivity**: Adjustable scaling factors for fine vs. coarse control

#### Button Controls

- **Left Button**: Close gripper
- **Right Button**: Open gripper
- **Additional Buttons**: Customizable for task-specific actions

#### Keyboard Overrides

- **SPACE**: Mark episode as SUCCESS
- **C**: Mark episode as FAILURE
- **R**: Re-record current episode

### SpaceMouse Recording Command

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=spacemouse \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/spacemouse_demo \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick and place objects" \
  --dataset.episode_time_s=120
```

### SpaceMouse Calibration

Before recording, calibrate the SpaceMouse:

1. **Dead Zone**: Set minimum movement threshold to avoid noise
2. **Scaling**: Adjust position and rotation scaling factors
3. **Gripper Mapping**: Configure button-to-gripper action mapping
4. **Test Movement**: Verify all axes respond correctly

## Configuration Files

### Environment Configuration

LeRobot uses JSON configuration files to define recording setups. Key sections include:

#### Robot Configuration

```json
{
  "robot": {
    "type": "so101_follower",
    "port": "/dev/ttyACM0",
    "id": "robot_001",
    "cameras": {
      "front": {
        "type": "opencv",
        "index_or_path": 0,
        "width": 640,
        "height": 480,
        "fps": 30
      }
    }
  }
}
```

#### Teleoperator Configuration

```json
{
  "teleop": {
    "type": "spacemouse",
    "port": "/dev/input/event0",
    "id": "spacemouse_001",
    "scaling": {
      "position": 0.1,
      "rotation": 0.05
    }
  }
}
```

#### Dataset Configuration

```json
{
  "dataset": {
    "repo_id": "user/dataset_name",
    "num_episodes": 50,
    "episode_time_s": 60,
    "reset_time_s": 10,
    "fps": 30,
    "single_task": "Task description",
    "push_to_hub": true
  }
}
```

## Data Collection Workflow

### Pre-Recording Setup

1. **Environment Preparation**
   - Set up robot workspace
   - Position objects and tools
   - Ensure adequate lighting
   - Test camera views

2. **Device Connection**
   - Connect robot via serial/USB
   - Connect teleoperator device
   - Test all device communications
   - Verify camera feeds

3. **Configuration**
   - Create or modify configuration file
   - Set Hugging Face credentials
   - Configure dataset parameters
   - Test recording setup

### Recording Session

1. **Start Recording**

   ```bash
   lerobot-record --config=my_config.json
   ```

2. **Episode Management**
   - Follow prompts for episode start/stop
   - Use keyboard shortcuts for episode control
   - Monitor recording quality in real-time
   - Handle any errors or interruptions

3. **Data Validation**
   - Check episode completeness
   - Verify data quality
   - Review camera feeds
   - Validate action sequences

### Post-Recording

1. **Dataset Processing**
   - Automatic conversion to LeRobotDataset v3.0 format
   - Metadata generation
   - Video encoding and compression
   - Quality checks

2. **Hub Upload**
   - Automatic upload to Hugging Face Hub
   - Repository creation and tagging
   - Access control configuration
   - Documentation generation

## Best Practices

### Data Quality

1. **Consistent Demonstrations**
   - Maintain consistent task execution
   - Use similar starting positions
   - Follow consistent action patterns
   - Avoid unnecessary movements

2. **Camera Setup**
   - Ensure good lighting conditions
   - Position cameras for optimal views
   - Avoid occlusions
   - Maintain consistent camera positions

3. **Task Design**
   - Define clear task objectives
   - Use appropriate objects and tools
   - Design for repeatability
   - Consider safety constraints

### Recording Efficiency

1. **Batch Recording**
   - Record multiple episodes in single session
   - Use consistent environment setup
   - Minimize reset time between episodes
   - Plan for efficient data collection

2. **Error Handling**
   - Use re-record functionality for failed episodes
   - Implement automatic error recovery
   - Monitor recording quality continuously
   - Have backup plans for device failures

### Dataset Organization

1. **Naming Conventions**
   - Use descriptive dataset names
   - Include task and robot information
   - Add version numbers for iterations
   - Follow Hugging Face naming guidelines

2. **Metadata**
   - Provide detailed task descriptions
   - Include robot and environment information
   - Document recording parameters
   - Add usage instructions

## Troubleshooting

### Common Issues

#### Device Connection Problems

```bash
# Check device connections
ls /dev/tty*
ls /dev/input/

# Test SpaceMouse connection
python -c "import hid; print(hid.enumerate())"
```

#### Camera Issues

```bash
# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

#### Recording Failures

- Check disk space availability
- Verify Hugging Face credentials
- Ensure stable network connection
- Monitor system resources

### Debug Mode

Enable debug logging for troubleshooting:

```bash
lerobot-record --config=my_config.json --log_level=DEBUG
```

### Recovery Procedures

1. **Resume Recording**

   ```bash
   lerobot-record --config=my_config.json --resume=true
   ```

2. **Manual Dataset Upload**

   ```bash
   huggingface-cli upload user/dataset_name ~/.cache/huggingface/lerobot/dataset_name --repo-type dataset
   ```

3. **Dataset Validation**
   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   dataset = LeRobotDataset("user/dataset_name")
   print(f"Episodes: {dataset.num_episodes}")
   print(f"Frames: {dataset.num_frames}")
   ```

## Advanced Features

### Multi-Camera Setup

Configure multiple cameras for comprehensive observation:

```json
{
  "robot": {
    "cameras": {
      "front": {
        "type": "opencv",
        "index_or_path": 0,
        "width": 640,
        "height": 480,
        "fps": 30
      },
      "wrist": {
        "type": "opencv",
        "index_or_path": 1,
        "width": 320,
        "height": 240,
        "fps": 30
      }
    }
  }
}
```

### Custom Processors

Implement custom data processing pipelines:

```python
from lerobot.processor import ProcessorStep

class CustomProcessor(ProcessorStep):
    def process(self, data):
        # Custom processing logic
        return processed_data
```

### Real-time Monitoring

Use the `--display_data=true` flag to monitor recording in real-time with Rerun visualization.

## Conclusion

This guide provides comprehensive coverage of LeRobot data collection capabilities. For specific robot or teleoperator setups, refer to the individual documentation pages in the LeRobot documentation.

Key takeaways:

- Use `lerobot-record` for all data collection
- Configure devices and parameters via JSON files
- SpaceMouse provides intuitive 6-DOF control
- Data is automatically processed and uploaded to Hugging Face Hub
- Follow best practices for consistent, high-quality datasets

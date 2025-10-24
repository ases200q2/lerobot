---
noteId: "637f5e80b09611f08bd0898d85b76daa"
tags: []
---

# LeRobot Data Collection Guide

## Overview

LeRobot provides a comprehensive, modular framework for collecting demonstration data through teleoperation and policy-based recording. The system supports multiple robot types, teleoperation devices, and seamless integration with training pipelines and HuggingFace Hub.

## Data Collection Architecture

### Core Pipeline

```
Robot → get_observation() → robot_observation_processor → Action Logic (Teleop/Policy) → robot_action_processor → send_action() → Dataset Storage
```

### Key Components

- **`lerobot-record`**: Main entry point for data collection
- **`LeRobotDataset`**: Central data structure managing dataset creation and storage
- **Processor Pipeline**: Modular transformation system for observations and actions
- **Robot Interface**: Abstract base class with specific implementations for different robots
- **Teleoperator Interface**: Various input device support for manual control

### Modular Design

- **Configuration System**: Uses `draccus` for type-safe YAML configurations
- **Factory Pattern**: `make_robot_from_config()` and `make_teleoperator_from_config()` for component instantiation
- **Pipeline Processing**: Extensible processor system for data transformations

## Prerequisites

### Installation

```bash
# Install from source (development mode)
pip install -e .

# Install with specific features for data collection
pip install -e ".[aloha,pusht]"  # Simulation environments
pip install -e ".[all]"          # All features
pip install -e ".[dev,test]"     # Development dependencies
```

### Hardware Setup

```bash
# Find available cameras and ports
lerobot-find-cameras
lerobot-find-port

# Setup motors (for SO100/SO101 robots)
lerobot-setup-motors --robot-config-path=configs/robot/so100.yaml

# Find joint limits
lerobot-find-joint-limits --robot-config-path=configs/robot/so100.yaml

# Calibrate robot
lerobot-calibrate
```

### Authentication

```bash
# Login to HuggingFace Hub for automatic dataset uploads
huggingface-cli login
export HF_USER=$(huggingface-cli whoami)
```

## Robot Configuration

### Supported Robot Types

1. **SO100/SO101**: Affordable 7-DOF robot arms
2. **Bimanual SO100**: Dual-arm configuration for complex tasks
3. **ALOHA**: Bimanual teleoperation system
4. **Reachy2**: Advanced humanoid robot
5. **Koch**: Industrial robot arms
6. **LeKiwi**: Custom mobile robot platform
7. **HOPE Jr**: Educational robot

### Robot Configuration Example

```yaml
# configs/robot/so100.yaml
robot:
  type: so100_follower
  id: black_arm
  port: /dev/tty.usbmodem58760431541
  cameras:
    front:
      type: opencv
      index_or_path: 0
      width: 640
      height: 480
      fps: 30
    wrist:
      type: opencv
      index_or_path: 1
      width: 320
      height: 240
      fps: 30
```

### Bimanual Configuration

```yaml
# configs/robot/bimanual_so100.yaml
robot:
  type: bi_so100_follower
  left_arm:
    port: /dev/tty.usbmodem5A460851411
    id: left_arm
  right_arm:
    port: /dev/tty.usbmodem5A460812391
    id: right_arm
  cameras:
    laptop:
      type: opencv
      index_or_path: 0
      width: 640
      height: 480
      fps: 30
```

## Teleoperation Devices

### Supported Teleoperators

1. **SO100/SO101 Leader**: Physical robot arm controllers
2. **SpaceMouse**: 6-DOF input device for precise control
3. **Gamepad**: Xbox, PlayStation controllers
4. **Keyboard**: WASD + mouse control
5. **Phone**: Mobile device teleoperation via HTTP interface
6. **Homunculus**: Custom control interfaces

### Teleoperator Configuration Examples

#### SpaceMouse Configuration

```bash
# SpaceMouse teleoperation
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --dataset.repo_id=${HF_USER}/spacemouse_dataset \
    --dataset.num_episodes=10 \
    --teleop.type=spacemouse
```

#### Gamepad Configuration

```bash
# Gamepad teleoperation
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --robot.id=black \
    --dataset.repo_id=${HF_USER}/gamepad_dataset \
    --dataset.num_episodes=10 \
    --teleop.type=gamepad
```

#### Bimanual Configuration

```bash
# Bimanual recording with dual teleoperators
lerobot-record \
    --robot.type=bi_so100_follower \
    --robot.left_arm_port=/dev/tty.usbmodem5A460851411 \
    --robot.right_arm_port=/dev/tty.usbmodem5A460812391 \
    --robot.id=bimanual_follower \
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=${HF_USER}/bimanual_dataset \
    --dataset.num_episodes=25 \
    --dataset.single_task="Grab and handover the red cube" \
    --teleop.type=bi_so100_leader \
    --teleop.left_arm_port=/dev/tty.usbmodem58760431551 \
    --teleop.right_arm_port=/dev/tty.usbmodem58760431561 \
    --teleop.id=bimanual_leader
```

## Data Collection Commands

### Basic Recording

```bash
# Simple recording with automatic upload
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=spacemouse \
    --dataset.repo_id=${HF_USER}/my_first_dataset \
    --dataset.num_episodes=10 \
    --dataset.push_to_hub=true
```

### Advanced Recording with Custom Parameters

```bash
# Recording with custom configuration
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 60}, wrist: {type: opencv, index_or_path: 1, width: 640, height: 480, fps: 30}}" \
    --robot.id=black_arm \
    --teleop.type=spacemouse \
    --teleop.control_mode=7dof \
    --dataset.repo_id=${HF_USER}/high_quality_dataset \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.fps=60 \
    --dataset.video=true \
    --dataset.single_task="Stack the blocks in ascending order" \
    --dataset.push_to_hub=true \
    --dataset.private=false \
    --dataset.num_image_writer_processes=2 \
    --dataset.num_image_writer_threads_per_camera=4
```

### Policy-Based Recording

```bash
# Record using a trained policy for autonomous demonstrations
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --dataset.repo_id=${HF_USER}/policy demonstrations \
    --dataset.num_episodes=20 \
    --policy.path=${HF_USER}/my_trained_policy \
    --policy.type=act \
    --policy.device=cuda
```

### Hybrid Recording (Teleop + Policy)

```bash
# Hybrid control with switching between teleoperation and policy
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=spacemouse \
    --policy.path=${HF_USER}/my_trained_policy \
    --policy.type=act \
    --dataset.repo_id=${HF_USER}/hybrid_dataset \
    --dataset.num_episodes=30 \
    --control_mode=hybrid
```

## Dataset Configuration

### Dataset Parameters

```python
@dataclass
class DatasetRecordConfig:
    repo_id: str                    # HuggingFace dataset ID
    single_task: str               # Task description
    fps: int = 30                  # Collection frame rate
    episode_time_s: int = 60       # Duration per episode
    reset_time_s: int = 60         # Reset time between episodes
    num_episodes: int = 50         # Number of episodes to collect
    video: bool = True             # Enable video encoding
    push_to_hub: bool = True       # Auto-upload to HuggingFace
    num_image_writer_processes: int = 0    # Parallel image writing
    num_image_writer_threads_per_camera: int = 4  # Threads per camera
    private: bool = False          # Private repository
    episodes: List[int] = None     # Specific episodes to record
```

### Dataset Structure (v3.0 Format)

```
dataset_repo/
├── data/                   # Parquet files with episode data
│   ├── episode_000000.parquet
│   ├── episode_000001.parquet
│   └── ...
├── videos/                # MP4 video files (chunked)
│   ├── observation.images.laptop/
│   │   ├── chunk-000/file-000.mp4
│   │   └── ...
│   └── observation.images.wrist/
│       ├── chunk-000/file-000.mp4
│       └── ...
├── meta/                   # Metadata files
│   ├── info.json          # Dataset info and metadata
│   ├── stats.json         # Statistical information
│   └── episodes/          # Episode metadata in chunked parquet files
├── images/                # Individual PNG frames (optional)
└── README.md              # Dataset card
```

### Data Format Specification

```json
{
  "features": {
    "observation/state": {
      "dtype": "float32",
      "shape": [7],
      "names": ["joint_positions"]
    },
    "action": {
      "dtype": "float32",
      "shape": [7],
      "names": ["joint_targets"]
    },
    "observation/images": {
      "dtype": "image",
      "shape": [480, 640, 3]
    },
    "timestamp": {
      "dtype": "float32",
      "shape": [1]
    },
    "frame_index": {
      "dtype": "int64",
      "shape": [1]
    },
    "episode_index": {
      "dtype": "int64",
      "shape": [1]
    }
  }
}
```

## Data Collection Workflow

### Step-by-Step Process

1. **Configuration Setup**

   ```bash
   # Create robot configuration file
   mkdir configs/robot
   cp templates/so100.yaml configs/robot/my_robot.yaml
   # Edit configuration with your hardware parameters
   ```

2. **Hardware Connection**

   ```bash
   # Verify robot connection
   lerobot-setup-motors --robot-config-path=configs/robot/my_robot.yaml

   # Test teleoperation
   lerobot-teleoperate --robot-config-path=configs/robot/my_robot.yaml --teleop.type=spacemouse
   ```

3. **Dataset Recording**

   ```bash
   # Start recording
   lerobot-record \
       --robot.config_path=configs/robot/my_robot.yaml \
       --teleop.type=spacemouse \
       --dataset.repo_id=${HF_USER}/my_task_dataset \
       --dataset.num_episodes=20 \
       --dataset.single_task="Pick up the red block"
   ```

4. **Monitoring and Quality Control**

   ```bash
   # Visualize collected data
   lerobot-dataset-viz --repo-id ${HF_USER}/my_task_dataset --episode-index 0

   # Replay recorded data
   lerobot-replay --root ./data
   ```

5. **Upload and Sharing**
   ```bash
   # Automatic upload during recording (if push_to_hub=true)
   # Or manual upload
   python -c "
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   dataset = LeRobotDataset('path/to/local/dataset')
   dataset.push_to_hub('${HF_USER}/my_task_dataset', private=False)
   "
   ```

### Real-time Monitoring

```bash
# Monitor recording progress
lerobot-dataset-viz \
    --repo-id ${HF_USER}/my_task_dataset \
    --episode-index -1  # Latest episode
    --fps 30
```

## Best Practices

### Data Quality

1. **Consistent Task Definition**: Use clear, specific task descriptions
2. **Varied Initial Conditions**: Include diverse starting positions and orientations
3. **Sufficient Episodes**: Collect at least 20-50 episodes for meaningful training
4. **Quality Over Quantity**: Focus on successful demonstrations
5. **Multiple Perspectives**: Use multiple camera angles when possible

### Recording Strategy

1. **Progressive Difficulty**: Start with simple tasks, then increase complexity
2. **Error Recovery**: Include examples of recovering from failures
3. **Complete Demonstrations**: Ensure episodes start and end in defined states
4. **Consistent Frame Rate**: Maintain stable FPS throughout recording
5. **Proper Lighting**: Ensure good, consistent lighting conditions

### Hardware Setup

1. **Stable Mounting**: Securely mount robots and cameras
2. **Cable Management**: Prevent cable interference during operation
3. **Calibration**: Regularly calibrate cameras and robot kinematics
4. **Backup Power**: Use stable power sources to prevent interruptions
5. **Environmental Control**: Maintain consistent temperature and humidity

## Troubleshooting

### Common Issues

#### Connection Problems

```bash
# Check device connections
lerobot-find-port
lerobot-find-cameras

# Test robot connection
python -c "
from lerobot.robots.factory import make_robot_from_config
robot = make_robot_from_config('configs/robot/so100.yaml')
robot.connect()
robot.disconnect()
"
```

#### Teleoperation Issues

```bash
# Test teleoperation separately
lerobot-teleoperate \
    --robot-config-path=configs/robot/so100.yaml \
    --teleop.type=spacemouse

# Check device permissions
sudo usermod -a -G dialout $USER
# Re-login required
```

#### Data Recording Issues

```bash
# Check disk space
df -h

# Verify dataset creation
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
try:
    dataset = LeRobotDataset('test_dataset')
    print('Dataset creation successful')
except Exception as e:
    print(f'Error: {e}')
"
```

#### Upload Problems

```bash
# Check HuggingFace authentication
huggingface-cli whoami

# Test upload
python -c "
from huggingface_hub import HfApi
api = HfApi()
user_info = api.whoami()
print(f'Logged in as: {user_info[\"name\"]}')
"
```

### Performance Optimization

1. **Increase Writer Processes**: For high-resolution video recording
2. **Adjust Chunk Size**: For large datasets (>10GB)
3. **Use SSD Storage**: For faster write speeds
4. **Optimize Camera Settings**: Balance quality vs performance
5. **Monitor Memory Usage**: Prevent system overload

## Advanced Features

### Custom Processors

```python
# Custom data processing pipeline
from lerobot.processor.processor import ProcessorStep

class CustomProcessorStep(ProcessorStep):
    def process(self, data):
        # Custom data transformation
        return processed_data

# Add to configuration
processor_config = [
    RenameObservationsProcessorStep(rename_map={}),
    AddBatchDimensionProcessorStep(),
    CustomProcessorStep(),
    DeviceProcessorStep(device="cuda"),
]
```

### Multi-Robot Recording

```bash
# Record with multiple robots simultaneously
lerobot-record \
    --robots.config_paths="{robot1: configs/robot/so100_1.yaml, robot2: configs/robot/so100_2.yaml}" \
    --teleop.type=multi_spacemouse \
    --dataset.repo_id=${HF_USER}/multi_robot_dataset \
    --dataset.num_episodes=30
```

### Real-time Augmentation

```bash
# Record with real-time data augmentation
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/ttyACM0 \
    --dataset.repo_id=${HF_USER}/augmented_dataset \
    --dataset.augmentation.enabled=true \
    --dataset.augmentation.brightness_range=0.2 \
    --dataset.augmentation.rotation_range=15
```

## Integration with Training

### Direct Training Integration

```bash
# Train directly on collected dataset
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=${HF_USER}/my_task_dataset \
    --output_dir=./outputs/my_training \
    --batch_size=8 \
    --steps=100000
```

### Dataset Evaluation

```bash
# Evaluate dataset quality
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset('${HF_USER}/my_task_dataset')
print(f'Dataset info: {dataset.info}')
print(f'Statistics: {dataset.stats}')
"
```

This comprehensive guide provides all the necessary information for collecting high-quality demonstration data using LeRobot, from basic setup to advanced features and integration with training pipelines.

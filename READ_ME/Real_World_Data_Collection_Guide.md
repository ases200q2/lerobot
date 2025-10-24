---
noteId: "d8c26291b09c11f08bd0898d85b76daa"
tags: []
---

# LeRobot Real-World Data Collection Guide

This comprehensive guide covers how to collect demonstration data using real robots with LeRobot, including hardware setup, safety considerations, and best practices for real-world robot learning.

## Table of Contents

1. [Overview](#overview)
2. [Supported Robots](#supported-robots)
3. [Hardware Setup](#hardware-setup)
4. [Safety Considerations](#safety-considerations)
5. [Data Collection Workflow](#data-collection-workflow)
6. [Robot-Specific Guides](#robot-specific-guides)
7. [Quality Assurance](#quality-assurance)
8. [Troubleshooting](#troubleshooting)

## Overview

Real-world data collection with LeRobot involves:

- **Physical Robot Setup**: Hardware configuration and calibration
- **Teleoperation**: Human control using various input devices
- **Data Recording**: Simultaneous capture of robot states and observations
- **Safety Management**: Ensuring safe operation throughout data collection
- **Quality Control**: Maintaining consistent, high-quality demonstrations

### Key Components

- **Robot Hardware**: Physical robot arms, grippers, and sensors
- **Control Systems**: Robot controllers and communication interfaces
- **Teleoperation Devices**: SpaceMouse, gamepad, keyboard, or custom devices
- **Cameras**: Visual observation capture systems
- **Safety Systems**: Emergency stops, collision detection, and monitoring

## Supported Robots

LeRobot supports various real-world robot platforms:

### ALOHA (A Low-cost Open-source Hardware for Autonomous manipulation)

- **Description**: Bimanual robot system for manipulation tasks
- **Configuration**: Two 6-DOF arms with grippers
- **Teleoperation**: SpaceMouse, gamepad, or custom devices
- **Applications**: Bimanual manipulation, assembly tasks

### SO-100/SO-101 (Stretch Open)

- **Description**: Mobile manipulator with telescoping arm
- **Configuration**: 7-DOF arm with mobile base
- **Teleoperation**: SpaceMouse, gamepad, or custom devices
- **Applications**: Mobile manipulation, household tasks

### Koch

- **Description**: Compact desktop manipulator
- **Configuration**: 6-DOF arm with gripper
- **Teleoperation**: Keyboard, gamepad, or custom devices
- **Applications**: Desktop manipulation, pick and place

### Custom Robots

- **Integration**: Support for custom robot configurations
- **Requirements**: ROS2 compatibility, joint control interface
- **Development**: Custom robot drivers and configurations

## Hardware Setup

### Robot Hardware Requirements

#### ALOHA Setup

```bash
# Hardware components
- 2x 6-DOF robot arms (ALOHA arms)
- 2x grippers (ALOHA grippers)
- 2x cameras (Intel RealSense or similar)
- Control computer (Ubuntu 20.04+)
- Power supplies and cables
- Safety equipment (emergency stops, barriers)
```

#### SO-100/SO-101 Setup

```bash
# Hardware components
- 1x SO-100/SO-101 robot
- 1x camera (Intel RealSense or similar)
- Control computer (Ubuntu 20.04+)
- Power supply and cables
- Safety equipment
```

#### Koch Setup

```bash
# Hardware components
- 1x Koch robot arm
- 1x gripper
- 1x camera (USB webcam or similar)
- Control computer (Ubuntu 20.04+)
- Power supply and cables
- Safety equipment
```

### Computer Requirements

#### Minimum Specifications

```bash
# Hardware
- CPU: Intel i5 or AMD Ryzen 5 (8 cores recommended)
- RAM: 16GB (32GB recommended)
- GPU: NVIDIA GTX 1060 or better (for training)
- Storage: 500GB SSD (1TB recommended)
- Network: Ethernet connection (WiFi for teleoperation)

# Software
- Ubuntu 20.04 LTS or newer
- ROS2 Humble
- Python 3.8+
- CUDA 11.8+ (for GPU training)
```

#### Recommended Specifications

```bash
# Hardware
- CPU: Intel i7 or AMD Ryzen 7 (16 cores)
- RAM: 32GB
- GPU: NVIDIA RTX 3080 or better
- Storage: 1TB NVMe SSD
- Network: Gigabit Ethernet

# Software
- Ubuntu 22.04 LTS
- ROS2 Humble
- Python 3.10+
- CUDA 12.0+
```

### Network Configuration

```bash
# Robot network setup
# Robot IP: 192.168.1.100 (example)
# Computer IP: 192.168.1.101 (example)

# Configure network
sudo ip addr add 192.168.1.101/24 dev eth0
sudo ip route add 192.168.1.0/24 dev eth0

# Test connectivity
ping 192.168.1.100
```

## Safety Considerations

### Safety Equipment

#### Essential Safety Equipment

```bash
# Physical safety
- Emergency stop buttons (multiple locations)
- Safety barriers or cages
- Warning signs and lights
- First aid kit
- Fire extinguisher

# Software safety
- Collision detection systems
- Joint limit monitoring
- Velocity and acceleration limits
- Emergency stop software
- Safety monitoring scripts
```

#### Safety Procedures

1. **Pre-Operation Checklist**

   ```bash
   # Safety checklist
   □ Emergency stops tested and functional
   □ Safety barriers in place
   □ Robot workspace clear of obstacles
   □ All personnel aware of robot operation
   □ Safety equipment accessible
   □ Communication system tested
   ```

2. **During Operation**

   ```bash
   # Safety protocols
   - Always maintain line of sight with robot
   - Keep emergency stop within reach
   - Monitor robot behavior continuously
   - Stop immediately if any issues occur
   - Communicate with team members
   - Follow established procedures
   ```

3. **Emergency Procedures**
   ```bash
   # Emergency response
   1. Press emergency stop immediately
   2. Assess situation and ensure safety
   3. Notify team members
   4. Document incident
   5. Investigate and resolve issues
   6. Resume only after safety verification
   ```

### Safety Monitoring Script

```python
#!/usr/bin/env python3
"""
Safety monitoring script for robot operation
"""

import time
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

class SafetyMonitor:
    """Monitor robot safety during operation."""

    def __init__(self):
        self.joint_limits = {
            'joint_1': [-3.14, 3.14],
            'joint_2': [-1.57, 1.57],
            'joint_3': [-3.14, 3.14],
            'joint_4': [-1.57, 1.57],
            'joint_5': [-3.14, 3.14],
            'joint_6': [-1.57, 1.57],
            'joint_7': [-3.14, 3.14]
        }

        self.velocity_limits = {
            'joint_1': 2.0,
            'joint_2': 2.0,
            'joint_3': 2.0,
            'joint_4': 2.0,
            'joint_5': 2.0,
            'joint_6': 2.0,
            'joint_7': 2.0
        }

        self.safety_pub = rospy.Publisher('/safety_status', Bool, queue_size=1)
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_callback)

        self.safety_status = True
        self.last_joint_states = None

    def joint_callback(self, msg):
        """Monitor joint states for safety violations."""
        current_time = time.time()

        # Check joint limits
        for i, joint_name in enumerate(msg.name):
            if joint_name in self.joint_limits:
                position = msg.position[i]
                min_limit, max_limit = self.joint_limits[joint_name]

                if position < min_limit or position > max_limit:
                    print(f"⚠️  Joint limit violation: {joint_name} = {position}")
                    self.safety_status = False

        # Check velocity limits
        if self.last_joint_states is not None:
            dt = current_time - self.last_time
            for i, joint_name in enumerate(msg.name):
                if joint_name in self.velocity_limits:
                    velocity = abs(msg.velocity[i])
                    max_velocity = self.velocity_limits[joint_name]

                    if velocity > max_velocity:
                        print(f"⚠️  Velocity limit violation: {joint_name} = {velocity}")
                        self.safety_status = False

        # Publish safety status
        self.safety_pub.publish(Bool(data=self.safety_status))

        # Update for next iteration
        self.last_joint_states = msg
        self.last_time = current_time

    def run(self):
        """Run safety monitoring."""
        rospy.init_node('safety_monitor')
        print("🛡️  Safety monitor started")

        try:
            rospy.spin()
        except KeyboardInterrupt:
            print("🛡️  Safety monitor stopped")

if __name__ == "__main__":
    monitor = SafetyMonitor()
    monitor.run()
```

## Data Collection Workflow

### Pre-Collection Setup

1. **Robot Calibration**

   ```bash
   # Calibrate robot
   ros2 run robot_calibration calibrate_robot

   # Test robot movement
   ros2 run robot_control test_movement

   # Verify camera feeds
   ros2 run camera_calibration calibrate_cameras
   ```

2. **Environment Setup**

   ```bash
   # Prepare workspace
   - Clear workspace of obstacles
   - Position objects for manipulation
   - Ensure adequate lighting
   - Test camera views
   - Verify safety systems
   ```

3. **Teleoperator Setup**

   ```bash
   # Test teleoperation device
   python -c "from lerobot.teleoperators.spacemouse import SpaceMouseTeleop; teleop = SpaceMouseTeleop(); print('SpaceMouse connected')"

   # Calibrate input device
   python -c "from lerobot.teleoperators.spacemouse import SpaceMouseTeleop; teleop = SpaceMouseTeleop(); teleop.calibrate()"
   ```

### Data Collection Process

#### Basic Recording Command

```bash
lerobot-record \
  --robot.type=aloha \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=aloha_robot \
  --robot.cameras="{ front: {type: intelrealsense, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=spacemouse \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/aloha_demo \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick and place objects" \
  --dataset.episode_time_s=60
```

#### Advanced Recording with Safety

```python
#!/usr/bin/env python3
"""
Advanced data collection with safety monitoring
"""

import rospy
from std_msgs.msg import Bool
from lerobot.scripts.lerobot_record import record
from lerobot.datasets.lerobot_dataset import LeRobotDataset

class SafeDataCollection:
    """Data collection with safety monitoring."""

    def __init__(self, config):
        self.config = config
        self.safety_status = True
        self.recording_active = False

        # Subscribe to safety status
        self.safety_sub = rospy.Subscriber('/safety_status', Bool, self.safety_callback)

        # Initialize dataset
        self.dataset = LeRobotDataset(
            repo_id=config["dataset"]["repo_id"],
            root=config["dataset"]["root"],
            video=config["dataset"]["video"]
        )

    def safety_callback(self, msg):
        """Handle safety status updates."""
        self.safety_status = msg.data

        if not self.safety_status and self.recording_active:
            print("⚠️  Safety violation detected! Stopping recording...")
            self.stop_recording()

    def start_recording(self):
        """Start data collection with safety monitoring."""
        if not self.safety_status:
            print("❌ Cannot start recording: Safety violation detected")
            return False

        print("🎬 Starting data collection...")
        self.recording_active = True

        # Start recording
        try:
            record(self.config)
            print("✅ Data collection completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Data collection failed: {e}")
            return False
        finally:
            self.recording_active = False

    def stop_recording(self):
        """Stop data collection."""
        print("⏹️  Stopping data collection...")
        self.recording_active = False

        # Finalize dataset
        self.dataset.finalize()
        self.dataset.push_to_hub()

    def run(self):
        """Run data collection session."""
        rospy.init_node('safe_data_collection')

        print("🤖 Safe Data Collection Started")
        print("Press Ctrl+C to stop")

        try:
            self.start_recording()
        except KeyboardInterrupt:
            print("⏹️  Data collection interrupted by user")
            self.stop_recording()
        except Exception as e:
            print(f"❌ Error during data collection: {e}")
            self.stop_recording()

if __name__ == "__main__":
    # Configuration
    config = {
        "robot": {
            "type": "aloha",
            "port": "/dev/ttyUSB0",
            "id": "aloha_robot"
        },
        "teleop": {
            "type": "spacemouse"
        },
        "dataset": {
            "repo_id": "user/aloha_demo",
            "root": "./data/aloha",
            "video": True,
            "num_episodes": 20,
            "episode_time_s": 60
        }
    }

    collector = SafeDataCollection(config)
    collector.run()
```

### Quality Control During Collection

```python
def monitor_data_quality(dataset, episode_idx):
    """Monitor data quality during collection."""

    # Get episode data
    from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

    # Quality metrics
    quality_metrics = {
        "episode_length": to_idx - from_idx,
        "action_variance": 0.0,
        "camera_quality": 0.0,
        "safety_violations": 0
    }

    # Analyze episode
    actions = []
    for frame_idx in range(from_idx, to_idx):
        frame = dataset[frame_idx]
        actions.append(frame["action"].numpy())

    # Calculate action variance
    actions = np.array(actions)
    quality_metrics["action_variance"] = np.var(actions)

    # Check for safety violations
    for frame_idx in range(from_idx, to_idx):
        frame = dataset[frame_idx]
        if "safety_violation" in frame.get("info", {}):
            quality_metrics["safety_violations"] += 1

    # Assess quality
    if quality_metrics["episode_length"] < 50:
        print("⚠️  Episode too short")
    elif quality_metrics["action_variance"] < 0.01:
        print("⚠️  Low action variance (robot may be stuck)")
    elif quality_metrics["safety_violations"] > 0:
        print("⚠️  Safety violations detected")
    else:
        print("✅ Episode quality acceptable")

    return quality_metrics
```

## Robot-Specific Guides

### ALOHA Robot Setup

#### Hardware Configuration

```bash
# ALOHA hardware setup
1. Connect robot arms to control computer
2. Connect grippers to robot arms
3. Connect cameras to computer
4. Connect power supplies
5. Test all connections
```

#### Software Configuration

```bash
# ALOHA software setup
git clone https://github.com/tonyzhaozh/ALOHA.git
cd ALOHA
pip install -e .

# Test ALOHA connection
python -c "from aloha.robot import ALOHARobot; robot = ALOHARobot(); print('ALOHA connected')"
```

#### Data Collection Command

```bash
lerobot-record \
  --robot.type=aloha \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=aloha_robot \
  --robot.cameras="{ front: {type: intelrealsense, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=spacemouse \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/aloha_demo \
  --dataset.num_episodes=20 \
  --dataset.single_task="Bimanual manipulation task"
```

### SO-100/SO-101 Robot Setup

#### Hardware Configuration

```bash
# SO-100/SO-101 hardware setup
1. Connect robot to control computer
2. Connect camera to computer
3. Connect power supply
4. Test robot movement
5. Calibrate cameras
```

#### Software Configuration

```bash
# SO-100/SO-101 software setup
git clone https://github.com/hello-robot/stretch_ros2.git
cd stretch_ros2
pip install -e .

# Test SO-100 connection
python -c "from stretch_ros2.robot import StretchRobot; robot = StretchRobot(); print('SO-100 connected')"
```

#### Data Collection Command

```bash
lerobot-record \
  --robot.type=so100_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so100_robot \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=spacemouse \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/so100_demo \
  --dataset.num_episodes=20 \
  --dataset.single_task="Mobile manipulation task"
```

### Koch Robot Setup

#### Hardware Configuration

```bash
# Koch hardware setup
1. Connect robot to control computer
2. Connect gripper to robot
3. Connect camera to computer
4. Connect power supply
5. Test robot movement
```

#### Software Configuration

```bash
# Koch software setup
git clone https://github.com/koch-robot/koch_ros2.git
cd koch_ros2
pip install -e .

# Test Koch connection
python -c "from koch_ros2.robot import KochRobot; robot = KochRobot(); print('Koch connected')"
```

#### Data Collection Command

```bash
lerobot-record \
  --robot.type=koch \
  --robot.port=/dev/ttyUSB0 \
  --robot.id=koch_robot \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=keyboard \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/koch_demo \
  --dataset.num_episodes=20 \
  --dataset.single_task="Desktop manipulation task"
```

## Quality Assurance

### Data Quality Metrics

#### Episode Quality Assessment

```python
def assess_episode_quality(dataset, episode_idx):
    """Assess quality of recorded episode."""

    # Get episode data
    from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]
    episode_length = to_idx - from_idx

    # Quality metrics
    metrics = {
        "episode_length": episode_length,
        "action_smoothness": 0.0,
        "camera_quality": 0.0,
        "task_completion": False,
        "safety_violations": 0
    }

    # Analyze actions
    actions = []
    for frame_idx in range(from_idx, to_idx):
        frame = dataset[frame_idx]
        actions.append(frame["action"].numpy())

    actions = np.array(actions)

    # Calculate smoothness (low acceleration)
    accelerations = np.diff(actions, n=2, axis=0)
    metrics["action_smoothness"] = np.mean(np.linalg.norm(accelerations, axis=1))

    # Check camera quality
    for frame_idx in range(from_idx, min(from_idx + 10, to_idx)):
        frame = dataset[frame_idx]
        if "observation" in frame and "images" in frame["observation"]:
            for cam_name, image in frame["observation"]["images"].items():
                # Check image quality (brightness, contrast, etc.)
                brightness = np.mean(image)
                contrast = np.std(image)

                if brightness < 50 or brightness > 200:
                    metrics["camera_quality"] += 1
                if contrast < 20:
                    metrics["camera_quality"] += 1

    # Check for safety violations
    for frame_idx in range(from_idx, to_idx):
        frame = dataset[frame_idx]
        if "info" in frame and "safety_violation" in frame["info"]:
            metrics["safety_violations"] += 1

    # Assess overall quality
    quality_score = 0

    if metrics["episode_length"] >= 50:
        quality_score += 1
    if metrics["action_smoothness"] < 0.1:
        quality_score += 1
    if metrics["camera_quality"] == 0:
        quality_score += 1
    if metrics["safety_violations"] == 0:
        quality_score += 1

    metrics["quality_score"] = quality_score / 4.0

    return metrics
```

#### Dataset Quality Assessment

```python
def assess_dataset_quality(dataset):
    """Assess overall dataset quality."""

    print(f"📊 Dataset Quality Assessment")
    print(f"Episodes: {dataset.num_episodes}")
    print(f"Frames: {dataset.num_frames}")

    # Analyze all episodes
    episode_metrics = []
    for episode_idx in range(dataset.num_episodes):
        metrics = assess_episode_quality(dataset, episode_idx)
        episode_metrics.append(metrics)

    # Calculate overall metrics
    avg_quality_score = np.mean([m["quality_score"] for m in episode_metrics])
    avg_episode_length = np.mean([m["episode_length"] for m in episode_metrics])
    total_safety_violations = sum([m["safety_violations"] for m in episode_metrics])

    print(f"\n📈 Overall Quality Metrics:")
    print(f"  Average Quality Score: {avg_quality_score:.3f}")
    print(f"  Average Episode Length: {avg_episode_length:.1f} frames")
    print(f"  Total Safety Violations: {total_safety_violations}")

    # Quality recommendations
    if avg_quality_score < 0.7:
        print("⚠️  Dataset quality is below recommended threshold")
        print("   Consider re-recording episodes with low quality scores")

    if total_safety_violations > 0:
        print("⚠️  Safety violations detected in dataset")
        print("   Review and remove episodes with safety violations")

    if avg_episode_length < 50:
        print("⚠️  Episodes are too short")
        print("   Consider increasing episode duration")

    return {
        "avg_quality_score": avg_quality_score,
        "avg_episode_length": avg_episode_length,
        "total_safety_violations": total_safety_violations,
        "episode_metrics": episode_metrics
    }
```

### Data Validation

```python
def validate_dataset(dataset):
    """Validate dataset for training readiness."""

    print(f"🔍 Dataset Validation")

    validation_results = {
        "valid": True,
        "issues": [],
        "recommendations": []
    }

    # Check dataset structure
    if dataset.num_episodes < 10:
        validation_results["issues"].append("Too few episodes (minimum 10 recommended)")
        validation_results["valid"] = False

    if dataset.num_frames < 1000:
        validation_results["issues"].append("Too few frames (minimum 1000 recommended)")
        validation_results["valid"] = False

    # Check episode consistency
    episode_lengths = []
    for episode_idx in range(dataset.num_episodes):
        from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
        to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]
        episode_lengths.append(to_idx - from_idx)

    if np.std(episode_lengths) > np.mean(episode_lengths) * 0.5:
        validation_results["issues"].append("High episode length variance")
        validation_results["recommendations"].append("Consider standardizing episode lengths")

    # Check action space coverage
    all_actions = []
    for episode_idx in range(min(5, dataset.num_episodes)):  # Sample first 5 episodes
        episode_data = get_episode_data(dataset, episode_idx)
        for frame in episode_data:
            all_actions.append(frame["action"].numpy())

    all_actions = np.array(all_actions)
    action_ranges = np.max(all_actions, axis=0) - np.min(all_actions, axis=0)

    if np.any(action_ranges < 0.1):
        validation_results["issues"].append("Limited action space coverage")
        validation_results["recommendations"].append("Ensure diverse action demonstrations")

    # Check camera quality
    camera_issues = 0
    for episode_idx in range(min(3, dataset.num_episodes)):  # Sample first 3 episodes
        episode_data = get_episode_data(dataset, episode_idx)
        for frame in episode_data[:10]:  # Check first 10 frames
            if "observation" in frame and "images" in frame["observation"]:
                for cam_name, image in frame["observation"]["images"].items():
                    if np.mean(image) < 30 or np.mean(image) > 220:
                        camera_issues += 1

    if camera_issues > 0:
        validation_results["issues"].append(f"Camera quality issues in {camera_issues} frames")
        validation_results["recommendations"].append("Check lighting and camera settings")

    # Print results
    if validation_results["valid"]:
        print("✅ Dataset validation passed")
    else:
        print("❌ Dataset validation failed")
        print("Issues:")
        for issue in validation_results["issues"]:
            print(f"  - {issue}")

    if validation_results["recommendations"]:
        print("Recommendations:")
        for rec in validation_results["recommendations"]:
            print(f"  - {rec}")

    return validation_results
```

## Troubleshooting

### Common Hardware Issues

#### Robot Connection Problems

```bash
# Check robot connection
ls /dev/tty*
ls /dev/usb*

# Test robot communication
python -c "from lerobot.robots.aloha import ALOHARobot; robot = ALOHARobot(); print('Robot connected')"

# Check ROS2 topics
ros2 topic list
ros2 topic echo /joint_states
```

#### Camera Issues

```bash
# Check camera connection
ls /dev/video*

# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera connected:', cap.isOpened())"

# Check camera feeds
ros2 run image_view image_view --ros-args --remap image:=/camera/image_raw
```

#### Teleoperator Issues

```bash
# Check SpaceMouse connection
python -c "import hid; print(hid.enumerate())"

# Test SpaceMouse
python -c "from lerobot.teleoperators.spacemouse import SpaceMouseTeleop; teleop = SpaceMouseTeleop(); print('SpaceMouse connected')"

# Check gamepad connection
python -c "from lerobot.teleoperators.gamepad import GamepadTeleop; teleop = GamepadTeleop(); print('Gamepad connected')"
```

### Software Issues

#### ROS2 Problems

```bash
# Check ROS2 installation
ros2 --version

# Check ROS2 environment
echo $ROS_DISTRO
echo $ROS_DOMAIN_ID

# Restart ROS2 daemon
ros2 daemon stop
ros2 daemon start
```

#### LeRobot Issues

```bash
# Check LeRobot installation
python -c "import lerobot; print('LeRobot installed')"

# Check robot drivers
python -c "from lerobot.robots import available_robots; print(available_robots)"

# Check teleoperators
python -c "from lerobot.teleoperators import available_teleoperators; print(available_teleoperators)"
```

### Performance Issues

#### Slow Data Collection

```python
# Optimize data collection performance
def optimize_data_collection():
    """Optimize data collection performance."""

    # Use faster video encoding
    config = {
        "dataset": {
            "video_backend": "pyav",  # Faster than opencv
            "video_quality": "medium",  # Balance quality and speed
            "fps": 30  # Reduce if needed
        }
    }

    # Use SSD storage
    config["dataset"]["root"] = "/fast_ssd/lerobot_data"

    # Reduce camera resolution if needed
    config["robot"]["cameras"] = {
        "front": {
            "type": "opencv",
            "index_or_path": 0,
            "width": 320,  # Reduced resolution
            "height": 240,
            "fps": 30
        }
    }

    return config
```

#### Memory Issues

```python
# Handle memory issues during data collection
def handle_memory_issues():
    """Handle memory issues during data collection."""

    # Clear unused variables
    import gc
    gc.collect()

    # Use smaller batch sizes
    batch_size = 16

    # Process data in chunks
    chunk_size = 1000

    # Monitor memory usage
    import psutil
    memory_usage = psutil.virtual_memory().percent
    if memory_usage > 80:
        print(f"⚠️  High memory usage: {memory_usage}%")
        gc.collect()
```

### Recovery Procedures

#### Resume Interrupted Collection

```bash
# Resume interrupted data collection
lerobot-record \
  --config=my_config.json \
  --resume=true \
  --dataset.num_episodes=10  # Additional episodes to record
```

#### Recover from Crashes

```python
def recover_from_crash(dataset_path):
    """Recover from data collection crash."""

    # Check dataset integrity
    dataset = LeRobotDataset(dataset_path)

    # Find incomplete episodes
    incomplete_episodes = []
    for episode_idx in range(dataset.num_episodes):
        from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
        to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

        if to_idx - from_idx < 10:  # Episode too short
            incomplete_episodes.append(episode_idx)

    # Remove incomplete episodes
    for episode_idx in incomplete_episodes:
        dataset.remove_episode(episode_idx)

    # Rebuild dataset
    dataset.rebuild()

    print(f"Recovered dataset: {dataset.num_episodes} episodes")
    return dataset
```

## Conclusion

This guide provides comprehensive coverage of real-world data collection with LeRobot. Key takeaways:

- Follow safety procedures and use appropriate safety equipment
- Ensure proper hardware setup and calibration
- Use quality control measures during data collection
- Monitor data quality and validate datasets
- Handle common issues with appropriate troubleshooting procedures
- Follow robot-specific setup procedures
- Implement safety monitoring and emergency procedures

For specific robot setups or advanced configurations, refer to the individual robot documentation pages in the LeRobot documentation.

---
noteId: "d8c23b80b09c11f08bd0898d85b76daa"
tags: []
---

# LeRobot Data Replay Guide

This comprehensive guide covers how to replay demonstration data using LeRobot, including various replay methods, tools, and best practices for analyzing recorded robot demonstrations.

## Table of Contents

1. [Overview](#overview)
2. [Basic Data Replay](#basic-data-replay)
3. [Advanced Replay Methods](#advanced-replay-methods)
4. [Replay Tools and Scripts](#replay-tools-and-scripts)
5. [Configuration and Customization](#configuration-and-customization)
6. [Analysis and Visualization](#analysis-and-visualization)
7. [Troubleshooting](#troubleshooting)

## Overview

Data replay in LeRobot allows you to:

- **Replay recorded demonstrations** on real robots or in simulation
- **Analyze demonstration quality** and consistency
- **Validate data collection** before training
- **Test robot behavior** with known action sequences
- **Debug and troubleshoot** robot control issues

### Key Components

- **LeRobotDataset**: Standardized dataset format for robot demonstrations
- **Replay Scripts**: Tools for executing recorded actions
- **Visualization**: Real-time monitoring and analysis tools
- **Robot Interface**: Connection to physical robots or simulation environments

## Basic Data Replay

### Using lerobot-replay Command

The primary tool for replaying demonstrations is the `lerobot-replay` command:

```bash
lerobot-replay \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=my_robot \
  --dataset.repo_id=user/dataset_name \
  --dataset.episode=0 \
  --display_data=true
```

### Key Parameters

#### Robot Configuration

- `--robot.type`: Robot type for replay execution
- `--robot.port`: Serial port for robot communication
- `--robot.id`: Unique identifier for the robot

#### Dataset Configuration

- `--dataset.repo_id`: Hugging Face repository ID of the dataset
- `--dataset.episode`: Specific episode index to replay
- `--dataset.root`: Local path to dataset (optional)

#### Display Options

- `--display_data=true`: Enable real-time visualization
- `--play_sounds=true`: Enable audio feedback

### Basic Replay Workflow

1. **Load Dataset**

   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   dataset = LeRobotDataset("user/dataset_name")
   ```

2. **Select Episode**

   ```python
   episode_frames = dataset.hf_dataset.filter(
       lambda x: x["episode_index"] == episode_number
   )
   ```

3. **Execute Actions**
   - Connect to robot
   - Iterate through episode frames
   - Send actions to robot at recorded timing
   - Monitor execution

## Advanced Replay Methods

### Custom Replay Scripts

Create custom replay scripts for specific analysis needs:

```python
#!/usr/bin/env python3
"""
Custom dataset replay script with analysis capabilities
"""

import argparse
import time
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def replay_episode_with_analysis(dataset_id, episode_idx, max_frames=100, fps=10.0):
    """Replay episode with detailed analysis and statistics."""

    print("📺 Advanced Dataset Replay with Analysis")
    print("=" * 50)

    # Load dataset
    print(f"📊 Loading dataset: {dataset_id}")
    dataset = LeRobotDataset(dataset_id)
    print(f"✅ Dataset loaded: {dataset.num_episodes} episodes, {dataset.num_frames} frames")

    # Validate episode
    if episode_idx >= dataset.num_episodes:
        print(f"❌ Episode {episode_idx} not found. Dataset has {dataset.num_episodes} episodes.")
        return

    # Get episode boundaries
    from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]
    episode_length = to_idx - from_idx

    print(f"\n📋 Episode {episode_idx} Details:")
    print(f"  Frames: {from_idx} to {to_idx - 1} (length: {episode_length})")
    print(f"  Replay FPS: {fps}")
    print(f"  Max frames to replay: {max_frames}")

    # Calculate timing
    frame_delay = 1.0 / fps
    frames_to_replay = min(max_frames, episode_length)
    estimated_duration = frames_to_replay * frame_delay

    print(f"  Estimated duration: {estimated_duration:.1f} seconds")

    # Collect action statistics
    all_actions = []
    action_ranges = {}

    try:
        # Replay frames with analysis
        for frame_idx in range(frames_to_replay):
            global_idx = from_idx + frame_idx
            frame = dataset[global_idx]

            # Extract action data
            action = frame["action"]
            all_actions.append(action.numpy())

            # Calculate action statistics
            if frame_idx == 0:
                for i, name in enumerate(dataset.features["action"]["names"]):
                    action_ranges[name] = {"min": action[i].item(), "max": action[i].item()}
            else:
                for i, name in enumerate(dataset.features["action"]["names"]):
                    action_ranges[name]["min"] = min(action_ranges[name]["min"], action[i].item())
                    action_ranges[name]["max"] = max(action_ranges[name]["max"], action[i].item())

            # Display progress
            if frame_idx % 10 == 0:
                progress = (frame_idx / frames_to_replay) * 100
                print(f"  Progress: {progress:.1f}% ({frame_idx}/{frames_to_replay})")

            # Simulate action execution timing
            time.sleep(frame_delay)

        # Display analysis results
        print(f"\n📈 Action Analysis Results:")
        print(f"  Total frames replayed: {frames_to_replay}")
        print(f"  Action ranges:")
        for name, range_info in action_ranges.items():
            print(f"    {name}: [{range_info['min']:.3f}, {range_info['max']:.3f}]")

        # Calculate action statistics
        all_actions = np.array(all_actions)
        print(f"  Action statistics:")
        for i, name in enumerate(dataset.features["action"]["names"]):
            mean_val = np.mean(all_actions[:, i])
            std_val = np.std(all_actions[:, i])
            print(f"    {name}: mean={mean_val:.3f}, std={std_val:.3f}")

        print(f"\n✅ Replay completed successfully!")

    except KeyboardInterrupt:
        print(f"\n⏹️  Replay interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during replay: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced dataset replay with analysis")
    parser.add_argument("--dataset", type=str, default="user/dataset_name", help="Dataset to replay")
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay")
    parser.add_argument("--max-frames", type=int, default=100, help="Maximum frames to replay")
    parser.add_argument("--fps", type=float, default=10.0, help="Replay FPS")

    args = parser.parse_args()
    replay_episode_with_analysis(args.dataset, args.episode, args.max_frames, args.fps)
```

### Action-Only Replay

For testing without full robot execution, use action-only replay:

```python
def replay_actions_only(dataset_id, episode_idx, max_frames=50):
    """Replay only the actions without robot execution for analysis."""

    print("🎬 Action-Only Replay")
    print("=" * 30)

    # Load dataset
    dataset = LeRobotDataset(dataset_id)

    # Get episode data
    from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]
    episode_length = to_idx - from_idx
    frames_to_replay = min(max_frames, episode_length)

    print(f"Replaying {frames_to_replay} actions from episode {episode_idx}")

    # Collect action data
    actions = []
    for frame_idx in range(frames_to_replay):
        global_idx = from_idx + frame_idx
        frame = dataset[global_idx]
        action = frame["action"]
        actions.append(action.numpy())

        # Display action every 10 frames
        if frame_idx % 10 == 0:
            action_str = f"[{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}]"
            print(f"  Frame {frame_idx}: {action_str}")

    # Analyze action patterns
    actions = np.array(actions)
    print(f"\nAction Analysis:")
    print(f"  Total actions: {len(actions)}")
    print(f"  Action shape: {actions.shape}")
    print(f"  Mean action: {np.mean(actions, axis=0)}")
    print(f"  Std action: {np.std(actions, axis=0)}")

    return actions
```

## Replay Tools and Scripts

### Built-in Replay Scripts

LeRobot provides several built-in replay tools:

#### 1. Simple Replay Script

```bash
python EVALUATE/replay_dataset_simple.py \
    --dataset "user/dataset_name" \
    --episode 0 \
    --max-frames 100 \
    --fps 10.0
```

#### 2. Action-Only Replay Script

```bash
python EVALUATE/replay_dataset_actions_only.py \
    --dataset "user/dataset_name" \
    --episode 0 \
    --max-frames 50 \
    --fps 10.0
```

#### 3. Batch Replay Script

```bash
#!/bin/bash
# Replay multiple episodes

DATASET="user/dataset_name"
MAX_FRAMES=100
FPS=10.0

for episode in {0..4}; do
    echo "Replaying episode $episode..."
    python EVALUATE/replay_dataset_simple.py \
        --dataset "$DATASET" \
        --episode "$episode" \
        --max-frames "$MAX_FRAMES" \
        --fps "$FPS"
    echo "Episode $episode completed."
done
```

### Custom Replay Utilities

#### Episode Comparison Tool

```python
def compare_episodes(dataset_id, episode1, episode2):
    """Compare two episodes from the same dataset."""

    dataset = LeRobotDataset(dataset_id)

    # Get episode data
    ep1_data = get_episode_data(dataset, episode1)
    ep2_data = get_episode_data(dataset, episode2)

    print(f"Episode {episode1} vs Episode {episode2}")
    print(f"Length: {len(ep1_data)} vs {len(ep2_data)}")

    # Compare action patterns
    ep1_actions = np.array([frame["action"].numpy() for frame in ep1_data])
    ep2_actions = np.array([frame["action"].numpy() for frame in ep2_data])

    # Calculate differences
    if len(ep1_actions) == len(ep2_actions):
        action_diff = np.mean(np.abs(ep1_actions - ep2_actions), axis=0)
        print(f"Mean action difference: {action_diff}")

    return ep1_data, ep2_data

def get_episode_data(dataset, episode_idx):
    """Extract all frames for a specific episode."""
    from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
    to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

    frames = []
    for global_idx in range(from_idx, to_idx):
        frames.append(dataset[global_idx])

    return frames
```

#### Dataset Statistics Tool

```python
def analyze_dataset_statistics(dataset_id):
    """Generate comprehensive dataset statistics."""

    dataset = LeRobotDataset(dataset_id)

    print(f"Dataset: {dataset_id}")
    print(f"Episodes: {dataset.num_episodes}")
    print(f"Total frames: {dataset.num_frames}")
    print(f"FPS: {dataset.fps}")

    # Episode length analysis
    episode_lengths = []
    for i in range(dataset.num_episodes):
        from_idx = dataset.meta.episodes[i]["dataset_from_index"]
        to_idx = dataset.meta.episodes[i]["dataset_to_index"]
        length = to_idx - from_idx
        episode_lengths.append(length)

    print(f"\nEpisode Length Statistics:")
    print(f"  Mean: {np.mean(episode_lengths):.1f} frames")
    print(f"  Std: {np.std(episode_lengths):.1f} frames")
    print(f"  Min: {np.min(episode_lengths)} frames")
    print(f"  Max: {np.max(episode_lengths)} frames")

    # Action analysis across all episodes
    all_actions = []
    for i in range(min(10, dataset.num_episodes)):  # Sample first 10 episodes
        episode_data = get_episode_data(dataset, i)
        for frame in episode_data:
            all_actions.append(frame["action"].numpy())

    all_actions = np.array(all_actions)
    print(f"\nAction Statistics (sampled):")
    for i, name in enumerate(dataset.features["action"]["names"]):
        mean_val = np.mean(all_actions[:, i])
        std_val = np.std(all_actions[:, i])
        print(f"  {name}: mean={mean_val:.3f}, std={std_val:.3f}")

    return {
        "episode_lengths": episode_lengths,
        "action_stats": all_actions
    }
```

## Configuration and Customization

### Replay Configuration Files

Create configuration files for consistent replay setups:

```json
{
  "replay_config": {
    "dataset": {
      "repo_id": "user/dataset_name",
      "episode": 0,
      "max_frames": 100
    },
    "robot": {
      "type": "so101_follower",
      "port": "/dev/ttyACM0",
      "id": "replay_robot"
    },
    "display": {
      "enabled": true,
      "fps": 10.0,
      "show_actions": true,
      "show_observations": true
    },
    "analysis": {
      "collect_statistics": true,
      "save_results": true,
      "output_dir": "replay_results"
    }
  }
}
```

### Custom Replay Processors

Implement custom data processing for replay:

```python
from lerobot.processor import ProcessorStep

class ReplayActionProcessor(ProcessorStep):
    """Custom processor for replay actions."""

    def __init__(self, scaling_factor=1.0, offset=None):
        self.scaling_factor = scaling_factor
        self.offset = offset or [0.0] * 4

    def process(self, action):
        """Process action for replay."""
        processed_action = action * self.scaling_factor + self.offset
        return processed_action

class ReplayObservationProcessor(ProcessorStep):
    """Custom processor for replay observations."""

    def __init__(self, crop_region=None, resize=None):
        self.crop_region = crop_region
        self.resize = resize

    def process(self, observation):
        """Process observation for replay."""
        if self.crop_region and "images" in observation:
            for key, image in observation["images"].items():
                x, y, w, h = self.crop_region
                observation["images"][key] = image[y:y+h, x:x+w]

        if self.resize and "images" in observation:
            for key, image in observation["images"].items():
                observation["images"][key] = cv2.resize(image, self.resize)

        return observation
```

## Analysis and Visualization

### Real-time Visualization

Use Rerun for real-time replay visualization:

```python
import rerun as rr

def visualize_replay(dataset_id, episode_idx):
    """Visualize replay with Rerun."""

    # Initialize Rerun
    rr.init("lerobot_replay", spawn=True)

    dataset = LeRobotDataset(dataset_id)
    episode_data = get_episode_data(dataset, episode_idx)

    for frame_idx, frame in enumerate(episode_data):
        # Log action data
        action = frame["action"].numpy()
        rr.log("action", rr.Scalar(action[0]), time=frame_idx)

        # Log observation data
        if "observation" in frame:
            obs = frame["observation"]
            if "images" in obs:
                for cam_name, image in obs["images"].items():
                    rr.log(f"images/{cam_name}", rr.Image(image), time=frame_idx)

            if "state" in obs:
                state = obs["state"].numpy()
                rr.log("state", rr.Scalar(state[0]), time=frame_idx)

        time.sleep(0.1)  # Control replay speed
```

### Action Trajectory Analysis

```python
def analyze_action_trajectories(dataset_id, episode_idx):
    """Analyze action trajectories for patterns."""

    dataset = LeRobotDataset(dataset_id)
    episode_data = get_episode_data(dataset, episode_idx)

    # Extract action sequences
    actions = np.array([frame["action"].numpy() for frame in episode_data])

    # Calculate trajectory statistics
    trajectory_length = np.sum(np.linalg.norm(np.diff(actions, axis=0), axis=1))
    max_velocity = np.max(np.linalg.norm(np.diff(actions, axis=0), axis=1))
    mean_velocity = np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1))

    print(f"Trajectory Analysis:")
    print(f"  Total length: {trajectory_length:.3f}")
    print(f"  Max velocity: {max_velocity:.3f}")
    print(f"  Mean velocity: {mean_velocity:.3f}")

    # Plot trajectory
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    for i, name in enumerate(dataset.features["action"]["names"]):
        ax = axes[i//2, i%2]
        ax.plot(actions[:, i])
        ax.set_title(f"{name} over time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Value")

    plt.tight_layout()
    plt.savefig(f"trajectory_analysis_episode_{episode_idx}.png")
    plt.show()

    return {
        "trajectory_length": trajectory_length,
        "max_velocity": max_velocity,
        "mean_velocity": mean_velocity,
        "actions": actions
    }
```

### Episode Quality Assessment

```python
def assess_episode_quality(dataset_id, episode_idx):
    """Assess the quality of a demonstration episode."""

    dataset = LeRobotDataset(dataset_id)
    episode_data = get_episode_data(dataset, episode_idx)

    # Extract data
    actions = np.array([frame["action"].numpy() for frame in episode_data])

    # Quality metrics
    metrics = {}

    # 1. Smoothness (low acceleration)
    accelerations = np.diff(actions, n=2, axis=0)
    smoothness = np.mean(np.linalg.norm(accelerations, axis=1))
    metrics["smoothness"] = smoothness

    # 2. Consistency (low variance in action space)
    consistency = np.mean(np.std(actions, axis=0))
    metrics["consistency"] = consistency

    # 3. Completeness (coverage of action space)
    action_ranges = np.max(actions, axis=0) - np.min(actions, axis=0)
    completeness = np.mean(action_ranges)
    metrics["completeness"] = completeness

    # 4. Efficiency (directness of trajectory)
    start_action = actions[0]
    end_action = actions[-1]
    direct_distance = np.linalg.norm(end_action - start_action)
    actual_distance = np.sum(np.linalg.norm(np.diff(actions, axis=0), axis=1))
    efficiency = direct_distance / actual_distance if actual_distance > 0 else 0
    metrics["efficiency"] = efficiency

    print(f"Episode {episode_idx} Quality Assessment:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.3f}")

    return metrics
```

## Troubleshooting

### Common Issues

#### Dataset Loading Problems

```python
# Check dataset availability
try:
    dataset = LeRobotDataset("user/dataset_name")
    print("Dataset loaded successfully")
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Check if dataset exists on Hugging Face Hub
    from huggingface_hub import list_datasets
    datasets = list_datasets(filter="lerobot")
    print("Available LeRobot datasets:", [d.id for d in datasets])
```

#### Robot Connection Issues

```bash
# Check robot connection
ls /dev/tty*
# Test robot communication
python -c "import serial; ser = serial.Serial('/dev/ttyACM0'); print('Connected')"
```

#### Memory Issues

```python
# For large datasets, use streaming
from lerobot.datasets.lerobot_dataset import StreamingLeRobotDataset

dataset = StreamingLeRobotDataset("user/dataset_name")
# This loads data on-demand instead of all at once
```

### Performance Optimization

#### Efficient Replay

```python
def efficient_replay(dataset_id, episode_idx, batch_size=10):
    """Replay with optimized performance."""

    dataset = LeRobotDataset(dataset_id)
    episode_data = get_episode_data(dataset, episode_idx)

    # Process in batches
    for i in range(0, len(episode_data), batch_size):
        batch = episode_data[i:i+batch_size]

        # Process batch
        for frame in batch:
            # Execute action
            pass

        # Clear memory
        del batch
```

#### Parallel Analysis

```python
from multiprocessing import Pool

def analyze_episode_parallel(args):
    """Analyze single episode in parallel."""
    dataset_id, episode_idx = args
    return assess_episode_quality(dataset_id, episode_idx)

def analyze_dataset_parallel(dataset_id, num_episodes=10):
    """Analyze multiple episodes in parallel."""

    args = [(dataset_id, i) for i in range(num_episodes)]

    with Pool() as pool:
        results = pool.map(analyze_episode_parallel, args)

    return results
```

## Conclusion

This guide provides comprehensive coverage of LeRobot data replay capabilities. Key takeaways:

- Use `lerobot-replay` for basic replay functionality
- Implement custom scripts for advanced analysis
- Leverage visualization tools for quality assessment
- Follow best practices for efficient and reliable replay
- Use parallel processing for large-scale analysis

For specific robot or dataset configurations, refer to the individual documentation pages in the LeRobot documentation.

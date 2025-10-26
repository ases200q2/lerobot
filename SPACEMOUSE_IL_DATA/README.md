---
noteId: "50f5e6d0b21711f0a9b1cb8d56441717"
tags: []

---

# SpaceMouse Imitation Learning Pipeline for PandaPickCube

A complete, modular pipeline for collecting SpaceMouse demonstrations, training ACT policies with imitation learning, and evaluating on the PandaPickCubeSpacemouse-v0 simulation environment. Designed for simplicity and extensibility to Isaac Sim.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Quick Start](#quick-start)
- [Detailed Workflow](#detailed-workflow)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Extensibility to Isaac Sim](#extensibility-to-isaac-sim)
- [Troubleshooting](#troubleshooting)
- [Advanced Usage](#advanced-usage)

## Overview

This project provides a **complete end-to-end pipeline** for robot imitation learning:

1. **Data Collection**: Use SpaceMouse 3D controller for intuitive teleoperation
2. **Training**: Train Action Chunking Transformer (ACT) policy with imitation learning
3. **Evaluation**: Test trained policy in simulation with comprehensive metrics

**Key Design Principles:**
- **Simplicity**: Each component is minimal and focused
- **Pure Imitation Learning**: No RL or HIL-SERL frameworks
- **Modularity**: Easy to modify individual components
- **Extensibility**: Designed for future Isaac Sim integration

## Features

- ✅ SpaceMouse teleoperation for natural 6-DOF control
- ✅ Automatic data collection with episode management
- ✅ HuggingFace Hub integration for dataset storage
- ✅ ACT (Action Chunking Transformer) policy training
- ✅ Proper observation preprocessing (image crop/resize, state vectors)
- ✅ Action transformation handling (4-dim policy → 7-dim environment)
- ✅ Real-time GUI visualization during evaluation
- ✅ Comprehensive logging and metrics
- ✅ Easy-to-use shell scripts for all operations
- ✅ Extensible to Isaac Sim environments

## Quick Start

### Prerequisites

```bash
# Install LeRobot with gym_hil support
cd /path/to/lerobot
pip install -e ".[hilserl]"

# Verify SpaceMouse is connected
python -c "import hid; print([d for d in hid.enumerate() if '3Dconnexion' in str(d)])"
```

### 1. Configure Your Project

Update the repository IDs in configuration files:

```bash
# Edit data collection config
nano SPACEMOUSE_IL_DATA/configs/collect_data_config.json
# Change: "repo_id": "YOUR_HF_USERNAME/panda_spacemouse_il_data"
```

### 2. Collect Demonstration Data

```bash
# Collect 30 episodes with SpaceMouse
cd SPACEMOUSE_IL_DATA
./scripts/run_data_collection.sh

# Or specify custom parameters
./scripts/run_data_collection.sh 50 username/custom_dataset
```

**SpaceMouse Controls:**
- **Movement**: Move SpaceMouse in 3D space → end-effector position
- **Rotation**: Rotate SpaceMouse → end-effector orientation
- **Left Button**: Close gripper
- **Right Button**: Open gripper
- **SPACE**: Mark episode as SUCCESS
- **C**: Mark episode as FAILURE
- **R**: Re-record current episode

### 3. Train ACT Policy

```bash
# Train with collected data
./scripts/run_training.sh username/panda_spacemouse_il_data

# Custom training parameters
./scripts/run_training.sh username/dataset 50000 16  # 50k steps, batch size 16
```

Training takes approximately 30-60 minutes on a modern GPU.

### 4. Evaluate Trained Policy

```bash
# Evaluate policy from HuggingFace Hub
./scripts/run_evaluation.sh username/panda_spacemouse_act_policy

# Evaluate local checkpoint
./scripts/run_evaluation.sh outputs/train/panda_spacemouse_act_*/checkpoints/last/pretrained_model
```

## Detailed Workflow

### Phase 1: Data Collection

#### Configuration

The data collection is configured via `configs/collect_data_config.json`:

```json
{
  "task": "PandaPickCubeSpacemouse-v0",
  "num_episodes": 30,
  "fps": 10,
  "control_mode": "spacemouse",
  "repo_id": "username/panda_spacemouse_il_data",
  "push_to_hub": true
}
```

**Key Parameters:**
- `num_episodes`: Number of demonstrations to collect (30-50 recommended)
- `fps`: Recording frame rate (10 FPS default)
- `control_mode`: Set to "spacemouse" for SpaceMouse teleoperation
- `push_to_hub`: Upload to HuggingFace Hub automatically

#### Image Processing

Images are preprocessed during collection:
- **Crop**: Top-left corner to 128×128
- **Resize**: Ensure 128×128 final size
- **Cameras**: Front and wrist views

#### State Vector

18-dimensional state vector:
- **7 dims**: Joint positions (Panda arm)
- **7 dims**: Joint velocities
- **4 dims**: End-effector pose (xyz + gripper)

#### Action Space

4-dimensional actions:
- **3 dims**: Delta position (dx, dy, dz)
- **1 dim**: Gripper state (open/close)

#### Running Collection

```bash
# Basic usage
./scripts/run_data_collection.sh

# Advanced usage with Python script
python scripts/collect_data.py \
    --config configs/collect_data_config.json \
    --episodes 50 \
    --repo_id username/custom_dataset
```

#### Data Quality Tips

1. **Consistent Demonstrations**: Use similar trajectories for the same task
2. **Smooth Movements**: Avoid jerky or rapid movements
3. **Success Marking**: Press SPACE only when task is truly successful
4. **Lighting**: Ensure consistent, good lighting conditions
5. **Camera Position**: Keep cameras fixed throughout collection

### Phase 2: Training

#### Configuration

Training is configured via `configs/train_act_config.json`:

```json
{
  "policy": {
    "type": "act",
    "chunk_size": 100,
    "dim_model": 256
  },
  "training": {
    "batch_size": 32,
    "steps": 100000,
    "lr": 1e-5
  }
}
```

**Key Hyperparameters:**
- `batch_size`: 32 (reduce if GPU memory issues)
- `steps`: 100,000 (standard for this task)
- `lr`: 1e-5 (learning rate)
- `chunk_size`: 100 (action chunking horizon)

#### Why ACT?

Action Chunking Transformer (ACT) is chosen because:
- ✅ State-of-the-art for manipulation tasks
- ✅ Handles temporal dependencies via action chunking
- ✅ Uses transformer architecture for visual+state learning
- ✅ Strong performance with limited demonstrations

#### Running Training

```bash
# Using shell script (recommended)
./scripts/run_training.sh username/panda_spacemouse_il_data

# Using Python script directly
python scripts/train_policy.py \
    --dataset_id username/panda_spacemouse_il_data \
    --steps 100000 \
    --batch_size 32 \
    --wandb  # Optional: Enable Weights & Biases logging
```

#### Monitoring Training

1. **Terminal Output**: Real-time loss and metrics
2. **TensorBoard**: `tensorboard --logdir outputs/train/`
3. **Weights & Biases**: If enabled with `--wandb` flag

#### Checkpoints

Checkpoints are saved to:
```
outputs/train/panda_spacemouse_act_TIMESTAMP/checkpoints/
├── 010000/           # Checkpoint at 10k steps
├── 020000/           # Checkpoint at 20k steps
├── ...
└── last/             # Latest checkpoint
    └── pretrained_model/  # Use this for evaluation
```

### Phase 3: Evaluation

#### Configuration

Evaluation is configured via `configs/eval_policy_config.json`:

```json
{
  "evaluation": {
    "n_episodes": 10,
    "fps": 10
  },
  "visualization": {
    "gui": true
  }
}
```

#### Running Evaluation

```bash
# Evaluate HuggingFace Hub model
./scripts/run_evaluation.sh username/panda_spacemouse_act_policy username/dataset

# Evaluate local checkpoint
./scripts/run_evaluation.sh outputs/train/*/checkpoints/last/pretrained_model

# Custom number of episodes
./scripts/run_evaluation.sh username/policy username/dataset 20
```

#### Metrics

The evaluation script reports:
- **Success Rate**: Percentage of successful episodes
- **Average Reward**: Mean reward across episodes
- **Episode Length**: Steps per episode
- **Reward Range**: Min and max rewards

#### GUI Visualization

During evaluation with `--gui` (default):
- **Front Camera**: 512×512 window showing front view
- **Wrist Camera**: 512×512 window showing wrist view
- **Press 'q'**: Quit evaluation early

#### Understanding Results

**Good Performance:**
- Success rate > 70%
- Consistent episode lengths
- Smooth trajectories in visualization

**Poor Performance (requires improvement):**
- Success rate < 30%
- Erratic movements
- Early terminations

**If performance is poor:**
1. Collect more demonstration data (50-100 episodes)
2. Train for more steps (200k+)
3. Check data quality (visualize dataset)
4. Adjust hyperparameters (batch size, learning rate)

## Project Structure

```
SPACEMOUSE_IL_DATA/
├── configs/                          # Configuration files
│   ├── collect_data_config.json     # Data collection settings
│   ├── train_act_config.json        # Training hyperparameters
│   └── eval_policy_config.json      # Evaluation settings
│
├── scripts/                          # Executable scripts
│   ├── collect_data.py              # Data collection Python script
│   ├── run_data_collection.sh       # Data collection shell wrapper
│   ├── train_policy.py              # Training Python script
│   ├── run_training.sh              # Training shell wrapper
│   ├── eval_policy.py               # Evaluation Python script
│   └── run_evaluation.sh            # Evaluation shell wrapper
│
├── data/                             # Local data storage (optional)
│
├── docs/                             # Documentation
│   ├── ISAAC_SIM_EXTENSION.md       # Isaac Sim integration guide
│   └── TROUBLESHOOTING.md           # Common issues and solutions
│
└── README.md                         # This file
```

## Configuration

### Environment Variables

Set your HuggingFace username for automatic configuration:

```bash
export HF_USER=your_username
```

### HuggingFace Authentication

For uploading datasets and models:

```bash
# Login to HuggingFace Hub
huggingface-cli login

# Or set token as environment variable
export HF_TOKEN=your_token
```

### GPU Configuration

The pipeline automatically detects and uses available GPUs. To force CPU:

```bash
# Data collection
python scripts/collect_data.py --config configs/collect_data_config.json --device cpu

# Training
./scripts/run_training.sh username/dataset 100000 32 cpu

# Evaluation
python scripts/eval_policy.py --policy_path username/policy --device cpu
```

## Extensibility to Isaac Sim

This pipeline is **designed for easy extension to NVIDIA Isaac Sim**. See [`docs/ISAAC_SIM_EXTENSION.md`](docs/ISAAC_SIM_EXTENSION.md) for detailed integration guide.

### Key Compatibility Features

1. **LeRobotDataset v3.0 Format**: Compatible with Isaac Sim data loaders
2. **Modular Observation Processing**: Easy to adapt to Isaac Sim observations
3. **Configurable Action Spaces**: Action transformations cleanly separated
4. **Environment-Agnostic Policy**: ACT policy works with any observation/action space

### Migration Path

```
Current: gym-hil/PandaPickCubeSpacemouse-v0
         ↓
Future:  Isaac Sim/Panda Pick-and-Place Task

Changes Required:
1. Update environment creation in eval_policy.py
2. Adjust observation wrapper for Isaac Sim observations
3. Adapt action transformation if needed
4. Test and tune on Isaac Sim environment
```

## Troubleshooting

See [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md) for detailed troubleshooting guide.

### Quick Fixes

**SpaceMouse not detected:**
```bash
pip install hidapi
python -c "import hid; print(hid.enumerate())"
```

**CUDA out of memory:**
```bash
# Reduce batch size
./scripts/run_training.sh username/dataset 100000 16  # Batch size 16
```

**Dataset not found:**
```bash
# Verify HuggingFace authentication
huggingface-cli whoami

# Check dataset exists
huggingface-cli repo info username/dataset --repo-type dataset
```

## Advanced Usage

### Resume Training

```bash
python scripts/train_policy.py \
    --dataset_id username/dataset \
    --resume \
    --output_dir outputs/train/previous_run
```

### Custom Training Configuration

```bash
python scripts/train_policy.py \
    --config configs/train_act_config.json \
    --dataset_id username/dataset \
    --steps 200000 \
    --batch_size 64 \
    --wandb
```

### Visualize Dataset

```bash
# Online visualizer
# Visit: https://huggingface.co/spaces/lerobot/visualize_dataset
# Enter: username/panda_spacemouse_il_data

# Or use lerobot-dataset-viz
lerobot-dataset-viz --repo-id username/panda_spacemouse_il_data --episode-index 0
```

### Upload Checkpoints Manually

```bash
huggingface-cli upload username/model_name \
    outputs/train/panda_spacemouse_act_*/checkpoints/last/pretrained_model \
    --repo-type model
```

## Citation

If you use this pipeline in your research, please cite:

```bibtex
@software{spacemouse_il_pipeline,
  title = {SpaceMouse Imitation Learning Pipeline for Robot Manipulation},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/your-repo}
}

@article{zhao2023learning,
  title={Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware},
  author={Zhao, Tony Z and Kumar, Vikash and Levine, Sergey and Finn, Chelsea},
  journal={arXiv preprint arXiv:2304.13705},
  year={2023}
}
```

## License

This project follows the LeRobot Apache 2.0 license.

## Support

For questions or issues:
1. Check [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md)
2. Review LeRobot documentation: https://github.com/huggingface/lerobot
3. Open an issue on GitHub
4. Join LeRobot Discord: https://discord.com/invite/s3KuuzsPFb

---

**Happy Learning! 🤖🎮**

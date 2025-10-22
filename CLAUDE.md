---
noteId: "b6247590aee211f08bd0898d85b76daa"
tags: []
---

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Installation and Setup

```bash
# Install from source (development mode)
pip install -e .

# Install with specific features
pip install -e ".[aloha,pusht]"  # Simulation environments
pip install -e ".[all]"          # All features
pip install -e ".[dev,test]"     # Development dependencies
```

### Training and Evaluation

```bash
# Train a policy
lerobot-train --policy.type=act --env.type=aloha --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human

# Train from config file
lerobot-train --config_path=lerobot/diffusion_pusht

# Evaluate a trained policy
lerobot-eval --policy.path=path/to/pretrained_model --env.type=aloha

# Resume training from checkpoint
lerobot-train --config_path=path/to/train_config.json --resume=true
```

### Data Collection and Visualization

```bash
# Record datasets via teleoperation
lerobot-record --policy.path=path/to/model --output-dir=./data

# Teleoperate a robot
lerobot-teleoperate --robot-config-path=configs/robot/so100.yaml

# Visualize datasets
lerobot-dataset-viz --repo-id lerobot/pusht --episode-index 0

# Replay recorded data
lerobot-replay --root ./data
```

### Testing and Quality

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/policies/test_policies.py

# Run with coverage
pytest --cov=src/lerobot

# Lint and format
ruff check src/
ruff format src/

# End-to-end tests (for developers)
make test-end-to-end DEVICE=cpu
```

### Robot Hardware

```bash
# Find available cameras and ports
lerobot-find-cameras
lerobot-find-port

# Calibrate robot
lerobot-calibrate

# Setup motors
lerobot-setup-motors --robot-config-path=configs/robot/so100.yaml

# Find joint limits
lerobot-find-joint-limits --robot-config-path=configs/robot/so100.yaml
```

## High-Level Architecture

LeRobot is a modular robotics framework with a configuration-driven architecture:

### Core Components

- **`datasets/`**: Dataset handling with `LeRobotDataset` as the central data structure, supporting streaming, transformations, and HuggingFace Hub integration
- **`policies/`**: Policy implementations (ACT, Diffusion, TDMPC, VQBeT, etc.) with unified `PreTrainedPolicy` interface
- **`robots/`**: Robot abstraction layer supporting multiple hardware platforms (SO100, SO101, Reachy2, etc.)
- **`processor/`**: Data processing pipelines for converting between policy, robot, and environment representations
- **`envs/`**: Environment definitions for simulation and real-world tasks
- **`scripts/`**: Main entry points for training, recording, evaluation, and utilities

### Key Patterns

**Factory Pattern**: The codebase uses `make_policy()`, `make_robot_from_config()`, `make_dataset()`, and `make_env()` functions for component instantiation.

**Configuration-Driven**: Uses YAML configs with `draccus` for type-safe configurations. All components can be configured and overridden via command line arguments.

**Data Flow Pipeline**:

- `RobotObservation` → `processor` → `EnvAction` → `policy` → `PolicyAction` → `processor` → `RobotAction`

### Main Entry Points

- **Training**: `lerobot-train` - Main training script with comprehensive configuration options
- **Data Collection**: `lerobot-record` - Dataset recording via teleoperation or policy-based collection
- **Evaluation**: `lerobot-eval` - Policy evaluation with comprehensive metrics
- **Teleoperation**: `lerobot-teleoperate` - Real-time robot control interface
- **Visualization**: `lerobot-dataset-viz` - Dataset visualization and analysis tools

### Configuration System

The configuration system is hierarchical:

- Base configs in `lerobot/configs/`
- Robot-specific configs define hardware parameters
- Policy-specific configs define algorithm hyperparameters
- Training configs define learning parameters

Configurations can be overridden via command line using dot notation:

```bash
lerobot-train --policy.type=act --policy.dim_model=256 --batch_size=32
```

### Development Notes

- Supports Python 3.10+ and PyTorch 2.2+
- Uses `ruff` for linting and formatting (configured for line length 110)
- Extensive type hints with `mypy` support (gradually being enabled)
- Test suite uses `pytest` with fixtures for common test scenarios
- Integration with HuggingFace Hub for dataset and model sharing

### Robot Support

The framework supports various robot types:

- **SO100/SO101**: Affordable robot arms for manipulation tasks
- **ALOHA**: Bimanual teleoperation system
- **Reachy2**: Advanced humanoid robot
- **Custom robots**: Easy to add new robot types via configuration

Each robot is defined by its configuration file specifying motors, cameras, kinematics, and calibration parameters.

---
noteId: "3810e3a0b01811f08051957781aff2eb"
tags: []

---

# Simple Gym HIL Evaluation Script

This directory contains a simple evaluation script for testing trained ACT policies on the gym_hil simulation environment.

## Quick Start

1. **Activate the conda environment:**
   ```bash
   conda activate lerobot-org
   ```

2. **Run the evaluation script:**
   ```bash
   python EVAL/eval_gym_hil_simple.py
   ```

## Script Details

### `eval_gym_hil_simple.py`

A minimal evaluation script that:
- Loads a trained ACT policy from HuggingFace Hub
- Creates gym_hil PandaPickCubeKeyboard-v0 simulation environments with OpenCV GUI
- Uses the same configuration as data collection (gripper control, random positioning)
- Runs evaluation episodes with real-time camera visualization
- Outputs success rate, rewards, and timing information

### Default Configuration

- **Checkpoint**: `ases200q2/PandaPickCubeSpacemouseRandom2_ACT_test_20251023_134850`
- **Environment**: `PandaPickCubeKeyboard-v0` (with OpenCV GUI visualization)
- **Episodes**: 10
- **Batch Size**: 5 parallel environments
- **Device**: CUDA (GPU acceleration)
- **GUI**: OpenCV-based camera visualization (Front and Wrist cameras)
- **FPS Control**: Proper timing control matching training configuration (default: 10 FPS)

### Command Line Options

```bash
python EVAL/eval_gym_hil_simple.py [OPTIONS]

Options:
  --checkpoint CHECKPOINT    HuggingFace Hub checkpoint path or local path
  --episodes EPISODES        Number of evaluation episodes (default: 10)
  --batch_size BATCH_SIZE    Number of parallel environments (default: 5)
  --device DEVICE            Device to use: cuda, cpu, mps (default: cuda)
  --seed SEED                Random seed for reproducibility (default: 42)
  --output_dir OUTPUT_DIR    Output directory (default: outputs/eval/timestamp)
```

### Example Usage

```bash
# Basic evaluation with default settings
python EVAL/eval_gym_hil_simple.py

# Custom checkpoint and more episodes
python EVAL/eval_gym_hil_simple.py --checkpoint ases200q2/my_model --episodes 20

# CPU evaluation with custom output directory
python EVAL/eval_gym_hil_simple.py --device cpu --output_dir my_eval_results

# Local checkpoint evaluation
python EVAL/eval_gym_hil_simple.py --checkpoint outputs/train/my_model/checkpoints/020000/pretrained_model

# Disable GUI for headless evaluation
python EVAL/eval_gym_hil_simple.py --no-gui --episodes 20

# Enable GUI (default behavior)
python EVAL/eval_gym_hil_simple.py --gui

# Custom FPS for timing control
python EVAL/eval_gym_hil_simple.py --fps 5 --episodes 3
```

### Output

The script creates:
- **GUI**: OpenCV windows showing Front and Wrist camera feeds in real-time (if enabled)
- **Results**: `outputs/eval/[timestamp]/eval_info.json` - Detailed evaluation metrics
- **Console**: Success rate, average rewards, timing information

### Sample Output

```
============================================================
EVALUATION RESULTS
============================================================
Success Rate: 80.0%
Average Sum Reward: 0.750
Average Max Reward: 1.000
Evaluation Time: 45.23 seconds
Episodes per Second: 0.22
Total Episodes: 10
GUI: OpenCV windows were displayed during evaluation
Note: If you didn't see the windows, they might be behind other windows or on a different desktop
Results saved to: outputs/eval/20250123_143022/
```

## Requirements

- LeRobot environment with gym_hil installed
- CUDA-capable GPU (recommended)
- HuggingFace Hub access for downloading checkpoints

## Troubleshooting

1. **ModuleNotFoundError**: Make sure to activate the `lerobot-org` conda environment
2. **CUDA out of memory**: Reduce batch size with `--batch_size 2` or use `--device cpu`
3. **Checkpoint not found**: Verify the HuggingFace Hub path or use a local checkpoint path

---
noteId: "e2239310afe111f08bd0898d85b76daa"
tags: []
---

# Simple ACT Model Evaluation

This directory contains simple evaluation scripts for trained ACT policies.

## Files

- `evaluate_act_simple.py` - Evaluate ACT model on dataset (MSE error)
- `evaluate_act_simulation_simple.py` - Evaluate ACT model in simulation
- `run_eval.sh` - Simple script to run dataset evaluation

## Quick Start

### 1. Evaluate on Dataset (MSE Error)

```bash
# Use the trained model from your recent training
./EVALUATE/run_eval.sh

# Or specify custom parameters
./EVALUATE/run_eval.sh --checkpoint "your-model-repo" --episodes 10
```

### 2. Evaluate in Simulation

```bash
# Run simulation evaluation
python EVALUATE/evaluate_act_simulation_simple.py \
    --checkpoint "ases200q2/PandaPickCubeSpacemouseRandom2_ACT_test_20251023_134850" \
    --num-episodes 3
```

## Parameters

### Dataset Evaluation

- `--checkpoint`: Model checkpoint or HuggingFace repo
- `--dataset`: Dataset to evaluate on (default: ases200q2/PandaPickCubeSpacemouseRandom2_v30)
- `--num-episodes`: Number of episodes to test (default: 5)
- `--device`: Device to use (default: cuda:0)

### Simulation Evaluation

- `--checkpoint`: Model checkpoint or HuggingFace repo
- `--num-episodes`: Number of episodes to run (default: 3)
- `--device`: Device to use (default: cuda:0)

## Example Output

### Dataset Evaluation

```
🤖 Simple ACT Model Evaluation
========================================
📂 Loading policy: ases200q2/PandaPickCubeSpacemouseRandom2_ACT_test_20251023_134850
✅ Policy loaded
📊 Loading dataset: ases200q2/PandaPickCubeSpacemouseRandom2_v30
✅ Dataset loaded: 50 episodes

🧪 Testing 5 episodes...

Episode 1/5
  Frames: 200
  Mean MSE: 0.012345

📊 RESULTS
========================================
Episodes tested: 5
Successful predictions: 250/250
Mean MSE: 0.012345 ± 0.001234
Min/Max MSE: 0.008765 / 0.015432
✅ GOOD - Low error
```

### Simulation Evaluation

```
🤖 Simple ACT Simulation Evaluation
========================================
📂 Loading policy: ases200q2/PandaPickCubeSpacemouseRandom2_ACT_test_20251023_134850
✅ Policy loaded
🎮 Creating simulation environment...
✅ Environment created

🎮 Running 3 episodes...

Episode 1/3
  ✅ Success at step 45!
  Success: ✅ Yes
  Length: 45 steps
  Reward: 0.850

📊 SIMULATION RESULTS
========================================
Episodes completed: 3
Success rate: 66.7%
Average reward: 0.750
Average length: 52.3 steps
✅ GOOD - Decent success rate
```

## Notes

- The scripts are designed to be as simple as possible
- Dataset evaluation measures MSE error between predicted and ground truth actions
- Simulation evaluation measures success rate and rewards in the gym_hil environment
- Both scripts handle action chunking automatically
- Error handling is included for robustness

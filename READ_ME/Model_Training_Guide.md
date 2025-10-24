---
noteId: "d8c23b81b09c11f08bd0898d85b76daa"
tags: []
---

# LeRobot Model Training Guide

This comprehensive guide covers how to train various robot learning models using LeRobot, with detailed focus on ACT (Action Chunking Transformer) and other available policies.

## Table of Contents

1. [Overview](#overview)
2. [ACT Training](#act-training)
3. [Other Policy Training](#other-policy-training)
4. [Training Configuration](#training-configuration)
5. [Training Workflow](#training-workflow)
6. [Evaluation and Testing](#evaluation-and-testing)
7. [Advanced Training Techniques](#advanced-training-techniques)
8. [Troubleshooting](#troubleshooting)

## Overview

LeRobot provides a unified training framework for various robot learning policies:

- **ACT (Action Chunking Transformer)**: Transformer-based policy for smooth action sequences
- **Diffusion Policy**: Diffusion-based policy for diverse action generation
- **TDMPC**: Temporal Difference Model Predictive Control
- **VQ-BeT**: Vector Quantized Behavior Transformer
- **SmolVLA**: Small Vision-Language-Action model

### Training Pipeline

The training process follows this workflow:

1. **Dataset Loading**: Load LeRobotDataset v3.0 format data
2. **Data Preprocessing**: Apply transforms and augmentations
3. **Model Initialization**: Create policy with specified architecture
4. **Training Loop**: Iterative optimization with loss computation
5. **Checkpointing**: Save model weights and configurations
6. **Evaluation**: Test trained policy on validation data

## ACT Training

ACT (Action Chunking Transformer) is a transformer-based policy that processes robot states and camera inputs to generate smooth, chunked action sequences.

### ACT Architecture

ACT uses a transformer encoder-decoder architecture:

- **Input**: Current robot state + camera images + latent style variable
- **Output**: Chunk of k future action sequences
- **Key Features**: Action chunking, temporal ensembling, VAE support

### Basic ACT Training

#### Command Line Training

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=act \
  --output_dir=outputs/train/act_your_dataset \
  --job_name=act_your_dataset \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/act_policy
```

#### Key ACT Parameters

```bash
# Architecture parameters
--policy.dim_model=512              # Transformer hidden dimension
--policy.n_heads=8                  # Attention heads
--policy.n_encoder_layers=4         # Encoder layers
--policy.n_decoder_layers=1         # Decoder layers

# Action chunking parameters
--policy.chunk_size=100             # Actions predicted per forward pass
--policy.n_action_steps=100         # Actions executed from chunk
--policy.temporal_ensemble_coeff=None # Smoothing coefficient

# VAE parameters (optional)
--policy.use_vae=True               # Enable VAE training
--policy.latent_dim=32             # Latent space dimension
--policy.kl_weight=10.0            # KL divergence weight

# Training parameters
--policy.optimizer_lr=1e-5         # Learning rate
--policy.dropout=0.1               # Dropout rate
--batch_size=32                    # Batch size
--steps=20000                      # Training steps
```

### ACT Configuration File

Create a configuration file for consistent ACT training:

```json
{
  "policy": {
    "type": "act",
    "dim_model": 512,
    "n_heads": 8,
    "n_encoder_layers": 4,
    "n_decoder_layers": 1,
    "chunk_size": 100,
    "n_action_steps": 100,
    "use_vae": true,
    "latent_dim": 32,
    "kl_weight": 10.0,
    "optimizer_lr": 1e-5,
    "dropout": 0.1
  },
  "dataset": {
    "repo_id": "user/dataset_name",
    "video_backend": "pyav"
  },
  "training": {
    "batch_size": 32,
    "steps": 20000,
    "device": "cuda"
  },
  "output": {
    "output_dir": "outputs/train/act_experiment",
    "job_name": "act_experiment"
  },
  "wandb": {
    "enable": true,
    "project": "lerobot_act"
  }
}
```

### ACT Training Script

```python
#!/usr/bin/env python3
"""
Custom ACT training script with advanced features
"""

import argparse
import torch
import wandb
from lerobot.scripts.lerobot_train import train
from lerobot.policies.act.config import ACTConfig

def train_act_with_custom_config(
    dataset_id,
    output_dir,
    job_name,
    steps=20000,
    batch_size=32,
    learning_rate=1e-5,
    use_vae=True,
    chunk_size=100
):
    """Train ACT policy with custom configuration."""

    print("🤖 ACT Training with Custom Configuration")
    print("=" * 50)

    # Create ACT configuration
    act_config = ACTConfig(
        dim_model=512,
        n_heads=8,
        n_encoder_layers=4,
        n_decoder_layers=1,
        chunk_size=chunk_size,
        n_action_steps=chunk_size,
        use_vae=use_vae,
        latent_dim=32,
        kl_weight=10.0,
        optimizer_lr=learning_rate,
        dropout=0.1
    )

    # Training arguments
    args = [
        f"--dataset.repo_id={dataset_id}",
        f"--policy.type=act",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        f"--policy.device=cuda",
        f"--wandb.enable=true",
        f"--policy.repo_id={job_name}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        f"--dataset.video_backend=pyav"
    ]

    # Add ACT-specific parameters
    for key, value in act_config.__dict__.items():
        if value is not None:
            args.append(f"--policy.{key}={value}")

    print(f"Training configuration:")
    print(f"  Dataset: {dataset_id}")
    print(f"  Output: {output_dir}")
    print(f"  Steps: {steps}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  VAE: {use_vae}")
    print(f"  Chunk size: {chunk_size}")

    # Start training
    try:
        train(args)
        print("✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Custom ACT training")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset ID")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--job-name", type=str, required=True, help="Job name")
    parser.add_argument("--steps", type=int, default=20000, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--use-vae", action="store_true", help="Use VAE")
    parser.add_argument("--chunk-size", type=int, default=100, help="Chunk size")

    args = parser.parse_args()

    train_act_with_custom_config(
        dataset_id=args.dataset,
        output_dir=args.output_dir,
        job_name=args.job_name,
        steps=args.steps,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        use_vae=args.use_vae,
        chunk_size=args.chunk_size
    )
```

### ACT Loss Function

ACT uses a combination of reconstruction loss and optional VAE loss:

```python
def compute_act_loss(actions_pred, actions_target, mu=None, log_sigma=None, use_vae=True, kl_weight=10.0):
    """Compute ACT training loss."""

    # L1 Reconstruction Loss
    l1_loss = F.l1_loss(actions_pred, actions_target)

    # Optional VAE KL Divergence
    if use_vae and mu is not None and log_sigma is not None:
        kld_loss = (-0.5 * (1 + log_sigma - mu**2 - log_sigma.exp())).sum(-1).mean()
        total_loss = l1_loss + kl_weight * kld_loss
    else:
        total_loss = l1_loss

    return total_loss, l1_loss, kld_loss if use_vae else None
```

## Other Policy Training

### Diffusion Policy Training

Diffusion Policy uses diffusion models to generate diverse action sequences:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=diffusion \
  --output_dir=outputs/train/diffusion_your_dataset \
  --job_name=diffusion_your_dataset \
  --policy.device=cuda \
  --wandb.enable=true \
  --policy.repo_id=${HF_USER}/diffusion_policy
```

#### Diffusion Policy Parameters

```bash
# Diffusion parameters
--policy.num_inference_steps=10     # Number of denoising steps
--policy.horizon=16                 # Action horizon
--policy.num_action_chunks=8        # Number of action chunks
--policy.num_epochs=1000            # Training epochs

# Architecture parameters
--policy.backbone=resnet18          # Backbone architecture
--policy.visual_features=512        # Visual feature dimension
--policy.action_dim=7               # Action dimension
```

### TDMPC Training

TDMPC (Temporal Difference Model Predictive Control) for model-based RL:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=tdmpc \
  --output_dir=outputs/train/tdmpc_your_dataset \
  --job_name=tdmpc_your_dataset \
  --policy.device=cuda \
  --wandb.enable=true
```

### SmolVLA Training

SmolVLA (Small Vision-Language-Action) for vision-language-action models:

```bash
lerobot-train \
  --dataset.repo_id=${HF_USER}/your_dataset \
  --policy.type=smolvla \
  --output_dir=outputs/train/smolvla_your_dataset \
  --job_name=smolvla_your_dataset \
  --policy.device=cuda \
  --wandb.enable=true
```

## Training Configuration

### Dataset Configuration

```json
{
  "dataset": {
    "repo_id": "user/dataset_name",
    "video_backend": "pyav",
    "download_videos": true,
    "episodes": null,
    "root": null
  }
}
```

### Training Parameters

```json
{
  "training": {
    "batch_size": 32,
    "steps": 20000,
    "device": "cuda",
    "num_workers": 4,
    "pin_memory": true,
    "persistent_workers": true
  }
}
```

### Output Configuration

```json
{
  "output": {
    "output_dir": "outputs/train/experiment_name",
    "job_name": "experiment_name",
    "save_freq": 1000,
    "eval_freq": 2000,
    "log_freq": 100
  }
}
```

### Weights & Biases Integration

```json
{
  "wandb": {
    "enable": true,
    "project": "lerobot_training",
    "entity": "your_username",
    "tags": ["act", "manipulation"],
    "notes": "Training ACT on custom dataset"
  }
}
```

## Training Workflow

### Pre-Training Setup

1. **Environment Preparation**

   ```bash
   # Activate conda environment
   conda activate lerobot

   # Verify GPU availability
   python -c "import torch; print(torch.cuda.is_available())"

   # Check dataset availability
   python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; dataset = LeRobotDataset('user/dataset_name'); print(f'Episodes: {dataset.num_episodes}')"
   ```

2. **Configuration Setup**
   - Create training configuration file
   - Set Hugging Face credentials
   - Configure Weights & Biases (optional)
   - Set output directories

3. **Dataset Validation**

   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset

   # Load and validate dataset
   dataset = LeRobotDataset("user/dataset_name")
   print(f"Dataset info:")
   print(f"  Episodes: {dataset.num_episodes}")
   print(f"  Frames: {dataset.num_frames}")
   print(f"  FPS: {dataset.fps}")
   print(f"  Features: {list(dataset.features.keys())}")

   # Check data quality
   sample_frame = dataset[0]
   print(f"Sample frame keys: {list(sample_frame.keys())}")
   ```

### Training Execution

1. **Start Training**

   ```bash
   # Basic training
   lerobot-train --config=training_config.json

   # With custom parameters
   lerobot-train \
     --dataset.repo_id=user/dataset_name \
     --policy.type=act \
     --output_dir=outputs/train/act_experiment \
     --job_name=act_experiment \
     --batch_size=32 \
     --steps=20000
   ```

2. **Monitor Training**
   - Watch console output for progress
   - Monitor Weights & Biases dashboard
   - Check GPU utilization
   - Monitor disk space

3. **Handle Interruptions**
   ```bash
   # Resume training from checkpoint
   lerobot-train --config=training_config.json --resume=true
   ```

### Post-Training

1. **Model Evaluation**

   ```bash
   # Evaluate trained model
   lerobot-eval \
     --policy.path=outputs/train/act_experiment/checkpoints/002000 \
     --dataset.repo_id=user/test_dataset \
     --output_dir=outputs/eval/act_experiment
   ```

2. **Model Upload**
   ```bash
   # Upload to Hugging Face Hub
   huggingface-cli upload user/act_policy outputs/train/act_experiment/checkpoints/002000/pretrained_model
   ```

## Evaluation and Testing

### Policy Evaluation

```bash
lerobot-eval \
  --policy.path=${HF_USER}/act_policy \
  --dataset.repo_id=${HF_USER}/test_dataset \
  --output_dir=outputs/eval/act_evaluation \
  --num_episodes=10
```

### Custom Evaluation Script

```python
#!/usr/bin/env python3
"""
Custom policy evaluation script
"""

import argparse
import torch
from lerobot.policies.act.policy import ACTPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def evaluate_policy(policy_path, dataset_id, num_episodes=10):
    """Evaluate trained policy on test dataset."""

    print("🔍 Policy Evaluation")
    print("=" * 30)

    # Load policy
    print(f"Loading policy from: {policy_path}")
    policy = ACTPolicy.from_pretrained(policy_path)
    policy.eval()

    # Load test dataset
    print(f"Loading test dataset: {dataset_id}")
    dataset = LeRobotDataset(dataset_id)

    # Evaluation metrics
    total_rewards = []
    success_rates = []

    for episode_idx in range(min(num_episodes, dataset.num_episodes)):
        print(f"Evaluating episode {episode_idx}...")

        # Get episode data
        from_idx = dataset.meta.episodes[episode_idx]["dataset_from_index"]
        to_idx = dataset.meta.episodes[episode_idx]["dataset_to_index"]

        episode_reward = 0
        episode_success = False

        # Simulate episode execution
        for frame_idx in range(from_idx, to_idx):
            frame = dataset[frame_idx]

            # Get observation
            observation = frame["observation"]

            # Get policy action
            with torch.no_grad():
                action = policy.select_action(observation)

            # Simulate reward (replace with actual environment)
            reward = 0.0  # Placeholder
            episode_reward += reward

            # Check for success condition
            if frame_idx == to_idx - 1:  # Last frame
                episode_success = True  # Placeholder

        total_rewards.append(episode_reward)
        success_rates.append(episode_success)

        print(f"  Episode {episode_idx}: Reward={episode_reward:.3f}, Success={episode_success}")

    # Calculate final metrics
    avg_reward = sum(total_rewards) / len(total_rewards)
    success_rate = sum(success_rates) / len(success_rates)

    print(f"\n📊 Evaluation Results:")
    print(f"  Average Reward: {avg_reward:.3f}")
    print(f"  Success Rate: {success_rate:.3f}")
    print(f"  Episodes Evaluated: {len(total_rewards)}")

    return {
        "avg_reward": avg_reward,
        "success_rate": success_rate,
        "episode_rewards": total_rewards,
        "episode_successes": success_rates
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Policy evaluation")
    parser.add_argument("--policy-path", type=str, required=True, help="Path to trained policy")
    parser.add_argument("--dataset", type=str, required=True, help="Test dataset ID")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to evaluate")

    args = parser.parse_args()
    evaluate_policy(args.policy_path, args.dataset, args.num_episodes)
```

### Real Robot Testing

```bash
# Test trained policy on real robot
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/policy_test \
  --dataset.num_episodes=5 \
  --dataset.single_task="Test trained policy" \
  --policy.path=${HF_USER}/act_policy
```

## Advanced Training Techniques

### Multi-GPU Training

```bash
# Train with multiple GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 lerobot-train \
  --dataset.repo_id=user/dataset_name \
  --policy.type=act \
  --output_dir=outputs/train/act_multi_gpu \
  --job_name=act_multi_gpu \
  --batch_size=128 \
  --steps=20000
```

### Data Augmentation

```python
from torchvision import transforms

# Define augmentation pipeline
augmentation_pipeline = transforms.Compose([
    transforms.RandomRotation(degrees=5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.RandomHorizontalFlip(p=0.1)
])

# Apply during training
def augment_observation(observation):
    """Apply augmentation to observation images."""
    if "images" in observation:
        for key, image in observation["images"].items():
            observation["images"][key] = augmentation_pipeline(image)
    return observation
```

### Curriculum Learning

```python
def curriculum_learning_schedule(step, total_steps):
    """Implement curriculum learning for training."""

    # Start with easy episodes, gradually increase difficulty
    if step < total_steps * 0.3:
        # Easy episodes (first 30% of training)
        difficulty = "easy"
    elif step < total_steps * 0.7:
        # Medium episodes (middle 40% of training)
        difficulty = "medium"
    else:
        # Hard episodes (last 30% of training)
        difficulty = "hard"

    return difficulty
```

### Hyperparameter Optimization

```python
import optuna

def objective(trial):
    """Objective function for hyperparameter optimization."""

    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    chunk_size = trial.suggest_categorical("chunk_size", [50, 100, 200])

    # Train model with suggested parameters
    # ... training code ...

    # Return validation loss
    return validation_loss

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)
```

## Troubleshooting

### Common Training Issues

#### Out of Memory Errors

```bash
# Reduce batch size
--batch_size=16

# Use gradient accumulation
--gradient_accumulation_steps=2

# Enable mixed precision
--mixed_precision=true
```

#### Slow Training

```bash
# Increase number of workers
--num_workers=8

# Enable pin memory
--pin_memory=true

# Use persistent workers
--persistent_workers=true
```

#### Convergence Issues

```bash
# Adjust learning rate
--policy.optimizer_lr=5e-6

# Increase training steps
--steps=50000

# Enable learning rate scheduling
--lr_scheduler=true
```

### Debug Mode

```bash
# Enable debug logging
lerobot-train --config=training_config.json --log_level=DEBUG

# Single GPU debugging
CUDA_VISIBLE_DEVICES=0 lerobot-train --config=training_config.json
```

### Checkpoint Recovery

```python
def recover_from_checkpoint(checkpoint_path):
    """Recover training from checkpoint."""

    checkpoint = torch.load(checkpoint_path)

    # Restore model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Restore optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Restore training state
    epoch = checkpoint["epoch"]
    step = checkpoint["step"]

    print(f"Recovered from epoch {epoch}, step {step}")

    return epoch, step
```

## Conclusion

This guide provides comprehensive coverage of LeRobot model training capabilities. Key takeaways:

- Use `lerobot-train` for all policy training
- ACT is the most commonly used policy for manipulation tasks
- Configure training parameters via command line or JSON files
- Monitor training with Weights & Biases integration
- Evaluate trained policies before deployment
- Use advanced techniques for improved performance

For specific policy architectures or training scenarios, refer to the individual documentation pages in the LeRobot documentation.

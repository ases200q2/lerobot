---
noteId: "9b3e5d30d1ba11f08bd0898d85b76daa"
tags: []
---

# LeRobot ACT Training Guide

## Overview

Action Chunking with Transformers (ACT) is a powerful imitation learning algorithm that learns to predict action sequences for robot manipulation tasks. LeRobot provides a comprehensive implementation of ACT with extensive configuration options, multi-GPU training support, and seamless integration with collected demonstration data.

## ACT Architecture

### Core Components

```
Input Sequence
├── Visual Features (ResNet18 Backbone)
├── Robot State (Joint Positions/Velocities)
└── Environment State
    ↓
Feature Fusion
    ↓
VAE Encoder
├── Transformer Encoder (4 layers)
└── Latent Representation (32-dim)
    ↓
Transformer Decoder (1 layer)
    ↓
Action Sequence (100-step chunks)
    ↓
Temporal Ensembling (Optional)
    ↓
Output Actions
```

### Key Architecture Features

- **ResNet18 Backbone**: Pretrained ImageNet weights for visual feature extraction
- **VAE Encoder**: Encodes action sequences into latent representations
- **Transformer Architecture**: Multi-head attention with 4 encoder layers and 1 decoder layer
- **Action Chunking**: Predicts action sequences of 100 timesteps
- **Temporal Ensembling**: Optional exponential weighting for smooth action execution
- **Configurable Dimensions**: Adjustable hidden dimensions (default: 512)

### Model Parameters

```python
@dataclass
class ACTConfig:
    # Architecture
    dim_model: int = 512               # Transformer hidden dimension
    n_heads: int = 8                   # Multi-head attention
    n_encoder_layers: int = 4           # Encoder transformer layers
    n_decoder_layers: int = 1          # Decoder transformer layers
    latent_dim: int = 32                # VAE latent dimension

    # Vision backbone
    vision_backbone: str = "resnet18"   # Visual feature extractor
    pretrained: bool = True             # Use ImageNet pretrained weights

    # Action chunking
    chunk_size: int = 100               # Action prediction chunk size
    n_action_steps: int = 100           # Action steps to execute
    n_obs_steps: int = 1                # Observation context steps

    # Training
    optimizer_lr: float = 1e-5          # Learning rate
    optimizer_weight_decay: float = 1e-4
    dropout: float = 0.1                # Dropout rate

    # VAE parameters
    use_vae: bool = True                # Enable VAE
    kl_weight: float = 10.0             # KL divergence weight

    # Inference
    temporal_ensemble_coeff: float = None  # Temporal ensembling coefficient
```

## Training Pipeline

### Main Training Script

The primary training script is `lerobot-train` with comprehensive configuration options:

```bash
# Basic ACT training
lerobot-train \
    --policy.type=act \
    --policy.device=cuda \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --output_dir=./outputs/act_training \
    --batch_size=8 \
    --steps=100000
```

### Training Workflow

1. **Data Loading**: Loads demonstration dataset with streaming support
2. **Preprocessing**: Normalization, batching, and device placement
3. **Model Forward Pass**: VAE encoding → transformer processing → action prediction
4. **Loss Computation**: L1 action loss + optional KL divergence
5. **Optimization**: AdamW optimizer with configurable parameters
6. **Evaluation**: Periodic evaluation on validation episodes
7. **Checkpointing**: Automatic model saving and HuggingFace integration

### Data Preprocessing

```python
# From processor_act.py
def make_act_pre_post_processors(config, dataset_stats):
    """Create preprocessing pipeline for ACT training"""

    # Input processing steps
    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        DeviceProcessorStep(device=config.device),
        NormalizerProcessorStep(
            features={**config.input_features, **config.output_features},
            norm_map=config.normalization_mapping,
            stats=dataset_stats,
            device=config.device,
        ),
    ]

    # Output processing steps
    output_steps = [
        UnnormalizerProcessorStep(
            features=config.output_features,
            norm_map=config.normalization_mapping,
            stats=dataset_stats
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    return ProcessorPipeline(input_steps), ProcessorPipeline(output_steps)
```

## Configuration System

### Base Configuration Structure

```yaml
# lerobot/configs/policy/act.yaml
policy:
  type: "act"
  device: "cuda"
  dim_model: 512
  n_heads: 8
  n_encoder_layers: 4
  n_decoder_layers: 1
  latent_dim: 32
  chunk_size: 100
  n_action_steps: 100
  n_obs_steps: 1
  use_vae: true
  kl_weight: 10.0
  optimizer_lr: 1e-5
  optimizer_weight_decay: 1e-4
  dropout: 0.1
  temporal_ensemble_coeff: null
```

### Training Configuration

```yaml
# Training pipeline configuration
training:
  batch_size: 8
  steps: 100000
  eval_freq: 20000
  log_freq: 200
  save_freq: 20000
  num_workers: 4
  seed: 1000
  resume: false
  push_to_hub: false
  repo_id: null
  private: false
```

### Dataset Configuration

```yaml
# Dataset configuration
dataset:
  repo_id: "lerobot/aloha_sim_transfer_cube_human"
  episodes: null
  image_transforms:
    enable: false
  batch_size: 8
  num_workers: 4
  pin_memory: true
  persistent_workers: true
  prefetch_factor: 2
```

## Training Commands

### Basic Training

```bash
# Train ACT on ALOHA dataset
lerobot-train \
    --policy.type=act \
    --policy.device=cuda \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --output_dir=./outputs/aloha_act \
    --batch_size=8 \
    --steps=50000 \
    --eval_freq=10000 \
    --log_freq=100
```

### Custom Architecture Training

```bash
# Train with custom architecture parameters
lerobot-train \
    --policy.type=act \
    --policy.dim_model=256 \
    --policy.n_heads=4 \
    --policy.n_encoder_layers=6 \
    --policy.n_decoder_layers=2 \
    --policy.latent_dim=64 \
    --policy.chunk_size=50 \
    --policy.n_action_steps=25 \
    --policy.optimizer_lr=5e-5 \
    --policy.use_vae=true \
    --dataset.repo_id=my_custom_dataset \
    --output_dir=./outputs/custom_architecture \
    --batch_size=16 \
    --steps=200000
```

### Multi-GPU Training

```bash
# Distributed training on multiple GPUs
accelerate launch \
    --multi_gpu \
    --num_machines=1 \
    --machine_rank=0 \
    lerobot-train \
    --policy.type=act \
    --policy.device=cuda \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --output_dir=./outputs/multi_gpu_training \
    --batch_size=32 \
    --steps=100000
```

### Training with Automatic Upload

```bash
# Training with HuggingFace integration
lerobot-train \
    --policy.type=act \
    --policy.device=cuda \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/act_policy \
    --output_dir=./outputs/hub_training \
    --batch_size=16 \
    --steps=100000 \
    --eval_freq=10000 \
    --save_freq=10000
```

### Resume Training

```bash
# Resume from checkpoint
lerobot-train \
    --config_path=./outputs/previous_training/train_config.json \
    --resume=true \
    --steps=200000
```

## Dataset Integration

### Supported Dataset Formats

LeRobot ACT training supports multiple dataset sources:

#### HuggingFace Datasets

```bash
# Train on public HuggingFace dataset
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=lerobot/pusht \
    --output_dir=./outputs/pusht_training \
    --batch_size=8 \
    --steps=50000
```

#### Local Datasets

```bash
# Train on local dataset directory
lerobot-train \
    --policy.type=act \
    --dataset.root=./local_datasets/my_data \
    --output_dir=./outputs/local_training \
    --batch_size=8 \
    --steps=50000
```

#### Combined Datasets

```bash
# Train on multiple datasets
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    --dataset.repo_id_2=${HF_USER}/my_additional_data \
    --dataset.weight_1=0.7 \
    --dataset.weight_2=0.3 \
    --output_dir=./outputs/multi_dataset_training \
    --batch_size=8 \
    --steps=100000
```

### Feature Mapping

Customize feature mapping for your dataset:

```python
# Custom feature mapping
features_map = {
    "observation.images.front": "observation.images.front",
    "observation.images.wrist": "observation.images.wrist",
    "observation.state": "observation.state",
    "action": "action"
}

# In training configuration
--dataset.features_map='{"observation.images.front": "observation.images.front", "observation.state": "observation.state", "action": "action"}'
```

## Simulation vs Real-World Training

### Simulation Training

#### Advantages

- Unlimited data generation
- Perfect state information
- Safe failure modes
- Parallel environment support

#### Setup

```bash
# Train in simulation environment
lerobot-train \
    --policy.type=act \
    --env.type=gym_hil \
    --env.task=PandaPickCubeSpacemouse-v0 \
    --dataset.repo_id=simulation_data \
    --output_dir=./outputs/sim_training \
    --batch_size=32 \
    --steps=200000
```

#### Simulation Configuration

```yaml
# Gym-HIL environment configuration
env:
  type: "gym_hil"
  task: "PandaPickCubeSpacemouse-v0"
  wrapper:
    control_mode: "spacemouse"
    use_gripper: true
    resize_size: [128, 128]
    add_joint_velocity_to_observation: true
    gripper_penalty: -0.02
    display_cameras: false
    control_time_s: 150.0
    reset_time_s: 150.0
    fixed_reset_joint_positions: [0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0]
```

### Real-World Training

#### Advantages

- Real physics and dynamics
- Real sensor noise and artifacts
- Direct applicability
- No sim-to-real gap

#### Setup

```bash
# Train on real-world data
lerobot-train \
    --policy.type=act \
    --env.type=aloha \
    --dataset.repo_id=${HF_USER}/real_world_data \
    --env.config_path=configs/robot/aloha.yaml \
    --output_dir=./outputs/real_training \
    --batch_size=8 \
    --steps=50000
```

#### Real-World Configuration

```yaml
# ALOHA robot configuration
env:
  type: "aloha"
  config_path: "configs/robot/aloha.yaml"
  calibration_dir: "calibrations/"
  use_async_io: true
  control_frequency: 200
  observation_frequency: 30
```

### Domain Randomization

For bridging the sim-to-real gap:

```yaml
# Domain randomization configuration
domain_randomization:
  enabled: true

  # Visual randomization
  random_brightness: 0.3
  random_contrast: 0.3
  random_saturation: 0.2
  random_hue: 0.1
  gaussian_noise: 0.02

  # Physics randomization
  random_friction: 0.2
  random_mass: 0.1
  random_damping: 0.1

  # Camera randomization
  random_camera_position: 0.05
  random_camera_rotation: 0.1
  random_fov: 0.05
```

## Advanced Training Features

### Curriculum Learning

```bash
# Progressive training strategy
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=curriculum_data \
    --curriculum.enabled=true \
    --curriculum.stages=3 \
    --curriculum.stage_1.dataset="easy_tasks" \
    --curriculum.stage_1.steps=20000 \
    --curriculum.stage_2.dataset="medium_tasks" \
    --curriculum.stage_2.steps=30000 \
    --curriculum.stage_3.dataset="hard_tasks" \
    --curriculum.stage_3.steps=50000
```

### Data Augmentation

```yaml
# Data augmentation configuration
image_transforms:
  enabled: true

  # Basic transforms
  resize: [224, 224]
  normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]

  # Augmentations
  random_horizontal_flip: 0.5
  random_rotation: 10
  color_jitter:
    brightness: 0.2
    contrast: 0.2
    saturation: 0.1
    hue: 0.05

  # Advanced
  random_affine:
    degrees: 10
    translate: [0.1, 0.1]
    scale: [0.9, 1.1]
  random_erasing: 0.1
```

### Mixed Precision Training

```bash
# Enable mixed precision for faster training
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=my_dataset \
    --training.mixed_precision=true \
    --training.fp16_opt_level=O1 \
    --output_dir=./outputs/mixed_precision_training \
    --batch_size=16 \
    --steps=100000
```

### Gradient Accumulation

```bash
# Effective large batch size with limited memory
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=my_dataset \
    --training.gradient_accumulation_steps=8 \
    --batch_size=4 \
    --effective_batch_size=32 \
    --output_dir=./outputs/gradient_accumulation_training \
    --steps=100000
```

## Evaluation and Testing

### Training Evaluation

```bash
# Evaluate during training
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=my_dataset \
    --eval.enabled=true \
    --eval.freq=5000 \
    --eval.num_episodes=10 \
    --eval.render=false \
    --output_dir=./outputs/eval_training
```

### Standalone Evaluation

```bash
# Evaluate trained policy
lerobot-eval \
    --policy.path=${HF_USER}/act_policy \
    --env.type=aloha \
    --num_episodes=50 \
    --render=true \
    --output_dir=./evaluation_results
```

### Simulation Evaluation

```bash
# Evaluate in simulation
python EVAL/evaluate_act_simulation.py \
    --checkpoint=./outputs/checkpoint.pth \
    --num_episodes=20 \
    --render=true \
    --env_task=PandaPickCubeSpacemouse-v0
```

### Metrics and Logging

```yaml
# Evaluation metrics
evaluation:
  metrics:
    - success_rate
    - average_reward
    - episode_length
    - action_smoothness
    - goal_distance

  logging:
    wandb_project: "act_evaluation"
    tensorboard: true
    save_videos: true
    save_trajectories: true
```

## Model Optimization

### Hyperparameter Tuning

```bash
# Grid search example
for lr in 1e-5 5e-5 1e-4; do
  for chunk_size in 50 100 200; do
    lerobot-train \
        --policy.type=act \
        --policy.optimizer_lr=$lr \
        --policy.chunk_size=$chunk_size \
        --dataset.repo_id=tuning_data \
        --output_dir=./outputs/tuning_lr_${lr}_chunk_${chunk_size} \
        --batch_size=8 \
        --steps=50000
  done
done
```

### Model Distillation

```bash
# Distill large teacher model to small student model
lerobot-distill \
    --teacher_policy.path=./large_act_model \
    --student_policy.type=act \
    --student_policy.dim_model=256 \
    --student_policy.n_heads=4 \
    --dataset.repo_id=distillation_data \
    --output_dir=./outputs/distilled_model \
    --batch_size=16 \
    --steps=50000
```

### Model Pruning

```python
# Post-training pruning
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.optimization.pruning import prune_model

# Load trained model
policy = ACTPolicy.from_pretrained("path/to/model")

# Apply pruning
pruned_policy = prune_model(
    policy,
    sparsity=0.5,           # 50% pruning
    method="magnitude",      # Magnitude-based pruning
    structured=True          # Structured pruning
)

# Save pruned model
pruned_policy.save_pretrained("path/to/pruned_model")
```

## Troubleshooting

### Common Training Issues

#### Memory Issues

```bash
# Reduce memory usage
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=my_dataset \
    --batch_size=4 \                    # Reduce batch size
    --training.gradient_accumulation_steps=8  # Use gradient accumulation
    --training.pin_memory=false \       # Disable pin memory
    --training.num_workers=2 \          # Reduce workers
    --output_dir=./outputs/low_memory_training
```

#### Convergence Problems

```yaml
# Learning rate scheduling
training:
  scheduler: "cosine"
  warmup_steps: 1000
  min_lr: 1e-6

# Regularization
policy:
  dropout: 0.2
  optimizer_weight_decay: 1e-3
  gradient_clip_norm: 1.0
```

#### Overfitting

```yaml
# Data augmentation and regularization
dataset:
  image_transforms:
    enabled: true
    random_horizontal_flip: 0.5
    random_rotation: 15
    color_jitter: 0.2

policy:
  dropout: 0.3
  optimizer_weight_decay: 1e-3
```

### Performance Optimization

#### Fast Training Setup

```bash
# Optimize for speed
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=my_dataset \
    --training.num_workers=8 \          # More workers
    --training.prefetch_factor=4 \      # Higher prefetch
    --training.pin_memory=true \        # Enable pin memory
    --training.mixed_precision=true \   # Mixed precision
    --batch_size=32 \                   # Larger batch
    --training.compile_model=true       # Model compilation
```

#### Large Dataset Handling

```bash
# Handle large datasets efficiently
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=large_dataset \
    --dataset.streaming=true \           # Stream data
    --dataset.cache_dir=/tmp/dataset_cache \  # Local cache
    --dataset.max_size_gb=100 \         # Limit cache size
    --batch_size=8 \
    --steps=100000
```

## Best Practices

### Data Collection for ACT

1. **High-Quality Demonstrations**: Ensure teleoperated demonstrations are smooth and successful
2. **Diverse Initial States**: Include varied starting positions and object configurations
3. **Consistent Episode Structure**: Start and end episodes in defined states
4. **Sufficient Data**: Collect at least 100-500 episodes for complex tasks
5. **Multi-View Cameras**: Use multiple camera angles for better spatial understanding

### Training Best Practices

1. **Start Simple**: Begin with default hyperparameters and adjust gradually
2. **Monitor Metrics**: Track training loss, validation performance, and success rates
3. **Use Checkpoints**: Save intermediate models for potential recovery
4. **Experiment Systematically**: Change one parameter at a time
5. **Validate Frequently**: Regular evaluation prevents wasted training time

### Model Deployment

1. **Quantization**: Use INT8 quantization for faster inference
2. **Model Compilation**: Compile models for better performance
3. **Batch Inference**: Process multiple timesteps when possible
4. **Optimization Profiles**: Create TensorRT optimization profiles for deployment
5. **Edge Deployment**: Consider model size and compute constraints

This comprehensive guide covers all aspects of ACT training in LeRobot, from basic setup to advanced optimization and deployment strategies.

---
noteId: "a62cf170afb511f08bd0898d85b76daa"
tags: []
---

# ACT Policy Evaluation Guide for Gym-HIL Environment

## Table of Contents

1. [Understanding ACT Policy Architecture](#understanding-act-policy-architecture)
2. [Training and Evaluation Workflow](#training-and-evaluation-workflow)
3. [Gym-HIL Environment Setup](#gym-hil-environment-setup)
4. [Step-by-Step Evaluation Process](#step-by-step-evaluation-process)
5. [Evaluation Metrics and Interpretation](#evaluation-metrics-and-interpretation)
6. [Practical Examples and Commands](#practical-examples-and-commands)
7. [Advanced Evaluation Techniques](#advanced-evaluation-techniques)
8. [Troubleshooting and Best Practices](#troubleshooting-and-best-practices)
9. [Performance Analysis](#performance-analysis)
10. [Real-World Considerations](#real-world-considerations)

## Understanding ACT Policy Architecture

### What is ACT (Action Chunking with Transformers)?

ACT is a transformer-based policy that predicts sequences of actions (chunks) rather than single actions. This approach enables more efficient and stable execution by:

- **Action Chunking**: Predicts `chunk_size` actions at once (default: 100)
- **Transformer Architecture**: Uses encoder-decoder transformers for sequence modeling
- **Vision Processing**: ResNet backbone for image feature extraction
- **Optional VAE**: Variational Autoencoder for better action modeling

### Key Components

#### 1. Neural Network Architecture

```python
# ACT Policy Structure
ACTPolicy
├── ACT (neural network)
│   ├── ResNet-18 backbone (vision processing)
│   ├── ACTEncoder (transformer encoder)
│   ├── ACTDecoder (transformer decoder)
│   └── Optional VAE encoder (for training)
├── ACTTemporalEnsembler (action smoothing)
└── Preprocessor/Postprocessor (data transformation)
```

#### 2. Action Chunking Mechanism

- **Chunk Size**: 100 actions predicted per forward pass
- **Action Steps**: Number of actions to execute from chunk (default: 100)
- **Temporal Ensembling**: Optional smoothing for continuous execution

#### 3. Configuration Parameters

```python
# Key ACT Configuration
ACTConfig(
    # Architecture
    dim_model=512,              # Transformer hidden dimension
    n_heads=8,                  # Attention heads
    n_encoder_layers=4,         # Encoder layers
    n_decoder_layers=1,         # Decoder layers

    # Action Chunking
    chunk_size=100,             # Actions predicted per forward pass
    n_action_steps=100,         # Actions executed from chunk
    temporal_ensemble_coeff=None, # Smoothing coefficient

    # VAE (optional)
    use_vae=True,               # Enable VAE training
    latent_dim=32,             # Latent space dimension
    kl_weight=10.0,            # KL divergence weight

    # Training
    optimizer_lr=1e-5,         # Learning rate
    dropout=0.1,               # Dropout rate
)
```

## Training and Evaluation Workflow

### 1. Training Process Overview

```
Training Pipeline:
Dataset Loading → Preprocessing → Policy Forward → Loss Computation →
Gradient Update → Checkpointing → Evaluation
```

#### Training Loss Components

```python
# ACT Loss Function
def compute_loss(actions_pred, actions_target, mu=None, log_sigma=None):
    # L1 Reconstruction Loss
    l1_loss = F.l1_loss(actions_pred, actions_target)

    # Optional VAE KL Divergence
    if use_vae:
        kld_loss = (-0.5 * (1 + log_sigma - mu**2 - log_sigma.exp())).sum(-1).mean()
        total_loss = l1_loss + kl_weight * kld_loss
    else:
        total_loss = l1_loss

    return total_loss
```

#### Training Command Example

```bash
# Train ACT policy on gym-hil environment
lerobot-train \
    --policy.type=act \
    --env.type=gym_manipulator \
    --dataset.repo_id=your-dataset-id \
    --policy.dim_model=512 \
    --policy.chunk_size=100 \
    --policy.use_vae=True \
    --output-dir=outputs/train/gym_hil_act
```

### 2. Evaluation Process Overview

```
Evaluation Pipeline:
Policy Loading → Environment Setup → Rollout Execution →
Metrics Calculation → Video Generation → Results Analysis
```

#### Key Evaluation Components

- **Vectorized Environments**: Parallel episode execution
- **Action Processing**: Policy → Preprocessor → Postprocessor → Environment
- **Metrics Collection**: Rewards, success rates, timing
- **Video Generation**: Episode visualization for analysis

## Gym-HIL Environment Setup

### Environment Configuration

The gym-hil environment provides hardware-in-the-loop simulation with human intervention capabilities.

#### Basic Environment Config

```yaml
# gym_hil_config.yaml
env:
  type: gym_manipulator
  name: real_robot

  # Processor Configuration
  processor:
    control_mode: gamepad # Human input mode
    max_gripper_pos: 30.0
    reset:
      reset_time_s: 5.0
      control_time_s: 20.0

    # Gripper Configuration
    gripper:
      use_gripper: true
      gripper_penalty: -0.05
```

#### Available Wrappers

```python
# gym-hil environment wrappers
1. GripperPenaltyWrapper     # Penalizes inefficient gripper usage
2. EEActionWrapper           # End-effector space processing
3. InputsControlWrapper      # Human intervention support
4. SpaceMouseWrapper        # Alternative input method
```

### Environment Factory

```python
# Creating gym-hil environments
envs = make_env(
    cfg=env_config,
    n_envs=10,                    # Number of parallel environments
    use_async_envs=True,          # Async execution for efficiency
)
```

## Step-by-Step Evaluation Process

### Phase 1: Policy Loading and Setup

```python
# Load trained ACT policy
policy = make_policy(
    cfg=policy_config,
    env_cfg=env_config,
    dataset_metadata=dataset_meta
)

# Create processors for data transformation
preprocessor, postprocessor = make_pre_post_processors(
    policy_config=policy_config,
    env_cfg=env_config,
    dataset_meta=dataset_meta
)

# Load trained weights
policy.load_state_dict(checkpoint["state_dict"])
policy.eval()  # Set to evaluation mode
```

### Phase 2: Environment Initialization

```python
# Create vectorized environments
envs = make_env(
    cfg=env_config,
    n_envs=batch_size,
    use_async_envs=True
)

# Reset environments
observations, infos = envs.reset(seed=evaluation_seed)

# Preprocess observations for policy
batch = preprocessor(observations)
```

### Phase 3: Rollout Execution

```python
# Execute evaluation rollouts
for episode in range(n_episodes):
    done = False
    episode_rewards = []

    while not done:
        # Predict action chunk
        with torch.no_grad():
            actions_chunk = policy.predict_action_chunk(batch)

        # Select next action
        action = actions_chunk[:, :n_action_steps]

        # Postprocess action for environment
        env_action = postprocessor(action)

        # Execute action in environment
        observations, rewards, terminateds, truncateds, infos = envs.step(env_action)

        # Update batch with new observations
        batch = preprocessor(observations)

        # Check episode completion
        done = terminateds | truncateds

        episode_rewards.append(rewards)
```

### Phase 4: Metrics Collection

```python
# Calculate evaluation metrics
def calculate_metrics(all_episodes):
    sum_rewards = [sum(episode_rewards) for episode_rewards in all_episodes]
    max_rewards = [max(episode_rewards) for episode_rewards in all_episodes]
    successes = [episode['success'] for episode in all_episodes]

    metrics = {
        "avg_sum_reward": np.mean(sum_rewards),
        "avg_max_reward": np.mean(max_rewards),
        "pc_success": np.mean(successes) * 100,
        "n_episodes": len(all_episodes)
    }

    return metrics
```

## Evaluation Metrics and Interpretation

### Primary Metrics

#### 1. Success Rate (pc_success)

```python
success_rate = (successful_episodes / total_episodes) * 100
```

- **What it measures**: Percentage of episodes that achieve the task goal
- **Interpretation**:
  - < 50%: Poor performance, needs significant improvement
  - 50-75%: Moderate performance, promising but needs refinement
  - 75-90%: Good performance, ready for deployment
  - > 90%: Excellent performance, production-ready

#### 2. Average Sum Reward (avg_sum_reward)

```python
avg_sum_reward = mean(sum(rewards_per_episode))
```

- **What it measures**: Total accumulated reward across all timesteps
- **Interpretation**:
  - < 0.3: Policy not learning effectively
  - 0.3-0.6: Basic learning achieved
  - 0.6-0.8: Good learning progress
  - > 0.8: Strong task performance

#### 3. Average Max Reward (avg_max_reward)

```python
avg_max_reward = mean(max(rewards_per_episode))
```

- **What it measures**: Peak reward achieved during episodes
- **Interpretation**: Indicates the best-case performance potential

### Secondary Metrics

#### 1. Evaluation Time

```python
eval_ep_s = total_evaluation_time / number_of_episodes
```

- **What it measures**: Average time per episode
- **Interpretation**: Performance and efficiency indicator

#### 2. Per-Episode Breakdown

```json
{
  "per_episode": [
    {
      "episode_ix": 0,
      "sum_reward": 0.85,
      "max_reward": 0.95,
      "success": true,
      "seed": 42
    }
  ]
}
```

### Metrics Analysis Framework

```python
def analyze_performance(metrics):
    """Comprehensive performance analysis"""

    # Success Rate Analysis
    if metrics['pc_success'] > 90:
        success_level = "Excellent"
    elif metrics['pc_success'] > 75:
        success_level = "Good"
    elif metrics['pc_success'] > 50:
        success_level = "Moderate"
    else:
        success_level = "Needs Improvement"

    # Reward Analysis
    reward_level = "High" if metrics['avg_sum_reward'] > 0.8 else "Medium" if metrics['avg_sum_reward'] > 0.5 else "Low"

    # Overall Assessment
    assessment = {
        "success_level": success_level,
        "reward_level": reward_level,
        "ready_for_deployment": metrics['pc_success'] > 75 and metrics['avg_sum_reward'] > 0.7
    }

    return assessment
```

## Practical Examples and Commands

### Basic Evaluation Commands

#### 1. Simple Evaluation

```bash
# Evaluate trained ACT policy
lerobot-eval \
    --policy.path=outputs/train/gym_hil_act/policy_model \
    --env.type=gym_manipulator \
    --eval.n_episodes=10 \
    --eval.batch_size=10 \
    --output-dir=outputs/eval/basic_test
```

#### 2. Evaluation with Video Generation

```bash
# Generate videos for analysis
lerobot-eval \
    --policy.path=outputs/train/gym_hil_act/policy_model \
    --env.type=gym_manipulator \
    --eval.n_episodes=20 \
    --eval.batch_size=10 \
    --output-dir=outputs/eval/video_test \
    --env.max_episodes_rendered=20
```

#### 3. Reproducible Evaluation

```bash
# Fixed seed for reproducible results
lerobot-eval \
    --policy.path=outputs/train/gym_hil_act/policy_model \
    --env.type=gym_manipulator \
    --eval.n_episodes=50 \
    --eval.batch_size=25 \
    --seed=1000 \
    --output-dir=outputs/eval/reproducible_test
```

### Advanced Evaluation Examples

#### 1. Multi-Task Evaluation

```bash
# Evaluate on multiple tasks/configurations
for task in "pick_place" "push" "reach"; do
    lerobot-eval \
        --policy.path=outputs/train/gym_hil_act/policy_model \
        --env.type=gym_manipulator \
        --env.task=${task} \
        --eval.n_episodes=30 \
        --eval.batch_size=15 \
        --output-dir=outputs/eval/${task}_test
done
```

#### 2. Performance Benchmarking

```bash
# Comprehensive benchmarking
lerobot-eval \
    --policy.path=outputs/train/gym_hil_act/policy_model \
    --env.type=gym_manipulator \
    --eval.n_episodes=100 \
    --eval.batch_size=50 \
    --eval.use_async_envs=true \
    --output-dir=outputs/eval/benchmark \
    --policy.device=cuda
```

### Custom Evaluation Script

```python
#!/usr/bin/env python3
"""
Custom evaluation script for ACT policy in gym-hil
"""

import json
import numpy as np
from pathlib import Path
from lerobot.scripts.lerobot_eval import eval_policy_all
from lerobot.common.factory import make_env, make_policy

def evaluate_act_policy(policy_path, config_overrides=None):
    """Comprehensive ACT policy evaluation"""

    # Base configuration
    base_config = {
        "policy": {
            "path": policy_path,
            "type": "act"
        },
        "env": {
            "type": "gym_manipulator",
        },
        "eval": {
            "n_episodes": 50,
            "batch_size": 25,
            "use_async_envs": True
        }
    }

    # Apply overrides
    if config_overrides:
        base_config.update(config_overrides)

    # Create components
    envs = make_env(base_config["env"], n_envs=base_config["eval"]["batch_size"])
    policy = make_policy(cfg=base_config["policy"])

    # Run evaluation
    metrics = eval_policy_all(
        envs=envs,
        policy=policy,
        n_episodes=base_config["eval"]["n_episodes"],
        output_dir=Path("outputs/eval/custom_test")
    )

    # Analyze results
    analysis = analyze_performance(metrics["overall"])

    # Save comprehensive results
    results = {
        "metrics": metrics,
        "analysis": analysis,
        "config": base_config
    }

    with open("outputs/eval/custom_test/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation Results:")
    print(f"Success Rate: {metrics['overall']['pc_success']:.1f}%")
    print(f"Avg Reward: {metrics['overall']['avg_sum_reward']:.3f}")
    print(f"Performance Level: {analysis['success_level']}")

    return results

if __name__ == "__main__":
    # Run evaluation
    results = evaluate_act_policy(
        policy_path="outputs/train/gym_hil_act/policy_model",
        config_overrides={
            "eval": {"n_episodes": 100, "batch_size": 50}
        }
    )
```

## Advanced Evaluation Techniques

### 1. Temporal Analysis

#### Action Chunk Analysis

```python
def analyze_action_chunks(policy, observation_sample):
    """Analyze how action chunks are predicted and used"""

    policy.eval()
    with torch.no_grad():
        # Predict action chunk
        actions_chunk = policy.predict_action_chunk(observation_sample)

        # Analyze chunk properties
        chunk_stats = {
            "chunk_shape": actions_chunk.shape,
            "action_variance": torch.var(actions_chunk, dim=1).mean().item(),
            "action_magnitude": torch.abs(actions_chunk).mean().item(),
            "action_smoothness": calculate_smoothness(actions_chunk)
        }

    return chunk_stats

def calculate_smoothness(actions):
    """Calculate action smoothness across chunk"""
    diff = torch.diff(actions, dim=1)
    smoothness = torch.mean(torch.norm(diff, dim=-1))
    return smoothness.item()
```

#### Temporal Ensembling Analysis

```python
def evaluate_temporal_ensembling(policy, env, n_episodes=10):
    """Compare performance with and without temporal ensembling"""

    # Test without temporal ensembling
    policy.config.temporal_ensemble_coeff = None
    metrics_no_ensemble = run_evaluation(policy, env, n_episodes)

    # Test with temporal ensembling
    policy.config.temporal_ensemble_coeff = 0.01
    metrics_with_ensemble = run_evaluation(policy, env, n_episodes)

    comparison = {
        "no_ensemble": metrics_no_ensemble,
        "with_ensemble": metrics_with_ensemble,
        "improvement": {
            "success_rate": metrics_with_ensemble["pc_success"] - metrics_no_ensemble["pc_success"],
            "reward": metrics_with_ensemble["avg_sum_reward"] - metrics_no_ensemble["avg_sum_reward"]
        }
    }

    return comparison
```

### 2. Robustness Testing

#### Noise Robustness

```python
def test_noise_robustness(policy, env, noise_levels=[0.0, 0.01, 0.05, 0.1]):
    """Test policy robustness to observation noise"""

    results = {}

    for noise_level in noise_levels:
        def add_noise(obs):
            if isinstance(obs, dict):
                return {k: v + torch.randn_like(v) * noise_level for k, v in obs.items()}
            else:
                return obs + torch.randn_like(obs) * noise_level

        # Create noisy environment wrapper
        noisy_env = gym.wrappers.TransformObservation(env, add_noise)

        # Run evaluation
        metrics = run_evaluation(policy, noisy_env, n_episodes=20)
        results[f"noise_{noise_level}"] = metrics

    return results
```

#### Domain Randomization

```python
def test_domain_randomization(policy, randomization_configs):
    """Test policy performance under domain randomization"""

    results = {}

    for config_name, randomization_params in randomization_configs.items():
        # Create randomized environment
        env = make_randomized_env(randomization_params)

        # Run evaluation
        metrics = run_evaluation(policy, env, n_episodes=30)
        results[config_name] = metrics

    return results

randomization_configs = {
    "lighting_variation": {"lighting": {"range": [0.7, 1.3]}},
    "camera_perturbation": {"camera": {"position_noise": 0.02}},
    "object_variation": {"objects": {"size_variation": 0.1, "position_noise": 0.01}}
}
```

### 3. Ablation Studies

#### Component Ablation

```python
def ablate_vae_component(policy_path):
    """Evaluate policy with and without VAE component"""

    # Load policy with VAE
    policy_with_vae = load_policy(policy_path)
    policy_with_vae.config.use_vae = True

    # Create policy without VAE (disabling VAE during inference)
    policy_without_vae = load_policy(policy_path)
    policy_without_vae.config.use_vae = False

    # Compare performance
    results = {
        "with_vae": run_evaluation(policy_with_vae, env, n_episodes=50),
        "without_vae": run_evaluation(policy_without_vae, env, n_episodes=50)
    }

    return results
```

#### Architecture Ablation

```python
def test_different_chunk_sizes(policy_path, chunk_sizes=[25, 50, 100, 200]):
    """Test performance with different action chunk sizes"""

    results = {}

    for chunk_size in chunk_sizes:
        # Load policy and modify chunk size
        policy = load_policy(policy_path)
        policy.config.chunk_size = chunk_size
        policy.config.n_action_steps = min(chunk_size, 100)

        # Run evaluation
        metrics = run_evaluation(policy, env, n_episodes=30)
        results[f"chunk_size_{chunk_size}"] = metrics

    return results
```

## Troubleshooting and Best Practices

### Common Issues and Solutions

#### 1. Low Success Rate (< 50%)

**Symptoms**: Policy fails to complete task consistently
**Potential Causes**:

- Insufficient training data or variety
- Poor training hyperparameters
- Environment configuration mismatch
- Policy capacity issues

**Solutions**:

```bash
# Check training data quality
lerobot-dataset-viz --repo-id your-dataset-id --episode-index 0

# Re-train with adjusted hyperparameters
lerobot-train \
    --policy.type=act \
    --policy.optimizer_lr=5e-5 \  # Adjust learning rate
    --policy.dim_model=768 \      # Increase model capacity
    --policy.kl_weight=5.0 \      # Reduce KL weight
    # ... other parameters
```

#### 2. High Action Variability

**Symptoms**: Actions are jerky or inconsistent
**Solution**: Enable temporal ensembling

```python
# Add to policy config
policy.config.temporal_ensemble_coeff = 0.01
```

#### 3. Slow Evaluation

**Symptoms**: Evaluation takes too long
**Solution**: Optimize batch configuration

```bash
lerobot-eval \
    --eval.batch_size=50 \        # Match with n_episodes
    --eval.use_async_envs=true \  # Enable async execution
    --policy.device=cuda          # Use GPU acceleration
```

#### 4. Memory Issues

**Symptoms**: Out of memory errors during evaluation
**Solution**: Reduce batch size and enable gradient checkpointing

```python
# Reduce batch size
--eval.batch_size=10

# Enable gradient checkpointing (if available)
policy.config.gradient_checkpointing = True
```

### Best Practices

#### 1. Evaluation Setup

```python
# Recommended evaluation configuration
eval_config = {
    "n_episodes": 50,          # Minimum for statistical significance
    "batch_size": 25,          # Balance efficiency and memory
    "use_async_envs": True,    # Parallel execution
    "seed": 1000,             # Reproducible results
    "max_episodes_rendered": 10  # Limit video generation
}
```

#### 2. Performance Monitoring

```python
def monitor_evaluation_progress(current_episode, total_episodes, start_time):
    """Monitor evaluation progress and performance"""

    progress = current_episode / total_episodes
    elapsed = time.time() - start_time
    estimated_total = elapsed / progress
    remaining = estimated_total - elapsed

    print(f"Progress: {progress:.1%} | Episodes: {current_episode}/{total_episodes} | "
          f"ETA: {remaining/60:.1f}min")
```

#### 3. Result Validation

```python
def validate_evaluation_results(metrics):
    """Validate evaluation results for consistency"""

    # Check for missing data
    if metrics['n_episodes'] == 0:
        raise ValueError("No episodes evaluated")

    # Check for reasonable values
    if metrics['pc_success'] < 0 or metrics['pc_success'] > 100:
        raise ValueError("Invalid success rate")

    # Check for NaN values
    if np.isnan(metrics['avg_sum_reward']):
        raise ValueError("NaN reward detected")

    return True
```

## Performance Analysis

### Performance Benchmarking Framework

```python
class PerformanceBenchmark:
    """Comprehensive performance benchmarking for ACT policies"""

    def __init__(self, policy_path, env_config):
        self.policy_path = policy_path
        self.env_config = env_config
        self.results = {}

    def run_full_benchmark(self):
        """Run comprehensive performance benchmark"""

        # 1. Baseline Performance
        self.results["baseline"] = self.run_baseline_evaluation()

        # 2. Robustness Tests
        self.results["robustness"] = self.run_robustness_tests()

        # 3. Ablation Studies
        self.results["ablations"] = self.run_ablation_studies()

        # 4. Temporal Analysis
        self.results["temporal"] = self.run_temporal_analysis()

        # 5. Efficiency Metrics
        self.results["efficiency"] = self.run_efficiency_analysis()

        return self.generate_report()

    def run_baseline_evaluation(self):
        """Baseline performance evaluation"""
        return run_evaluation(
            self.policy_path,
            self.env_config,
            n_episodes=100,
            seed=1000
        )

    def run_efficiency_analysis(self):
        """Analyze computational efficiency"""
        import time
        import torch.profiler

        # Measure inference time
        policy = load_policy(self.policy_path)
        policy.eval()

        observation = get_sample_observation(self.env_config)

        with torch.no_grad():
            # Time measurement
            start_time = time.time()
            for _ in range(100):
                action = policy.select_action(observation)
            avg_inference_time = (time.time() - start_time) / 100

            # Memory usage
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
                _ = policy.select_action(observation)
                peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB

        return {
            "avg_inference_time_ms": avg_inference_time * 1000,
            "peak_memory_mb": peak_memory if torch.cuda.is_available() else None,
            "model_params": sum(p.numel() for p in policy.parameters()),
            "model_size_mb": sum(p.numel() * p.element_size() for p in policy.parameters()) / 1024**2
        }

    def generate_report(self):
        """Generate comprehensive performance report"""

        report = {
            "summary": {
                "overall_performance": self.results["baseline"]["overall"],
                "robustness_score": self.calculate_robustness_score(),
                "efficiency_metrics": self.results["efficiency"]
            },
            "detailed_results": self.results,
            "recommendations": self.generate_recommendations()
        }

        return report
```

### Performance Optimization

#### 1. Memory Optimization

```python
# Enable memory-efficient evaluation
policy.config.gradient_checkpointing = True
policy.config.use_flash_attention = True  # If available
```

#### 2. Inference Optimization

```python
# Compile model for faster inference (PyTorch 2.0+)
if hasattr(torch, 'compile'):
    policy = torch.compile(policy)
```

#### 3. Batch Optimization

```python
# Optimal batch size calculation
def calculate_optimal_batch_size(env, policy):
    """Calculate optimal batch size based on available memory"""

    # Start with small batch and increase until memory limit
    batch_size = 1
    max_batch_size = 100

    while batch_size <= max_batch_size:
        try:
            # Test with current batch size
            test_envs = make_env(env, n_envs=batch_size)
            run_evaluation(policy, test_envs, n_episodes=1)
            batch_size *= 2
        except RuntimeError as e:
            if "out of memory" in str(e):
                return batch_size // 2
            else:
                raise e

    return batch_size
```

## Real-World Considerations

### Safety and Reliability

#### 1. Safety Constraints

```python
class SafetyWrapper(gym.Wrapper):
    """Safety wrapper for real-world robot evaluation"""

    def __init__(self, env, safety_config):
        super().__init__(env)
        self.safety_config = safety_config
        self.emergency_stop = False

    def step(self, action):
        """Apply safety constraints before executing action"""

        # Check action bounds
        if not self.is_safe_action(action):
            self.emergency_stop = True
            return self.get_safe_termination()

        # Execute action with monitoring
        obs, reward, done, info = self.env.step(action)

        # Check for unsafe states
        if not self.is_safe_state(obs):
            self.emergency_stop = True
            return self.get_safe_termination()

        return obs, reward, done, info

    def is_safe_action(self, action):
        """Check if action is within safe bounds"""
        # Implement safety checks here
        return True
```

#### 2. Intervention Handling

```python
def evaluate_with_intervention_support(policy, env, intervention_callback=None):
    """Evaluation with human intervention support"""

    intervention_count = 0
    successful_interventions = 0

    for episode in range(n_episodes):
        episode_interventions = 0

        while not done:
            # Predict action
            action = policy.select_action(observation)

            # Check for intervention
            if intervention_callback and intervention_callback(observation, action):
                action = get_intervention_action()
                episode_interventions += 1

            # Execute action
            observation, reward, done, info = env.step(action)

            # Record intervention success
            if episode_interventions > 0 and reward > 0:
                successful_interventions += 1

        intervention_count += episode_interventions

    metrics["intervention_rate"] = intervention_count / (n_episodes * episode_length)
    metrics["intervention_success_rate"] = successful_interventions / max(1, intervention_count)

    return metrics
```

### Environmental Factors

#### 1. Lighting and Vision Conditions

```python
def test_lighting_conditions(policy, lighting_configs):
    """Test policy under different lighting conditions"""

    results = {}

    for lighting_name, lighting_params in lighting_configs.items():
        # Create environment with specific lighting
        env = make_env_with_lighting(lighting_params)

        # Run evaluation
        metrics = run_evaluation(policy, env, n_episodes=20)
        results[lighting_name] = metrics

    return results

lighting_configs = {
    "bright": {"brightness": 1.2, "contrast": 1.1},
    "dim": {"brightness": 0.7, "contrast": 0.9},
    "variable": {"brightness_range": [0.6, 1.3]}
}
```

#### 2. Hardware Variations

```python
def test_hardware_variations(policy, hardware_configs):
    """Test policy with different hardware configurations"""

    results = {}

    for config_name, hardware_params in hardware_configs.items():
        # Create environment with specific hardware
        env = make_env_with_hardware(hardware_params)

        # Run evaluation
        metrics = run_evaluation(policy, env, n_episodes=15)
        results[config_name] = metrics

    return results
```

### Deployment Checklist

#### 1. Pre-Deployment Validation

```python
def deployment_readiness_check(policy, env_config):
    """Comprehensive deployment readiness checklist"""

    checklist = {
        "performance_validation": False,
        "safety_checks": False,
        "robustness_validation": False,
        "efficiency_validation": False,
        "documentation_complete": False
    }

    # Performance validation
    baseline_metrics = run_evaluation(policy, env_config, n_episodes=100)
    if baseline_metrics["overall"]["pc_success"] > 75:
        checklist["performance_validation"] = True

    # Safety checks
    if safety_wrapper_tests_pass():
        checklist["safety_checks"] = True

    # Robustness validation
    robustness_results = test_noise_robustness(policy, make_env(env_config))
    if robustness_results["noise_0.01"]["pc_success"] > 70:
        checklist["robustness_validation"] = True

    # Efficiency validation
    efficiency_metrics = measure_inference_efficiency(policy)
    if efficiency_metrics["avg_inference_time_ms"] < 100:  # < 100ms per action
        checklist["efficiency_validation"] = True

    return checklist
```

#### 2. Continuous Monitoring

```python
class ContinuousMonitor:
    """Continuous performance monitoring for deployed policies"""

    def __init__(self, policy, env, alert_thresholds):
        self.policy = policy
        self.env = env
        self.alert_thresholds = alert_thresholds
        self.performance_history = []

    def monitor_episode(self, observation, action, reward, done):
        """Monitor single episode and detect anomalies"""

        # Log performance
        self.performance_history.append({
            "timestamp": time.time(),
            "reward": reward,
            "done": done,
            "action_stats": self.calculate_action_stats(action)
        })

        # Check for anomalies
        if self.detect_performance_degradation():
            self.send_alert("Performance degradation detected")

        if self.detect_action_anomalies(action):
            self.send_alert("Action anomalies detected")

    def detect_performance_degradation(self):
        """Detect significant performance degradation"""
        if len(self.performance_history) < 100:
            return False

        recent_performance = np.mean([ep["reward"] for ep in self.performance_history[-50:]])
        baseline_performance = np.mean([ep["reward"] for ep in self.performance_history[:50]])

        degradation = (baseline_performance - recent_performance) / baseline_performance

        return degradation > self.alert_thresholds["performance_degradation"]
```

## Summary and Next Steps

This comprehensive guide provides everything needed to evaluate your ACT policy in the gym-hil environment:

1. **Understanding**: Deep dive into ACT architecture and training process
2. **Setup**: Proper gym-hil environment configuration
3. **Evaluation**: Step-by-step evaluation process with metrics interpretation
4. **Analysis**: Advanced techniques for thorough performance analysis
5. **Optimization**: Best practices for efficient and reliable evaluation
6. **Real-world**: Safety, robustness, and deployment considerations

### Recommended Evaluation Workflow

1. **Start with Basic Evaluation**: Verify policy loads and executes correctly
2. **Scale Up**: Increase episode count for statistical significance
3. **Analyze Videos**: Review successful and failed episodes
4. **Test Robustness**: Evaluate under noise and domain variation
5. **Optimize Performance**: Fine-tune configuration for efficient evaluation
6. **Prepare for Deployment**: Conduct safety and reliability checks

Remember that evaluation is an iterative process. Use the insights gained from each evaluation round to improve both your policy and evaluation methodology.

For specific questions or issues, refer to the troubleshooting section or consult the LeRobot documentation and community resources.

---
noteId: "b1f57230f3ba11f08bd0898d85b76daa"
tags: []
---

# LeRobot HuggingFace Integration Guide

## Overview

LeRobot provides seamless integration with the HuggingFace Hub, enabling easy sharing, discovery, and collaboration on robot datasets and trained policies. This integration supports automatic uploads during data collection and training, comprehensive repository management, and robust versioning for both datasets and models.

## HuggingFace Hub Integration Architecture

### Core Components

```
LeRobot HuggingFace Integration
├── Dataset Upload Pipeline
│   ├── Automatic upload during recording
│   ├── Manual upload from local storage
│   ├── Batch upload for large datasets
│   └── Resume interrupted uploads
├── Model Sharing System
│   ├── Training checkpoint uploads
│   ├── Model card generation
│   ├── Evaluation results inclusion
│   └── Version management
├── Repository Management
│   ├── Automatic repository creation
│   ├── Metadata management
│   ├── File organization
│   └── Access control
└── Collaboration Features
    ├── Team workspaces
    ├── Pull requests
    ├── Issue tracking
    └── Discussion forums
```

### Key Integration Files

- **`src/lerobot/utils/hub.py`**: Core HuggingFace Hub utilities
- **`src/lerobot/datasets/lerobot_dataset.py`**: Dataset upload functionality
- **`src/lerobot/policies/pretrained.py`**: Model upload capabilities
- **`src/lerobot/configs/policies.py`**: Hub configuration parameters

## Setup and Authentication

### Installation

```bash
# Install HuggingFace Hub library
pip install huggingface_hub

# Install LeRobot with Hub support
pip install -e ".[hub]"

# Optional: Install fast transfer for uploads
pip install hf-transfer
```

### Authentication

```bash
# Login to HuggingFace Hub
huggingface-cli login --token YOUR_HUGGINGFACE_TOKEN

# Set environment variables
export HUGGINGFACE_TOKEN="your_token_here"
export HF_USER=$(huggingface-cli whoami)

# Verify authentication
huggingface-cli whoami
```

### Token Permissions

Required token permissions for LeRobot:

```python
# Token should have these scopes:
required_scopes = [
    "write",          # Upload datasets and models
    "read",           # Access public repositories
    "repo.create",    # Create new repositories
    "inference",      # Use models in inference
]
```

## Dataset Upload to HuggingFace Hub

### Automatic Upload During Recording

#### Basic Automatic Upload

```bash
# Record with automatic upload
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=spacemouse \
    --dataset.repo_id=${HF_USER}/my_dataset \
    --dataset.num_episodes=10 \
    --dataset.push_to_hub=true
```

#### Advanced Upload Configuration

```bash
# Recording with comprehensive upload settings
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 1280, height: 720, fps: 60}}" \
    --dataset.repo_id=${HF_USER}/high_quality_dataset \
    --dataset.num_episodes=50 \
    --dataset.push_to_hub=true \
    --dataset.private=false \
    --dataset.commit_message="Initial dataset upload" \
    --dataset.create_repo=true \
    --dataset.revision=main \
    --dataset.num_upload_workers=4
```

### Manual Dataset Upload

#### Upload Local Dataset

```python
# Upload local dataset to Hub
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Load local dataset
dataset = LeRobotDataset("path/to/local/dataset")

# Upload to Hub with custom settings
dataset.push_to_hub(
    repo_id=f"{HF_USER}/my_awesome_dataset",
    private=False,
    token="your_huggingface_token",
    commit_message="Add comprehensive robot manipulation dataset",
    create_repo=True,
    revision="main",
    run_as_future=False  # Blocking upload
)
```

#### Upload with Custom Metadata

```python
# Upload with enhanced metadata
from huggingface_hub import HfApi

api = HfApi()

# Create repository with detailed information
repo_url = api.create_repo(
    repo_id=f"{HF_USER}/comprehensive_dataset",
    token="your_token",
    private=False,
    repo_type="dataset",
    space_sdk=None,
    description="Large-scale robot manipulation dataset with multi-view observations",
    readme_content="# Dataset Description\n\nThis dataset contains..."
)

# Upload dataset
dataset.push_to_hub(
    repo_id=f"{HF_USER}/comprehensive_dataset",
    private=False,
    commit_message="Upload comprehensive dataset with metadata"
)
```

### Batch Upload for Large Datasets

#### Using SLURM Upload Script

```python
# Large-scale batch upload (from examples/port_datasets/slurm_upload.py)
import argparse
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi

def upload_large_dataset(dataset_path, repo_id, num_workers=50):
    """Upload large dataset with parallel processing"""

    api = HfApi()

    # Create repository
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        token="your_token"
    )

    # Upload with parallel workers
    dataset = LeRobotDataset(dataset_path)

    # Configure upload workers
    upload_config = {
        "num_workers": num_workers,
        "chunk_size": "100MB",  # Chunk size for large files
        "resume": True,  # Resume interrupted uploads
        "retry": 3,  # Number of retry attempts
        "timeout": 300  # Timeout per chunk
    }

    dataset.push_to_hub(
        repo_id=repo_id,
        **upload_config
    )

# Usage
upload_large_dataset(
    dataset_path="./large_dataset",
    repo_id=f"{HF_USER}/massive_dataset",
    num_workers=50
)
```

#### Parallel Upload with Multiple Workers

```bash
# Parallel upload script
#!/bin/bash

DATASET_PATH="./my_large_dataset"
REPO_ID="${HF_USER}/large_parallel_dataset"
NUM_WORKERS=20

# Split dataset into chunks
python -c "
from lerobot.utils.hub import split_dataset_for_upload
split_dataset_for_upload('$DATASET_PATH', '$REPO_ID', num_workers=$NUM_WORKERS)
"

# Upload in parallel
for i in $(seq 0 $((NUM_WORKERS-1))); do
    python upload_worker.py \
        --dataset_path "$DATASET_PATH" \
        --repo_id "$REPO_ID" \
        --worker_id $i \
        --num_workers $NUM_WORKERS &
done

# Wait for all uploads to complete
wait
```

### Upload Optimization

#### High-Speed Upload Configuration

```python
# Optimized upload settings
from huggingface_hub import HfApi

api = HfApi()

# Enable fast transfer
api.configure_upload(
    use_auth_token="your_token",
    max_retries=5,
    retry_on_status_codes=[500, 502, 503, 504],
    timeout=30,
    max_memory=1024 * 1024 * 1024,  # 1GB memory limit
    enable_multiprocessing=True,
    multiprocessing_start_method="spawn"
)
```

#### Resume Interrupted Uploads

```python
# Resume upload from checkpoint
from lerobot.datasets.lerobot_dataset import LeRobotDataset

dataset = LeRobotDataset("path/to/partial_dataset")

# Resume upload with same repo_id
dataset.push_to_hub(
    repo_id=f"{HF_USER}/resumed_upload",
    resume=True,  # Resume from last checkpoint
    commit_message="Resume interrupted upload",
    create_repo=False  # Repository already exists
)
```

## Model Upload and Sharing

### Automatic Upload During Training

#### Training with Hub Integration

```bash
# Train with automatic model upload
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=${HF_USER}/training_dataset \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/act_policy \
    --output_dir=./outputs/training \
    --batch_size=8 \
    --steps=100000 \
    --eval_freq=10000 \
    --save_freq=10000
```

#### Advanced Model Upload Configuration

```bash
# Training with comprehensive model sharing
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=${HF_USER}/training_dataset \
    --policy.push_to_hub=true \
    --policy.repo_id=${HF_USER}/advanced_act_policy \
    --policy.private=false \
    --policy.hub_tags=["robotics", "manipulation", "act", "so100"] \
    --policy.commit_message="Upload trained ACT policy with evaluation results" \
    --policy.create_repo=true \
    --policy.revision=main \
    --policy.eval_results=true \
    --output_dir=./outputs/advanced_training \
    --batch_size=16 \
    --steps=200000
```

### Manual Model Upload

#### Upload Trained Model

```python
from lerobot.policies.act.policy import ACTPolicy

# Load trained model
policy = ACTPolicy.from_pretrained("path/to/trained_model")

# Upload to Hub with comprehensive information
policy.push_to_hub(
    repo_id=f"{HF_USER}/my_act_policy",
    private=False,
    token="your_huggingface_token",
    commit_message="Upload ACT policy trained on custom dataset",
    create_repo=True,
    revision="main",
    # Model metadata
    tags=["robotics", "act", "manipulation", "so100"],
    license="apache-2.0",
    library_name="lerobot",
    # Optional: include evaluation results
    include_evaluation=True,
    eval_results_path="./evaluation_results.json"
)
```

#### Upload with Custom Model Card

````python
# Upload with custom model card
model_card = """
# ACT Policy for Object Manipulation

## Model Description
This ACT (Action Chunking with Transformers) policy is trained for precise object manipulation tasks.

## Training Details
- **Policy Type**: ACT
- **Dataset**: {}/{}
- **Training Steps**: {}
- **Batch Size**: {}
- **Architecture**: {}

## Performance
- **Success Rate**: {}
- **Average Episode Length**: {}
- **Environment**: {}

## Usage
```python
from lerobot.policies.act.policy import ACTPolicy
policy = ACTPolicy.from_pretrained("{}/{}")
````

## Hardware Requirements

- GPU: Required for inference
- Memory: {}
- Storage: {}
  """.format(
  HF_USER, dataset_name, training_steps, batch_size,
  architecture, success_rate, avg_episode_length,
  environment, HF_USER, repo_id, memory_req, storage_req
  )

policy.push_to_hub(
repo_id=f"{HF_USER}/policy_with_card",
model_card=model_card,
include_license=True
)

````

## Repository Management

### Repository Creation and Configuration

#### Create Dataset Repository

```python
from huggingface_hub import HfApi

api = HfApi()

# Create dataset repository
repo_url = api.create_repo(
    repo_id=f"{HF_USER}/comprehensive_dataset",
    token="your_token",
    repo_type="dataset",
    private=False,
    space_sdk=None,
    description="Large-scale robot manipulation dataset",
    readme_content="# Dataset Description\n\n...",
    license="mit"
)
````

#### Create Model Repository

```python
# Create model repository
api.create_repo(
    repo_id=f"{HF_USER}/advanced_policy",
    token="your_token",
    repo_type="model",
    private=False,
    space_sdk=None,
    description="Advanced robotics policy for manipulation tasks",
    readme_content="# Model Description\n\n...",
    license="apache-2.0",
    tags=["robotics", "manipulation", "act"]
)
```

### Repository Organization

#### Dataset Repository Structure

```
username/
├── dataset_name/
│   ├── data/                    # Episode data (parquet files)
│   │   ├── episode_000000.parquet
│   │   ├── episode_000001.parquet
│   │   └── ...
│   ├── videos/                  # Video observations
│   │   ├── observation.images.front/
│   │   │   ├── chunk-000/file-000.mp4
│   │   │   └── ...
│   │   └── observation.images.wrist/
│   │       ├── chunk-000/file-000.mp4
│   │       └── ...
│   ├── meta/                    # Metadata
│   │   ├── info.json           # Dataset information
│   │   ├── stats.json          # Statistics
│   │   └── episodes/           # Episode metadata
│   ├── images/                  # Individual frames (optional)
│   ├── README.md               # Dataset card
│   └── dataset card            # Auto-generated card
```

#### Model Repository Structure

```
username/
├── model_name/
│   ├── config.json              # Model configuration
│   ├── model.safetensors       # Model weights
│   ├── train_config.json        # Training configuration
│   ├── training_args.bin        # Training arguments
│   ├── tokenizer_config.json   # Tokenizer (if applicable)
│   ├── preprocessor_config.json # Preprocessor settings
│   ├── README.md               # Model card
│   ├── EVALUATION.md           # Evaluation results
│   └── usage_examples.py       # Usage examples
```

### Metadata Management

#### Dataset Metadata

```json
{
  "dataset_info": {
    "features": {
      "observation.state": {
        "dtype": "float32",
        "shape": [7],
        "names": ["joint_positions"]
      },
      "action": {
        "dtype": "float32",
        "shape": [7],
        "names": ["joint_targets"]
      },
      "observation.images.front": {
        "dtype": "image",
        "shape": [3, 224, 224]
      }
    },
    "splits": ["train", "validation", "test"],
    "download_size": "1.2 GB",
    "dataset_size": "2.5 GB"
  },
  "task_categories": ["robotics", "manipulation"],
  "language_creators": ["machine-generated"],
  "size_categories": ["1M<n<10M"],
  "tags": ["robotics", "manipulation", "so100", "space-mouse"]
}
```

#### Model Metadata

```json
{
  "model_config": {
    "name": "ACT",
    "version": "1.0",
    "architecture": "transformer",
    "task_type": "robotic-manipulation"
  },
  "training_config": {
    "dataset": "username/dataset_name",
    "training_steps": 100000,
    "batch_size": 8,
    "learning_rate": 1e-5,
    "optimizer": "adamw"
  },
  "evaluation_results": {
    "success_rate": 0.85,
    "average_reward": 0.72,
    "episodes_evaluated": 100
  },
  "hardware_requirements": {
    "gpu_required": true,
    "gpu_memory": "4GB",
    "system_memory": "8GB"
  }
}
```

## Version Management

### Semantic Versioning

#### Dataset Versioning

```python
# Upload with version tag
dataset.push_to_hub(
    repo_id=f"{HF_USER}/my_dataset",
    revision="v1.0.0",  # Semantic version
    commit_message="Initial release v1.0.0"
)

# Upload new version
dataset.push_to_hub(
    repo_id=f"{HF_USER}/my_dataset",
    revision="v1.1.0",  # Patch version
    commit_message="Add 50 new episodes and improved metadata"
)
```

#### Model Versioning

```python
# Upload model with versioning
policy.push_to_hub(
    repo_id=f"{HF_USER}/my_policy",
    revision="v2.0.0",
    commit_message="Major architecture update with improved performance",
    tags=["robotics", "act", "v2.0"]
)
```

### Branch Management

#### Development Branches

```python
# Upload to development branch
dataset.push_to_hub(
    repo_id=f"{HF_USER}/my_dataset",
    revision="dev",  # Development branch
    commit_message="Development version with experimental features"
)

# Merge development to main
api = HfApi()
api.create_pull_request(
    repo_id=f"{HF_USER}/my_dataset",
    head="dev",
    base="main",
    title="Merge experimental features to main",
    description="This PR includes new dataset processing methods and improved metadata."
)
```

#### Experimental Features

```python
# Upload to experimental branch
policy.push_to_hub(
    repo_id=f"{HF_USER}/experimental_policy",
    revision="experimental/attn-update",
    commit_message="Experimental attention mechanism update"
)
```

### Release Management

#### Create Dataset Release

````python
from huggingface_hub import HfApi

api = HfApi()

# Create release tag
api.create_tag(
    repo_id=f"{HF_USER}/my_dataset",
    repo_type="dataset",
    tag="v1.0.0",
    tag_message="First stable release of the manipulation dataset"
)

# Generate release notes
release_notes = """
# Release v1.0.0

## What's New
- Initial dataset release with 1000 episodes
- Multi-view observations (front and wrist cameras)
- Comprehensive metadata and statistics
- Compatible with LeRobot v3.0

## Dataset Statistics
- Episodes: 1000
- Total Frames: 150,000
- Average Episode Length: 150 frames
- Success Rate: 92%

## Usage
```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
dataset = LeRobotDataset("username/my_dataset")
````

"""

api.create_release(
repo_id=f"{HF_USER}/my_dataset",
repo_type="dataset",
tag="v1.0.0",
release_notes=release_notes
)

````

## Integration Workflows

### Complete Data Collection to Deployment Workflow

```bash
#!/bin/bash
# Complete workflow script

HF_USER=$(huggingface-cli whoami)
TASK_NAME="pick_place_v1"
DATASET_NAME="${HF_USER}/${TASK_NAME}_dataset"
MODEL_NAME="${HF_USER}/${TASK_NAME}_model"

# 1. Data Collection with Upload
echo "Step 1: Collecting and uploading data..."
lerobot-record \
    --robot.type=so100_follower \
    --robot.port=/dev/tty.usbmodem58760431541 \
    --robot.cameras="{front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
    --teleop.type=spacemouse \
    --dataset.repo_id=${DATASET_NAME} \
    --dataset.num_episodes=100 \
    --dataset.push_to_hub=true \
    --dataset.single_task="Pick up the cube and place it in the target zone"

# 2. Train Policy with Hub Integration
echo "Step 2: Training policy and uploading model..."
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=${DATASET_NAME} \
    --policy.push_to_hub=true \
    --policy.repo_id=${MODEL_NAME} \
    --output_dir=./outputs/${TASK_NAME}_training \
    --batch_size=8 \
    --steps=50000 \
    --eval_freq=5000

# 3. Evaluate and Share Results
echo "Step 3: Evaluating model and sharing results..."
lerobot-eval \
    --policy.path=${MODEL_NAME} \
    --env.type=aloha \
    --num_episodes=50 \
    --output_dir=./evaluation_results/${TASK_NAME}

# Upload evaluation results
huggingface-cli upload ${MODEL_NAME} \
    ./evaluation_results/${TASK_NAME} \
    evaluation_results/ \
    --commit-message="Add evaluation results"
````

### Continuous Integration with Hub

```yaml
# .github/workflows/hub-integration.yml
name: Hub Integration

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test-and-upload:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: "3.9"

      - name: Install dependencies
        run: |
          pip install -e ".[dev,test,hub]"

      - name: Test dataset creation
        run: |
          python tests/test_dataset_creation.py

      - name: Test model training
        run: |
          python tests/test_training.py --steps=100

      - name: Upload to Hub (main branch only)
        if: github.ref == 'refs/heads/main'
        env:
          HF_TOKEN: ${{ secrets.HUGGINGFACE_TOKEN }}
        run: |
          huggingface-cli login --token $HF_TOKEN

          # Upload test dataset
          python scripts/upload_test_dataset.py

          # Upload test model
          python scripts/upload_test_model.py
```

## Best Practices

### Dataset Naming Conventions

```python
# Recommended naming patterns
naming_conventions = {
    "task_robot_teleop_version": "{task}_{robot}_{teleop}_v{version}",
    "examples": [
        "pick_place_so100_spacemouse_v1",
        "assembly_aloha_gamepad_v2",
        "grasping_pusht_keyboard_v1_5"
    ]
}
```

### Model Naming Conventions

```python
# Model naming patterns
model_naming = {
    "policy_task_robot_version": "{policy_type}_{task}_{robot}_v{version}",
    "examples": [
        "act_pick_place_so100_v1",
        "diffusion_assembly_aloha_v2",
        "tdpc_grasping_pusht_v1"
    ]
}
```

### Quality Standards

#### Dataset Quality Requirements

```python
# Dataset quality checklist
dataset_quality_checklist = [
    "✓ Minimum 20 episodes for training",
    "✓ Complete metadata and statistics",
    "✓ Consistent frame rate and resolution",
    "✓ Multiple camera angles (when possible)",
    "✓ Clear task description and examples",
    "✓ Proper episode segmentation",
    "✓ Success and failure examples included",
    "✓ Varied initial conditions"
]
```

#### Model Quality Requirements

```python
# Model quality checklist
model_quality_checklist = [
    "✓ Comprehensive model card with usage examples",
    "✓ Complete training configuration",
    "✓ Evaluation results with metrics",
    "✓ Hardware requirements clearly stated",
    "✓ License and terms of use specified",
    "✓ Version history and changelog",
    "✓ Compatible with current LeRobot version",
    "✓ Documentation for integration"
]
```

### Performance Optimization

#### Upload Speed Optimization

```python
# Optimize upload performance
upload_optimization = {
    "enable_hf_transfer": "pip install hf-transfer && export HF_HUB_ENABLE_HF_TRANSFER=1",
    "use_parallel_uploads": "num_upload_workers >= 4",
    "chunk_size": "100MB for large files",
    "resume_uploads": "Always enable for large datasets",
    "use_fast_safetensors": "Convert to safetensors format before upload"
}
```

#### Memory Optimization

```python
# Memory-efficient uploads
memory_optimization = {
    "stream_large_files": "Use streaming for files > 500MB",
    "batch_uploads": "Upload in batches to limit memory usage",
    "compress_before_upload": "Use appropriate compression",
    "monitor_memory": "Track memory usage during uploads"
}
```

## Troubleshooting

### Common Upload Issues

#### Authentication Problems

```bash
# Check authentication status
huggingface-cli whoami

# Re-login if needed
huggingface-cli login --token YOUR_TOKEN

# Test token permissions
python -c "
from huggingface_hub import HfApi
api = HfApi(token='YOUR_TOKEN')
try:
    user_info = api.whoami()
    print(f'✓ Authenticated as: {user_info[\"name\"]}')
except Exception as e:
    print(f'✗ Authentication failed: {e}')
"
```

#### Upload Failures

```python
# Handle upload failures
from huggingface_hub import HfApi, Repository

def robust_upload(local_path, repo_id, max_retries=3):
    api = HfApi()

    for attempt in range(max_retries):
        try:
            # Upload with retry logic
            api.upload_folder(
                folder_path=local_path,
                repo_id=repo_id,
                repo_type="dataset",
                ignore_patterns=["*.tmp", "*.lock"]
            )
            print(f"✓ Upload successful on attempt {attempt + 1}")
            return True

        except Exception as e:
            print(f"✗ Upload failed on attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff

    return False
```

#### Large File Issues

```python
# Handle large file uploads
from huggingface_hub import HfApi
import os

def upload_large_files(local_path, repo_id):
    api = HfApi()

    # Check for files > 5GB
    for root, dirs, files in os.walk(local_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)

            if file_size > 5 * 1024 * 1024 * 1024:  # 5GB
                print(f"Large file detected: {file_path} ({file_size / 1024**3:.1f}GB)")

                # Use chunked upload for large files
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=file_path.replace(local_path + "/", ""),
                    repo_id=repo_id,
                    repo_type="dataset",
                    chunk_size="100MB"
                )
```

This comprehensive HuggingFace integration guide provides everything needed to effectively share datasets and models, manage repositories, and collaborate on the LeRobot ecosystem through the HuggingFace Hub.

---
noteId: "6aac8b30b08f11f08bd0898d85b76daa"
tags: []
---

# LeRobot Hugging Face Upload Guide

This comprehensive guide covers how to upload datasets and trained models to Hugging Face Hub using LeRobot, including best practices for dataset organization, model sharing, and community collaboration.

## Table of Contents

1. [Overview](#overview)
2. [Dataset Upload](#dataset-upload)
3. [Model Upload](#model-upload)
4. [Upload Configuration](#upload-configuration)
5. [Best Practices](#best-practices)
6. [Community Guidelines](#community-guidelines)
7. [Troubleshooting](#troubleshooting)

## Overview

Hugging Face Hub integration in LeRobot enables:

- **Dataset Sharing**: Upload and share robot demonstration datasets
- **Model Distribution**: Share trained policies with the community
- **Version Control**: Track dataset and model versions
- **Collaboration**: Enable community access to your work
- **Discovery**: Make your datasets and models discoverable

### Key Components

- **LeRobotDataset v3.0**: Standardized format for robot data
- **Hugging Face Hub**: Central repository for datasets and models
- **Upload Tools**: Automated upload scripts and utilities
- **Metadata Management**: Rich metadata for search and discovery

## Dataset Upload

### Automatic Upload During Recording

The easiest way to upload datasets is through automatic upload during data collection:

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras="{ front: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --teleop.type=spacemouse \
  --display_data=true \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --dataset.num_episodes=20 \
  --dataset.single_task="Pick and place objects" \
  --dataset.push_to_hub=true
```

### Manual Dataset Upload

#### Using lerobot-record with push_to_hub

```bash
# Upload existing local dataset
lerobot-record \
  --dataset.repo_id=${HF_USER}/my_dataset \
  --dataset.push_to_hub=true \
  --resume=false
```

#### Using Hugging Face CLI

```bash
# Upload dataset directory
huggingface-cli upload ${HF_USER}/my_dataset ~/.cache/huggingface/lerobot/my_dataset --repo-type dataset
```

#### Using Python API

```python
#!/usr/bin/env python3
"""
Manual dataset upload script
"""

import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def upload_dataset_manually(repo_id, local_path=None, private=False):
    """Upload dataset to Hugging Face Hub manually."""

    print(f"📤 Uploading dataset to Hugging Face Hub")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")

    # Create repository if it doesn't exist
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="dataset",
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository created/verified: {repo_url}")
    except Exception as e:
        print(f"⚠️  Repository creation issue: {e}")

    # Load dataset
    if local_path:
        dataset = LeRobotDataset(repo_id, root=local_path)
    else:
        dataset = LeRobotDataset(repo_id)

    # Upload to hub
    try:
        print("🚀 Starting upload...")
        dataset.push_to_hub()
        print(f"✅ Dataset uploaded successfully!")

        # Display dataset info
        print(f"\n📊 Dataset Information:")
        print(f"  Repository: {repo_id}")
        print(f"  Episodes: {dataset.num_episodes}")
        print(f"  Frames: {dataset.num_frames}")
        print(f"  FPS: {dataset.fps}")
        print(f"  Features: {list(dataset.features.keys())}")

        return True

    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="Repository ID")
    parser.add_argument("--local-path", type=str, help="Local dataset path")
    parser.add_argument("--private", action="store_true", help="Make repository private")

    args = parser.parse_args()

    upload_dataset_manually(
        repo_id=args.repo_id,
        local_path=args.local_path,
        private=args.private
    )
```

### Dataset Upload with Custom Metadata

```python
def upload_dataset_with_metadata(repo_id, metadata):
    """Upload dataset with custom metadata."""

    # Create repository
    create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True
    )

    # Load dataset
    dataset = LeRobotDataset(repo_id)

    # Add custom metadata
    dataset_info = {
        "task": metadata.get("task", "Unknown"),
        "robot": metadata.get("robot", "Unknown"),
        "teleoperator": metadata.get("teleoperator", "Unknown"),
        "environment": metadata.get("environment", "Unknown"),
        "difficulty": metadata.get("difficulty", "Medium"),
        "tags": metadata.get("tags", ["lerobot", "robotics"]),
        "description": metadata.get("description", "Robot demonstration dataset"),
        "license": metadata.get("license", "mit"),
        "language": ["en"],
        "size_categories": metadata.get("size_categories", "1K<n<10K"),
        "task_categories": metadata.get("task_categories", ["manipulation"])
    }

    # Update dataset info
    dataset.meta.info.update(dataset_info)

    # Upload
    dataset.push_to_hub()

    print(f"✅ Dataset uploaded with custom metadata!")
    return True
```

### Batch Dataset Upload

```python
def upload_multiple_datasets(dataset_configs):
    """Upload multiple datasets in batch."""

    results = []

    for config in dataset_configs:
        repo_id = config["repo_id"]
        local_path = config.get("local_path")
        metadata = config.get("metadata", {})

        print(f"\n📤 Uploading dataset: {repo_id}")

        try:
            # Create repository
            create_repo(
                repo_id=repo_id,
                repo_type="dataset",
                private=config.get("private", False),
                exist_ok=True
            )

            # Load and upload dataset
            if local_path:
                dataset = LeRobotDataset(repo_id, root=local_path)
            else:
                dataset = LeRobotDataset(repo_id)

            # Add metadata if provided
            if metadata:
                dataset.meta.info.update(metadata)

            dataset.push_to_hub()

            results.append({
                "repo_id": repo_id,
                "status": "success",
                "episodes": dataset.num_episodes,
                "frames": dataset.num_frames
            })

            print(f"✅ {repo_id} uploaded successfully!")

        except Exception as e:
            results.append({
                "repo_id": repo_id,
                "status": "failed",
                "error": str(e)
            })
            print(f"❌ {repo_id} upload failed: {e}")

    # Summary
    print(f"\n📊 Upload Summary:")
    successful = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")

    return results

# Example usage
dataset_configs = [
    {
        "repo_id": "user/pick_place_dataset",
        "local_path": "/path/to/pick_place",
        "metadata": {
            "task": "Pick and place objects",
            "robot": "SO-101",
            "teleoperator": "SpaceMouse",
            "tags": ["lerobot", "manipulation", "pick-place"]
        }
    },
    {
        "repo_id": "user/grasping_dataset",
        "local_path": "/path/to/grasping",
        "metadata": {
            "task": "Object grasping",
            "robot": "ALOHA",
            "teleoperator": "Gamepad",
            "tags": ["lerobot", "manipulation", "grasping"]
        }
    }
]

upload_multiple_datasets(dataset_configs)
```

## Model Upload

### Upload Trained Policies

After training a policy, upload it to Hugging Face Hub:

```bash
# Upload trained model
huggingface-cli upload ${HF_USER}/act_policy outputs/train/act_experiment/checkpoints/002000/pretrained_model
```

### Model Upload Script

````python
#!/usr/bin/env python3
"""
Upload trained model to Hugging Face Hub
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo
import json

def upload_trained_model(model_path, repo_id, private=False, description=None):
    """Upload trained model to Hugging Face Hub."""

    print(f"🤖 Uploading trained model")
    print(f"Model path: {model_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")

    # Validate model path
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"❌ Model path does not exist: {model_path}")
        return False

    # Check for required files
    required_files = ["config.json", "model.safetensors", "train_config.json"]
    missing_files = []

    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)

    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False

    # Create repository
    try:
        repo_url = create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True
        )
        print(f"✅ Repository created/verified: {repo_url}")
    except Exception as e:
        print(f"⚠️  Repository creation issue: {e}")

    # Upload model files
    try:
        api = HfApi()

        print("🚀 Starting model upload...")

        # Upload all files in the model directory
        api.upload_folder(
            repo_id=repo_id,
            folder_path=model_path,
            repo_type="model",
            commit_message="Upload trained LeRobot policy"
        )

        # Add model card if description provided
        if description:
            model_card = f"""---
license: mit
tags:
- lerobot
- robotics
- manipulation
- act
---

# {repo_id}

{description}

## Usage

```python
from lerobot.policies.act.policy import ACTPolicy

# Load the trained policy
policy = ACTPolicy.from_pretrained("{repo_id}")

# Use for inference
action = policy.select_action(observation)
````

## Training Details

This model was trained using LeRobot framework.
"""

            api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=model_card.encode(),
                path_in_repo="README.md",
                repo_type="model"
            )

        print(f"✅ Model uploaded successfully!")
        print(f"🔗 Model URL: https://huggingface.co/{repo_id}")

        return True

    except Exception as e:
        print(f"❌ Model upload failed: {e}")
        return False

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Upload trained model to Hugging Face Hub")
parser.add_argument("--model-path", type=str, required=True, help="Path to trained model")
parser.add_argument("--repo-id", type=str, required=True, help="Repository ID")
parser.add_argument("--private", action="store_true", help="Make repository private")
parser.add_argument("--description", type=str, help="Model description")

    args = parser.parse_args()

    upload_trained_model(
        model_path=args.model_path,
        repo_id=args.repo_id,
        private=args.private,
        description=args.description
    )

````

### Model Upload with Evaluation Results

```python
def upload_model_with_evaluation(model_path, repo_id, eval_results):
    """Upload model with evaluation results."""

    # Upload model
    upload_trained_model(model_path, repo_id)

    # Create evaluation report
    eval_report = f"""# Evaluation Results

## Performance Metrics

- **Average Reward**: {eval_results.get('avg_reward', 'N/A')}
- **Success Rate**: {eval_results.get('success_rate', 'N/A')}
- **Episodes Evaluated**: {eval_results.get('num_episodes', 'N/A')}

## Detailed Results

```json
{json.dumps(eval_results, indent=2)}
````

## Usage

```python
from lerobot.policies.act.policy import ACTPolicy

# Load the trained policy
policy = ACTPolicy.from_pretrained("{repo_id}")

# Use for inference
action = policy.select_action(observation)
```

"""

    # Upload evaluation report
    api = HfApi()
    api.upload_file(
        repo_id=repo_id,
        path_or_fileobj=eval_report.encode(),
        path_in_repo="EVALUATION.md",
        repo_type="model"
    )

    print("✅ Evaluation results uploaded!")

````

## Upload Configuration

### Hugging Face Authentication

```bash
# Login to Hugging Face
huggingface-cli login --token ${HUGGINGFACE_TOKEN}

# Verify authentication
huggingface-cli whoami
````

### Environment Variables

```bash
# Set environment variables
export HUGGINGFACE_TOKEN="your_token_here"
export HF_USER=$(huggingface-cli whoami | head -n 1)
```

### Upload Configuration File

```json
{
  "upload_config": {
    "default_repo_type": "dataset",
    "default_private": false,
    "chunk_size": 1000000,
    "max_retries": 3,
    "retry_delay": 5,
    "metadata": {
      "license": "mit",
      "language": ["en"],
      "tags": ["lerobot", "robotics"]
    }
  }
}
```

## Best Practices

### Dataset Organization

1. **Naming Conventions**

   ```
   {username}/{task}_{robot}_{teleoperator}_{version}
   Examples:
   - user/pick_place_so101_spacemouse_v1
   - user/grasping_aloha_gamepad_v2
   - user/assembly_koch_keyboard_v1
   ```

2. **Repository Structure**

   ```
   dataset_name/
   ├── data/
   │   ├── episode_000000.parquet
   │   ├── episode_000001.parquet
   │   └── ...
   ├── videos/
   │   ├── episode_000000.mp4
   │   ├── episode_000001.mp4
   │   └── ...
   ├── meta/
   │   ├── info.json
   │   ├── episodes.json
   │   └── features.json
   └── README.md
   ```

3. **Metadata Standards**
   ```json
   {
     "task": "Clear task description",
     "robot": "Robot type and configuration",
     "teleoperator": "Input device used",
     "environment": "Recording environment",
     "difficulty": "Easy/Medium/Hard",
     "tags": ["lerobot", "manipulation", "specific-task"],
     "description": "Detailed dataset description",
     "license": "mit",
     "language": ["en"],
     "size_categories": "1K<n<10K",
     "task_categories": ["manipulation"]
   }
   ```

### Model Organization

1. **Model Naming**

   ```
   {username}/{policy_type}_{task}_{robot}_{version}
   Examples:
   - user/act_pick_place_so101_v1
   - user/diffusion_grasping_aloha_v2
   - user/tdmpc_assembly_koch_v1
   ```

2. **Model Structure**

   ```
   model_name/
   ├── config.json
   ├── model.safetensors
   ├── train_config.json
   ├── README.md
   └── EVALUATION.md
   ```

3. **Model Documentation**
   - Include training details
   - Provide usage examples
   - Document performance metrics
   - Include evaluation results

### Upload Optimization

1. **Compression**

   ```python
   # Compress videos before upload
   import cv2

   def compress_video(input_path, output_path, quality=23):
       """Compress video for upload."""
       cap = cv2.VideoCapture(input_path)
       fourcc = cv2.VideoWriter_fourcc(*'mp4v')
       out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))

       while True:
           ret, frame = cap.read()
           if not ret:
               break
           out.write(frame)

       cap.release()
       out.release()
   ```

2. **Chunked Upload**

   ```python
   # Upload large files in chunks
   def upload_large_file(repo_id, file_path, chunk_size=1000000):
       """Upload large file in chunks."""
       api = HfApi()

       with open(file_path, 'rb') as f:
           chunk = f.read(chunk_size)
           while chunk:
               # Upload chunk
               api.upload_file(
                   repo_id=repo_id,
                   path_or_fileobj=chunk,
                   path_in_repo=os.path.basename(file_path)
               )
               chunk = f.read(chunk_size)
   ```

3. **Resume Upload**

   ```python
   # Resume interrupted uploads
   def resume_upload(repo_id, local_path):
       """Resume interrupted upload."""
       api = HfApi()

       # Check existing files
       existing_files = api.list_repo_files(repo_id, repo_type="dataset")

       # Upload only missing files
       for file_path in Path(local_path).rglob("*"):
           if file_path.is_file():
               relative_path = file_path.relative_to(local_path)
               if str(relative_path) not in existing_files:
                   api.upload_file(
                       repo_id=repo_id,
                       path_or_fileobj=file_path,
                       path_in_repo=str(relative_path),
                       repo_type="dataset"
                   )
   ```

## Community Guidelines

### Dataset Sharing

1. **Quality Standards**
   - Ensure data quality and consistency
   - Provide clear task descriptions
   - Include sufficient episodes for training
   - Document recording conditions

2. **Ethical Considerations**
   - Respect privacy and safety
   - Follow local regulations
   - Provide appropriate disclaimers
   - Ensure responsible use

3. **Documentation Requirements**
   - Clear README with usage instructions
   - Detailed metadata
   - Task descriptions
   - Recording environment details

### Model Sharing

1. **Performance Standards**
   - Include evaluation results
   - Provide baseline comparisons
   - Document limitations
   - Share training details

2. **Usage Guidelines**
   - Provide clear usage examples
   - Document requirements
   - Include safety considerations
   - Specify supported environments

3. **Attribution**
   - Credit original datasets
   - Acknowledge contributions
   - Follow licensing requirements
   - Maintain proper citations

## Troubleshooting

### Common Upload Issues

#### Authentication Problems

```bash
# Check authentication
huggingface-cli whoami

# Re-authenticate
huggingface-cli login --token ${HUGGINGFACE_TOKEN}
```

#### Network Issues

```python
# Retry with exponential backoff
import time
import random

def upload_with_retry(api, repo_id, file_path, max_retries=3):
    """Upload with retry logic."""
    for attempt in range(max_retries):
        try:
            api.upload_file(
                repo_id=repo_id,
                path_or_fileobj=file_path,
                path_in_repo=os.path.basename(file_path)
            )
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                delay = (2 ** attempt) + random.uniform(0, 1)
                print(f"Upload failed, retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                print(f"Upload failed after {max_retries} attempts: {e}")
                return False
```

#### Storage Issues

```bash
# Check available space
df -h

# Clean up temporary files
rm -rf /tmp/lerobot_*

# Compress large files
gzip large_file.parquet
```

#### Permission Issues

```bash
# Check repository permissions
huggingface-cli repo-info ${HF_USER}/dataset_name

# Make repository public
huggingface-cli repo-settings ${HF_USER}/dataset_name --repo-type dataset --private false
```

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Upload with debug info
api = HfApi()
api.upload_file(
    repo_id=repo_id,
    path_or_fileobj=file_path,
    path_in_repo=filename,
    repo_type="dataset"
)
```

### Upload Validation

```python
def validate_upload(repo_id, repo_type="dataset"):
    """Validate uploaded content."""
    api = HfApi()

    # List repository files
    files = api.list_repo_files(repo_id, repo_type=repo_type)
    print(f"Repository files: {files}")

    # Check file sizes
    for file in files:
        file_info = api.repo_info(repo_id, repo_type=repo_type)
        print(f"File: {file}, Size: {file_info.size}")

    # Validate dataset structure
    if repo_type == "dataset":
        dataset = LeRobotDataset(repo_id)
        print(f"Dataset validation:")
        print(f"  Episodes: {dataset.num_episodes}")
        print(f"  Frames: {dataset.num_frames}")
        print(f"  Features: {list(dataset.features.keys())}")

    return True
```

## Conclusion

This guide provides comprehensive coverage of LeRobot's Hugging Face Hub integration. Key takeaways:

- Use automatic upload during data collection for convenience
- Implement manual upload scripts for advanced control
- Follow naming conventions and metadata standards
- Optimize uploads for large datasets and models
- Maintain quality standards for community sharing
- Handle common issues with retry logic and validation

For specific upload scenarios or advanced configurations, refer to the Hugging Face Hub documentation and LeRobot examples.

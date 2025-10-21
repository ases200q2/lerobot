#!/usr/bin/env python3

"""
Script to upload the converted v3.0 dataset to a new Hugging Face repository.
This will create a new repository with the v3.0 format while preserving the original.
"""

import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi, create_repo
from lerobot.datasets.lerobot_dataset import LeRobotDataset

def upload_converted_dataset():
    # Configuration
    old_repo_id = "ases200q2/PandaPickCubeSpacemouseRandom2"
    new_repo_id = "ases200q2/PandaPickCubeSpacemouseRandom2_v30"
    local_path = Path("/tmp/dataset_conversion/ases200q2/PandaPickCubeSpacemouseRandom2")

    print(f"Uploading converted dataset from {local_path} to {new_repo_id}")

    # Create the new repository
    try:
        repo_url = create_repo(
            repo_id=new_repo_id,
            repo_type="dataset",
            private=False,
            exist_ok=True
        )
        print(f"Created repository: {repo_url}")
    except Exception as e:
        print(f"Repository might already exist or there was an error: {e}")

    # Load the dataset and push to hub
    try:
        # Use HfApi to directly upload the folder content
        from huggingface_hub import HfApi

        api = HfApi()

        # Update the info.json with new repo_id
        info_path = local_path / "meta" / "info.json"
        import json
        with open(info_path, 'r') as f:
            info_data = json.load(f)
        info_data["repo_id"] = new_repo_id
        with open(info_path, 'w') as f:
            json.dump(info_data, f, indent=2)

        # Push to hub
        print("Pushing dataset to hub...")
        api.upload_folder(
            repo_id=new_repo_id,
            folder_path=local_path,
            repo_type="dataset",
            commit_message="Upload v3.0 converted dataset"
        )

        print(f"✅ Successfully uploaded v3.0 dataset to: {new_repo_id}")
        print(f"📊 Dataset info:")
        print(f"   - Episodes: {info_data['total_episodes']}")
        print(f"   - Frames: {info_data['total_frames']}")
        print(f"   - Version: {info_data['codebase_version']}")
        print(f"   - Features: {len(info_data['features'])}")

    except Exception as e:
        print(f"❌ Error uploading dataset: {e}")
        raise

if __name__ == "__main__":
    upload_converted_dataset()
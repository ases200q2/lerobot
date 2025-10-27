#!/usr/bin/env python3

"""
Script to re-upload the v30.5 dataset to fix the corrupted parquet file.
"""

import json
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def reupload_v30_5_dataset(local_path: str | Path):
    """
    Re-upload the v30.5 dataset from a local path.

    Args:
        local_path: Path to the local dataset directory
    """
    local_path = Path(local_path)

    if not local_path.exists():
        print(f"❌ Error: Local path does not exist: {local_path}")
        return

    # Check if it's a valid dataset
    info_path = local_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"❌ Error: Not a valid LeRobot dataset: {info_path} not found")
        return

    print(f"📁 Local dataset path: {local_path}")

    # Load info
    with open(info_path) as f:
        info_data = json.load(f)

    repo_id = info_data.get("repo_id", "ases200q2/PandaPickCubeSpacemouse_v30.5")

    print("📊 Dataset info:")
    print(f"   - Repo ID: {repo_id}")
    print(f"   - Episodes: {info_data['total_episodes']}")
    print(f"   - Frames: {info_data['total_frames']}")

    # Create repository
    try:
        repo_url = create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
        print(f"✓ Repository ready: {repo_url}")
    except Exception as e:
        print(f"⚠️  Repository might already exist: {e}")

    # Upload
    api = HfApi()
    print("📤 Uploading dataset...")

    try:
        api.upload_folder(
            repo_id=repo_id,
            folder_path=str(local_path),
            repo_type="dataset",
            commit_message="Re-upload v30.5 dataset with fixed parquet files",
        )

        print(f"✅ Successfully uploaded dataset to: {repo_id}")
        print("\n🎉 You can now use this dataset for training!")

    except Exception as e:
        print(f"❌ Error uploading: {e}")
        raise


if __name__ == "__main__":
    # You need to provide the local path to your v30.5 dataset
    # Examples:
    # - If you have it in ~/.cache/huggingface/lerobot/ases200q2/PandaPickCubeSpacemouse_v30.5
    # - Or wherever your local dataset is stored

    print("=" * 60)
    print("Re-upload v30.5 Dataset")
    print("=" * 60)
    print()

    # TODO: Specify your local dataset path here
    local_dataset_path = input("Enter the path to your local v30.5 dataset: ").strip()

    if not local_dataset_path:
        print("\n❌ No path provided. Exiting.")
        exit(1)

    reupload_v30_5_dataset(local_dataset_path)

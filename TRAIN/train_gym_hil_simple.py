#!/usr/bin/env python3

"""
Simple script to train a policy for gym_hil/PandaPickCubeSpacemouse-v0 environment
using the ases200q2/PandaPickCubeSpacemouseRandom2_v30 dataset.
"""

import subprocess
import sys
from datetime import datetime


def main():
    """Train ACT policy for gym_hil environment."""

    # Generate date suffix
    date_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Training configuration
    dataset_id = "ases200q2/PandaPickCubeSpacemouseRandom2_v30"
    policy_type = "act"  # Using ACT policy as it's commonly used
    output_dir = f"outputs/train/PandaPickCubeSpacemouseRandom2_ACT_test_{date_suffix}"
    job_name = f"PandaPickCubeSpacemouseRandom2_ACT_test_{date_suffix}"
    steps = 20000  # Reasonable number of steps for initial training
    batch_size = 32

    print(f"Training {policy_type} policy...")
    print(f"Dataset: {dataset_id}")
    print(f"Output: {output_dir}")
    print(f"Steps: {steps}")
    print(f"Batch size: {batch_size}")

    # Build the lerobot-train command
    cmd = [
        "python",
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={dataset_id}",
        f"--policy.type={policy_type}",
        f"--output_dir={output_dir}",
        f"--job_name={job_name}",
        "--policy.device=cuda",
        "--wandb.enable=true",  # Enable wandb for logging
        f"--policy.repo_id={job_name}",
        f"--batch_size={batch_size}",
        f"--steps={steps}",
        "--dataset.video_backend=pyav",
    ]

    print(f"Running command: {' '.join(cmd)}")

    try:
        # Run the training command
        subprocess.run(cmd, check=True, capture_output=False)
        print("Training completed successfully!")

    except subprocess.CalledProcessError as e:
        print(f"Training failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()

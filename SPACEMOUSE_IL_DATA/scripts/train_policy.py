#!/usr/bin/env python3

"""
ACT Policy Training Script for PandaPickCubeSpacemouse-v0

This script trains an Action Chunking Transformer (ACT) policy using imitation learning
on the collected SpaceMouse demonstration data.

ACT is chosen because:
  - Strong performance on manipulation tasks
  - Handles temporal dependencies well through action chunking
  - Uses transformer architecture for learning from visual and state inputs
  - Standard choice for robot imitation learning

Usage:
    python scripts/train_policy.py --dataset_id username/panda_spacemouse_il_data
    python scripts/train_policy.py --dataset_id username/dataset --steps 50000 --batch_size 16
    python scripts/train_policy.py --config configs/train_act_config.json
"""

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def load_config(config_path):
    """Load training configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)


def check_dataset_exists(dataset_id):
    """Check if dataset exists on HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        try:
            api.dataset_info(dataset_id)
            logging.info(f"✓ Dataset found: {dataset_id}")
            return True
        except Exception:
            logging.error(f"✗ Dataset not found: {dataset_id}")
            logging.error("  Please check the repository ID or upload your dataset first")
            return False
    except ImportError:
        logging.warning("⚠ huggingface_hub not installed, skipping dataset check")
        return True


def check_gpu():
    """Check GPU availability and memory."""
    try:
        import torch
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logging.info(f"✓ GPU: {device} ({memory_gb:.1f} GB)")

            if memory_gb < 8:
                logging.warning(f"⚠ GPU memory is low ({memory_gb:.1f} GB)")
                logging.warning("  Consider reducing batch_size if you encounter OOM errors")

            return True
        else:
            logging.warning("⚠ No GPU detected, training will be slow on CPU")
            return False
    except ImportError:
        logging.error("✗ PyTorch not installed")
        return False


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train ACT policy with imitation learning on SpaceMouse demonstrations"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_id",
        type=str,
        default=None,
        help="HuggingFace dataset repository ID (e.g., username/panda_spacemouse_il_data)",
    )

    # Training arguments
    parser.add_argument(
        "--steps",
        type=int,
        default=100000,
        help="Number of training steps (default: 100000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=['cuda', 'cpu', 'mps'],
        help="Device to use for training (default: cuda)",
    )

    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/train/panda_spacemouse_act_TIMESTAMP)",
    )
    parser.add_argument(
        "--job_name",
        type=str,
        default=None,
        help="Job name for tracking (default: panda_spacemouse_act_TIMESTAMP)",
    )

    # Resume and upload
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint",
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        default=True,
        help="Upload checkpoints to HuggingFace Hub (default: True)",
    )
    parser.add_argument(
        "--no_push_to_hub",
        dest="push_to_hub",
        action="store_false",
        help="Do not upload to HuggingFace Hub",
    )

    # Logging
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )

    # Config file
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config JSON file (optional)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("  ACT Policy Training for PandaPickCubeSpacemouse-v0")
    print("="*70 + "\n")

    # Load config if provided
    config = {}
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            base_dir = Path(__file__).parent.parent
            config_path = base_dir / config_path

        if config_path.exists():
            config = load_config(config_path)
            logging.info(f"Loaded configuration from: {config_path}")
        else:
            logging.error(f"Configuration file not found: {config_path}")
            sys.exit(1)

    # Determine dataset ID
    dataset_id = args.dataset_id or config.get('dataset', {}).get('repo_id')

    if not dataset_id or dataset_id.startswith('YOUR_HF_USERNAME'):
        logging.error("Dataset ID not specified or contains placeholder")
        logging.error("Please provide --dataset_id or update config file")
        logging.error("Example: --dataset_id username/panda_spacemouse_il_data")
        sys.exit(1)

    # Pre-flight checks
    logging.info("\n--- Pre-flight Checks ---")
    check_gpu()
    check_dataset_exists(dataset_id)
    logging.info("-------------------------\n")

    # Generate timestamp for default names
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine output directory and job name
    output_dir = args.output_dir or config.get('output', {}).get('output_dir',
                                                                   f"outputs/train/panda_spacemouse_act_{timestamp}")
    job_name = args.job_name or config.get('output', {}).get('job_name',
                                                              f"panda_spacemouse_act_{timestamp}")

    # Create output directory path
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        base_dir = Path(__file__).parent.parent
        output_path = base_dir / output_path

    # Display training parameters
    print("\n--- Training Parameters ---")
    print(f"  Dataset: {dataset_id}")
    print(f"  Policy: ACT (Action Chunking Transformer)")
    print(f"  Steps: {args.steps:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Device: {args.device}")
    print(f"  Output dir: {output_path}")
    print(f"  Job name: {job_name}")
    print(f"  Resume: {args.resume}")
    print(f"  Push to Hub: {args.push_to_hub}")
    print(f"  WandB: {args.wandb}")
    print("---------------------------\n")

    # Build lerobot-train command
    cmd = [
        "python",
        "-m",
        "lerobot.scripts.lerobot_train",
        f"--dataset.repo_id={dataset_id}",
        "--policy.type=act",
        f"--output_dir={output_path}",
        f"--job_name={job_name}",
        f"--policy.device={args.device}",
        f"--batch_size={args.batch_size}",
        f"--steps={args.steps}",
        "--dataset.video_backend=pyav",
    ]

    # Add optional arguments
    if args.resume:
        cmd.append("--resume=true")

    if args.push_to_hub:
        cmd.append(f"--policy.repo_id={job_name}")

    if args.wandb:
        cmd.append("--wandb.enable=true")
        cmd.append("--wandb.project=lerobot-panda-spacemouse")

    # Add config-based parameters if available
    if config:
        policy_cfg = config.get('policy', {})
        training_cfg = config.get('training', {})

        # Add policy-specific parameters
        if 'chunk_size' in policy_cfg:
            cmd.append(f"--policy.chunk_size={policy_cfg['chunk_size']}")
        if 'dim_model' in policy_cfg:
            cmd.append(f"--policy.dim_model={policy_cfg['dim_model']}")

        # Add training parameters
        if 'lr' in training_cfg:
            cmd.append(f"--training.lr={training_cfg['lr']}")
        if 'weight_decay' in training_cfg:
            cmd.append(f"--training.weight_decay={training_cfg['weight_decay']}")

    print("\n--- Starting Training ---")
    print(f"Command: {' '.join(cmd)}\n")

    input("Press Enter to start training (Ctrl+C to cancel)...")

    try:
        # Run training
        subprocess.run(cmd, check=True)

        print("\n" + "="*70)
        print("  Training Completed Successfully!")
        print("="*70)

        print(f"\n✓ Checkpoints saved to: {output_path}/checkpoints/")

        if args.push_to_hub:
            print(f"✓ Policy uploaded to: https://huggingface.co/{job_name}")

        print("\nNext steps:")
        print("  1. Evaluate policy: ./scripts/run_evaluation.sh")
        print(f"  2. View training logs: tensorboard --logdir {output_path}")
        if args.wandb:
            print("  3. View WandB dashboard: https://wandb.ai/")

    except subprocess.CalledProcessError as e:
        logging.error(f"\n✗ Training failed with error code {e.returncode}")
        logging.error("Check the logs above for details")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("\n✗ Training interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()

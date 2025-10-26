#!/usr/bin/env python3

"""
SpaceMouse Data Collection Script for PandaPickCubeSpacemouse-v0

This script collects demonstration data using SpaceMouse teleoperation in the
PandaPickCubeSpacemouse-v0 simulation environment using imitation learning.

Usage:
    python scripts/collect_data.py --config configs/collect_data_config.json
    python scripts/collect_data.py --config configs/collect_data_config.json --episodes 50
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path


def check_spacemouse_connection():
    """Check if SpaceMouse is connected."""
    try:
        import hid
        devices = hid.enumerate()
        spacemouse_found = any('3Dconnexion' in str(d) or 'SpaceMouse' in str(d) for d in devices)
        if spacemouse_found:
            logging.info("✓ SpaceMouse device detected")
            return True
        else:
            logging.warning("⚠ SpaceMouse device NOT detected")
            logging.warning("  Make sure your SpaceMouse is connected via USB")
            return False
    except ImportError:
        logging.warning("⚠ hidapi library not installed, cannot check SpaceMouse")
        logging.warning("  Install with: pip install hidapi")
        return True  # Continue anyway


def check_gpu_availability():
    """Check if GPU is available."""
    try:
        import torch
        if torch.cuda.is_available():
            logging.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
            return True
        else:
            logging.warning("⚠ GPU not available, will use CPU (slower)")
            return False
    except ImportError:
        logging.warning("⚠ PyTorch not installed")
        return False


def update_config(config_path, episodes=None, repo_id=None, device=None):
    """Update configuration file with command-line arguments."""
    with open(config_path, 'r') as f:
        config = json.load(f)

    if episodes is not None:
        config['dataset']['num_episodes_to_record'] = episodes
        logging.info(f"  Updated num_episodes to: {episodes}")

    if repo_id is not None:
        config['dataset']['repo_id'] = repo_id
        logging.info(f"  Updated repo_id to: {repo_id}")

    if device is not None:
        config['device'] = device
        logging.info(f"  Updated device to: {device}")

    # Save updated config to temporary file
    temp_config_path = config_path.parent / f"temp_{config_path.name}"
    with open(temp_config_path, 'w') as f:
        json.dump(config, f, indent=2)

    return temp_config_path, config


def main():
    """Main data collection function."""
    parser = argparse.ArgumentParser(
        description="Collect demonstration data using SpaceMouse teleoperation"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/collect_data_config.json",
        help="Path to data collection configuration file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to collect (overrides config)",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=['cuda', 'cpu', 'mps'],
        help="Device to use (overrides config)",
    )
    parser.add_argument(
        "--skip_checks",
        action="store_true",
        help="Skip pre-flight checks",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*70)
    print("  SpaceMouse Data Collection for PandaPickCubeSpacemouse-v0")
    print("="*70 + "\n")

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # Resolve relative to SPACEMOUSE_IL_DATA directory
        base_dir = Path(__file__).parent.parent
        config_path = base_dir / config_path

    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        sys.exit(1)

    logging.info(f"Using configuration: {config_path}")

    # Pre-flight checks
    if not args.skip_checks:
        logging.info("\n--- Pre-flight Checks ---")
        check_spacemouse_connection()
        check_gpu_availability()
        logging.info("-------------------------\n")

    # Update configuration if needed
    temp_config_path, config = update_config(
        config_path,
        episodes=args.episodes,
        repo_id=args.repo_id,
        device=args.device
    )

    # Display collection parameters
    print("\n--- Data Collection Parameters ---")
    print(f"  Environment: {config.get('env', {}).get('task', 'PandaPickCubeSpacemouse-v0')}")
    print(f"  Episodes: {config.get('dataset', {}).get('num_episodes_to_record', 30)}")
    print(f"  FPS: {config.get('env', {}).get('fps', 10)}")
    print(f"  Device: {config.get('device', 'cuda')}")
    print(f"  Repository: {config.get('dataset', {}).get('repo_id', 'YOUR_HF_USERNAME/dataset')}")
    print(f"  Push to Hub: {config.get('dataset', {}).get('push_to_hub', True)}")
    print("----------------------------------\n")

    # Check if repo_id needs to be updated
    repo_id = config.get('dataset', {}).get('repo_id', '')
    if repo_id.startswith('YOUR_HF_USERNAME'):
        print("⚠ WARNING: repo_id contains 'YOUR_HF_USERNAME'")
        print("  Please update the repo_id in the config file or use --repo_id flag")
        print("  Example: --repo_id username/panda_spacemouse_il_data\n")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            logging.info("Aborted by user")
            sys.exit(0)

    print("\n--- SpaceMouse Controls ---")
    print("  Movement: Move SpaceMouse in 3D space to control end-effector")
    print("  Rotation: Rotate SpaceMouse to control end-effector orientation")
    print("  Left Button: Close gripper")
    print("  Right Button: Open gripper")
    print("  SPACE: Mark episode as SUCCESS")
    print("  C: Mark episode as FAILURE")
    print("  R: Re-record current episode")
    print("---------------------------\n")

    input("Press Enter to start data collection...")

    # Build the command to run gym_manipulator
    cmd = [
        "python",
        "-m",
        "lerobot.rl.gym_manipulator",
        f"--config_path={temp_config_path.absolute()}"
    ]

    logging.info(f"Running command: {' '.join(cmd)}\n")

    try:
        # Run the data collection
        subprocess.run(cmd, check=True, cwd=config_path.parent.parent.parent)

        print("\n" + "="*70)
        print("  Data Collection Completed Successfully!")
        print("="*70)

        repo_id = config.get('dataset', {}).get('repo_id', '')
        push_to_hub = config.get('dataset', {}).get('push_to_hub', False)

        if push_to_hub and repo_id:
            print(f"\n✓ Dataset uploaded to: https://huggingface.co/datasets/{repo_id}")

        if repo_id:
            print(f"\n✓ Local dataset saved to: ~/.cache/huggingface/lerobot/{repo_id.split('/')[-1]}")

        print("\nNext steps:")
        print("  1. Verify dataset: Visit HuggingFace Hub link above")
        print("  2. Visualize dataset: Use lerobot dataset visualizer")
        print("  3. Train policy: Run scripts/train_policy.py")

    except subprocess.CalledProcessError as e:
        logging.error(f"\n✗ Data collection failed with error code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.warning("\n✗ Data collection interrupted by user")
        sys.exit(1)
    finally:
        # Clean up temporary config file
        if temp_config_path.exists():
            temp_config_path.unlink()


if __name__ == "__main__":
    main()

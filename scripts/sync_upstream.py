#!/usr/bin/env python3
"""
LeRobot Upstream Sync Script

This script provides a comprehensive solution to sync your local LeRobot repository
with the upstream original repository, handling merges between main and dev branches.

Features:
- Safe sync with backup capabilities
- Conflict detection and resolution assistance
- Branch-specific merge strategies
- Detailed logging and reporting
- Rollback functionality
- Multiple merge strategies available

Usage:
    python scripts/sync_upstream.py [OPTIONS]
"""

import argparse
import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("sync_upstream.log"), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


class GitRepo:
    """Git repository wrapper with enhanced functionality for LeRobot sync."""

    def __init__(self, repo_path: Path):
        self.repo_path = repo_path
        self.original_dir = Path.cwd()

    def _run_git_command(self, cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
        """Run git command and return result."""
        try:
            os.chdir(self.repo_path)
            logger.debug(f"Running: git {' '.join(cmd)}")
            result = subprocess.run(["git"] + cmd, capture_output=True, text=True, check=check)
            return result
        finally:
            os.chdir(self.original_dir)

    def _run_git_command_with_output(self, cmd: list[str]) -> str:
        """Run git command and return stdout."""
        result = self._run_git_command(cmd)
        return result.stdout.strip()

    def get_current_branch(self) -> str:
        """Get current branch name."""
        return self._run_git_command_with_output(["rev-parse", "--abbrev-ref", "HEAD"])

    def get_remote_url(self, remote: str = "origin") -> str:
        """Get remote URL."""
        try:
            return self._run_git_command_with_output(["remote", "get-url", remote])
        except subprocess.CalledProcessError:
            return ""

    def add_remote(self, name: str, url: str) -> bool:
        """Add remote if it doesn't exist."""
        try:
            self._run_git_command(["remote", "add", name, url])
            logger.info(f"Added remote '{name}': {url}")
            return True
        except subprocess.CalledProcessError:
            logger.info(f"Remote '{name}' already exists")
            return False

    def fetch_remote(self, remote: str) -> bool:
        """Fetch from remote."""
        try:
            logger.info(f"Fetching from remote '{remote}'...")
            result = self._run_git_command(["fetch", remote])
            return result.returncode == 0
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to fetch from '{remote}': {e}")
            return False

    def checkout_branch(self, branch: str, create: bool = False) -> bool:
        """Checkout branch, optionally creating it."""
        try:
            cmd = ["checkout", branch]
            if create:
                cmd.insert(1, "-b")

            self._run_git_command(cmd)
            logger.info(f"Checked out branch '{branch}'")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout branch '{branch}': {e}")
            return False

    def pull_branch(self, remote: str, branch: str, strategy: str = "merge") -> bool:
        """Pull branch from remote with specified strategy."""
        try:
            if strategy == "merge":
                cmd = ["pull", f"{remote}/{branch}"]
            elif strategy == "rebase":
                cmd = ["pull", "--rebase", f"{remote}/{branch}"]
            else:
                logger.error(f"Unknown pull strategy: {strategy}")
                return False

            logger.info(f"Pulling '{branch}' from '{remote}' using {strategy} strategy...")
            result = self._run_git_command(cmd, check=False)

            if result.returncode != 0:
                logger.error(f"Pull failed: {result.stderr}")
                return False

            return True
        except Exception as e:
            logger.error(f"Exception during pull: {e}")
            return False

    def has_conflicts(self) -> bool:
        """Check if repository has merge conflicts."""
        try:
            result = self._run_git_command(["diff", "--name-only", "--diff-filter=U"])
            return bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

    def get_conflicted_files(self) -> list[str]:
        """Get list of conflicted files."""
        try:
            result = self._run_git_command(["diff", "--name-only", "--diff-filter=U"])
            return [f.strip() for f in result.stdout.split("\n") if f.strip()]
        except subprocess.CalledProcessError:
            return []

    def get_commit_hash(self, ref: str = "HEAD") -> str:
        """Get commit hash for reference."""
        return self._run_git_command_with_output(["rev-parse", ref])

    def is_clean(self) -> bool:
        """Check if working directory is clean."""
        try:
            result = self._run_git_command(["status", "--porcelain"])
            return not bool(result.stdout.strip())
        except subprocess.CalledProcessError:
            return False

    def get_untracked_files(self) -> list[str]:
        """Get list of untracked files."""
        try:
            result = self._run_git_command(["ls-files", "--others", "--exclude-standard"])
            return [f.strip() for f in result.stdout.split("\n") if f.strip()]
        except subprocess.CalledProcessError:
            return []

    def stash_changes(self) -> bool:
        """Stash current changes."""
        try:
            self._run_git_command(["stash", "push", "-m", f"Sync script stash {datetime.now().isoformat()}"])
            logger.info("Stashed current changes")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to stash changes: {e}")
            return False

    def stash_pop(self) -> bool:
        """Pop stashed changes."""
        try:
            self._run_git_command(["stash", "pop"])
            logger.info("Popped stashed changes")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to pop stashed changes: {e}")
            return False

    def create_backup_branch(self, base_branch: str) -> str:
        """Create backup branch for current state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_branch = f"backup_{base_branch}_{timestamp}"

        try:
            self._run_git_command(["branch", backup_branch, "HEAD"])
            logger.info(f"Created backup branch: {backup_branch}")
            return backup_branch
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create backup branch: {e}")
            return ""

    def get_branch_diff(self, branch1: str, branch2: str) -> str:
        """Get diff summary between two branches."""
        try:
            result = self._run_git_command(["diff", "--stat", f"{branch1}...{branch2}"])
            return result.stdout.strip()
        except subprocess.CalledProcessError:
            return ""

    def get_ahead_behind(self, branch: str, remote_branch: str) -> tuple[int, int]:
        """Get ahead/behind count for branch."""
        try:
            result = self._run_git_command(
                ["rev-list", "--left-right", "--count", f"{remote_branch}...{branch}"]
            )
            ahead, behind = map(int, result.stdout.strip().split("\t"))
            return ahead, behind
        except subprocess.CalledProcessError:
            return 0, 0


class LeRobotSyncManager:
    """Main sync manager for LeRobot repository."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path).resolve()
        self.repo = GitRepo(self.repo_path)
        self.config_file = self.repo_path / "sync_config.json"
        self.default_config = {
            "upstream_remote": "upstream",
            "upstream_url": "https://github.com/huggingface/lerobot.git",
            "main_branch": "main",
            "dev_branch": "dev",
            "merge_strategy": "merge",  # "merge" or "rebase"
            "auto_backup": True,
            "auto_stash": True,
            "skip_conflicts": False,
        }

    def load_config(self) -> dict:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file) as f:
                    config = json.load(f)
                # Merge with defaults
                return {**self.default_config, **config}
            except Exception as e:
                logger.warning(f"Failed to load config: {e}")

        return self.default_config.copy()

    def save_config(self, config: dict) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved configuration to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")

    def validate_repo(self) -> bool:
        """Validate repository state before sync."""
        logger.info("Validating repository state...")

        # Check if we're in a git repository
        try:
            self.repo._run_git_command(["status"])
        except subprocess.CalledProcessError:
            logger.error("Not in a git repository!")
            return False

        # Check working directory
        if not self.repo.is_clean():
            logger.warning("Working directory is not clean!")

            untracked = self.repo.get_untracked_files()
            if untracked:
                logger.warning(f"Untracked files: {len(untracked)}")

            if not self.config.get("auto_stash", False):
                logger.error("Please commit or stash changes before sync, or enable auto_stash")
                return False

        return True

    def setup_remotes(self) -> bool:
        """Setup required remotes."""
        logger.info("Setting up remotes...")

        config = self.config

        # Add upstream remote if it doesn't exist
        upstream_url = config["upstream_url"]
        upstream_remote = config["upstream_remote"]

        current_upstream_url = self.repo.get_remote_url(upstream_remote)
        if (not current_upstream_url or current_upstream_url != upstream_url) and self.repo.add_remote(
            upstream_remote, upstream_url
        ):
            logger.info(f"Configured upstream remote: {upstream_remote} -> {upstream_url}")

        return True

    def sync_branch(self, target_branch: str, remote_branch: str) -> bool:
        """Sync a specific branch from upstream."""
        logger.info(f"Syncing branch '{target_branch}' from '{remote_branch}'...")

        config = self.config

        # Create backup if enabled
        backup_branch = ""
        if config.get("auto_backup", True):
            backup_branch = self.repo.create_backup_branch(target_branch)

        try:
            # Checkout target branch
            if not self.repo.checkout_branch(target_branch):
                logger.error(f"Failed to checkout branch '{target_branch}'")
                return False

            # Stash changes if needed and enabled
            stashed = False
            if not self.repo.is_clean() and config.get("auto_stash", False):
                stashed = self.repo.stash_changes()

            # Pull changes with configured strategy
            merge_strategy = config.get("merge_strategy", "merge")
            if not self.repo.pull_branch(config["upstream_remote"], remote_branch, merge_strategy):
                logger.error(f"Failed to sync branch '{target_branch}'")

                # Try to restore from backup if available
                if backup_branch:
                    logger.info(f"Attempting to restore from backup branch: {backup_branch}")
                    self.repo._run_git_command(["reset", "--hard", backup_branch])

                return False

            # Check for conflicts
            if self.repo.has_conflicts():
                logger.warning(f"Merge conflicts detected in '{target_branch}'")
                conflicted_files = self.repo.get_conflicted_files()
                logger.warning(f"Conflicted files: {conflicted_files}")

                if config.get("skip_conflicts", False):
                    logger.warning("Skipping conflicts as per configuration")
                    return True
                else:
                    logger.error("Please resolve conflicts manually")
                    return False

            # Pop stashed changes
            if stashed:
                self.repo.stash_pop()

            logger.info(f"Successfully synced branch '{target_branch}'")
            return True

        except Exception as e:
            logger.error(f"Exception during branch sync: {e}")
            return False

    def sync_all_branches(self) -> bool:
        """Sync all configured branches."""
        config = self.config

        # Sync main branch first
        logger.info("=" * 50)
        logger.info("Syncing main branch")
        logger.info("=" * 50)

        if not self.sync_branch(config["main_branch"], config["main_branch"]):
            logger.error("Failed to sync main branch")
            return False

        # Sync dev branch
        logger.info("=" * 50)
        logger.info("Syncing dev branch")
        logger.info("=" * 50)

        if not self.sync_branch(config["dev_branch"], config["dev_branch"]):
            logger.error("Failed to sync dev branch")
            return False

        return True

    def generate_report(self) -> dict:
        """Generate sync report."""
        config = self.config
        current_branch = self.repo.get_current_branch()

        report = {
            "timestamp": datetime.now().isoformat(),
            "current_branch": current_branch,
            "main_branch": config["main_branch"],
            "dev_branch": config["dev_branch"],
            "upstream_remote": config["upstream_remote"],
            "upstream_url": config["upstream_url"],
            "branches": {},
        }

        # Get status for main and dev branches
        for branch in [config["main_branch"], config["dev_branch"]]:
            remote_branch = f"{config['upstream_remote']}/{branch}"

            # Get ahead/behind counts
            ahead, behind = self.repo.get_ahead_behind(branch, remote_branch)

            report["branches"][branch] = {"ahead": ahead, "behind": behind, "remote_branch": remote_branch}

            # Get diff summary if there are differences
            if ahead > 0 or behind > 0:
                report["branches"][branch]["diff_summary"] = self.repo.get_branch_diff(branch, remote_branch)

        return report

    def run_sync(self, dry_run: bool = False) -> bool:
        """Run the complete sync process."""
        logger.info("Starting LeRobot upstream sync...")

        # Load configuration
        self.config = self.load_config()

        # Validate repository
        if not self.validate_repo():
            return False

        # Setup remotes
        if not self.setup_remotes():
            return False

        # Fetch from upstream
        if not self.repo.fetch_remote(self.config["upstream_remote"]):
            logger.error("Failed to fetch from upstream")
            return False

        if dry_run:
            logger.info("Dry run completed - would sync with the above configuration")
            return True

        # Sync branches
        success = self.sync_all_branches()

        if success:
            logger.info("Sync completed successfully!")

            # Generate and save report
            report = self.generate_report()
            report_file = self.repo_path / f"sync_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            try:
                with open(report_file, "w") as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Sync report saved to: {report_file}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
        else:
            logger.error("Sync failed!")

        return success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Sync LeRobot repository with upstream",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic sync
    python scripts/sync_upstream.py

    # Dry run to see what would happen
    python scripts/sync_upstream.py --dry-run

    # Use rebase strategy instead of merge
    python scripts/sync_upstream.py --strategy rebase

    # Sync specific branch only
    python scripts/sync_upstream.py --branch main

    # Custom configuration
    python scripts/sync_upstream.py --config custom_config.json
        """,
    )

    parser.add_argument(
        "--repo-path", "-p", default=".", help="Path to repository (default: current directory)"
    )

    parser.add_argument("--dry-run", "-n", action="store_true", help="Perform dry run without making changes")

    parser.add_argument(
        "--strategy",
        "-s",
        choices=["merge", "rebase"],
        default="merge",
        help="Merge strategy (default: merge)",
    )

    parser.add_argument("--branch", "-b", help="Sync specific branch only (main, dev, or custom)")

    parser.add_argument("--config", "-c", type=Path, help="Path to configuration file")

    parser.add_argument("--upstream-url", "-u", help="Custom upstream repository URL")

    parser.add_argument("--no-backup", action="store_true", help="Skip automatic backup creation")

    parser.add_argument("--no-stash", action="store_true", help="Skip automatic stashing of changes")

    parser.add_argument("--skip-conflicts", action="store_true", help="Skip branches with conflicts")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize sync manager
    sync_manager = LeRobotSyncManager(args.repo_path)

    # Load and update configuration
    config = sync_manager.load_config()

    if args.config:
        # Load custom config
        if args.config.exists():
            with open(args.config) as f:
                custom_config = json.load(f)
            config.update(custom_config)
        else:
            logger.error(f"Config file not found: {args.config}")
            return 1

    # Update config with command line arguments
    if args.strategy:
        config["merge_strategy"] = args.strategy

    if args.upstream_url:
        config["upstream_url"] = args.upstream_url

    if args.no_backup:
        config["auto_backup"] = False

    if args.no_stash:
        config["auto_stash"] = False

    if args.skip_conflicts:
        config["skip_conflicts"] = True

    # Save updated config
    sync_manager.config = config
    sync_manager.save_config(config)

    # Run sync
    if args.branch:
        # Sync specific branch only
        logger.info(f"Syncing specific branch: {args.branch}")
        config = sync_manager.config

        if not sync_manager.validate_repo():
            return 1

        if not sync_manager.setup_remotes():
            return 1

        if not sync_manager.repo.fetch_remote(config["upstream_remote"]):
            return 1

        if not args.dry_run and not sync_manager.sync_branch(args.branch, args.branch):
            logger.error("Failed to sync branch")
            return 1
    else:
        # Sync all branches
        success = sync_manager.run_sync(args.dry_run)
        return 0 if success else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

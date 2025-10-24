#!/bin/bash

# LeRobot Sync Script
# This script fetches updates from the original LeRobot repository
# and merges them into your main and dev branches

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not in a git repository"
    exit 1
fi

# Store current branch
CURRENT_BRANCH=$(git branch --show-current)
print_status "Current branch: $CURRENT_BRANCH"

# Function to add upstream remote if it doesn't exist
add_upstream_remote() {
    print_status "Setting up upstream remote..."

    if ! git remote | grep -q "upstream"; then
        print_status "Adding upstream remote..."
        git remote add upstream https://github.com/huggingface/lerobot.git
        print_success "Added upstream remote"
    else
        print_success "Upstream remote already exists"
    fi

    # Verify upstream remote
    UPSTREAM_URL=$(git remote get-url upstream)
    print_status "Upstream URL: $UPSTREAM_URL"
}

# Function to fetch latest changes from upstream
fetch_upstream() {
    print_status "Fetching latest changes from upstream..."
    git fetch upstream
    print_success "Fetched latest changes from upstream"
}

# Function to sync a branch
sync_branch() {
    local branch_name=$1
    print_status "Syncing branch: $branch_name"

    # Checkout the branch
    git checkout "$branch_name"

    # Ensure it's up to date with origin
    git pull origin "$branch_name" || print_warning "Could not pull from origin (branch might not exist remotely)"

    # Merge upstream changes
    print_status "Merging upstream/main into $branch_name..."
    if git merge upstream/main --no-edit; then
        print_success "Successfully merged upstream/main into $branch_name"
    else
        print_error "Merge failed for $branch_name"
        print_status "Attempting to abort merge..."
        git merge --abort
        return 1
    fi

    # Push changes to origin
    print_status "Pushing changes to origin/$branch_name..."
    git push origin "$branch_name"
    print_success "Pushed changes to origin/$branch_name"
}

# Main execution
main() {
    print_status "Starting LeRobot sync process..."

    # Add upstream remote
    add_upstream_remote

    # Fetch upstream changes
    fetch_upstream

    # Store current branch to return to it later
    ORIGINAL_BRANCH=$(git branch --show-current)

    # Sync main branch
    print_status "=== Syncing main branch ==="
    if sync_branch "main"; then
        print_success "Main branch synced successfully"
    else
        print_error "Failed to sync main branch"
        exit 1
    fi

    # Sync dev branch
    print_status "=== Syncing dev branch ==="
    if sync_branch "dev"; then
        print_success "Dev branch synced successfully"
    else
        print_error "Failed to sync dev branch"
        exit 1
    fi

    # Return to original branch
    print_status "Returning to original branch: $ORIGINAL_BRANCH"
    git checkout "$ORIGINAL_BRANCH"

    print_success "Sync process completed successfully!"
    print_status "Both main and dev branches have been updated with latest changes from upstream LeRobot repository"
}

# Handle script interruption
trap 'print_error "Script interrupted"; exit 1' INT

# Run main function
main "$@"

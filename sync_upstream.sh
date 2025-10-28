#!/bin/bash

# Script to sync with upstream repository (huggingface/lerobot)
# This script fetches latest changes from upstream and updates your branches

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

# Function to check if we're in a git repository
check_git_repo() {
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository!"
        exit 1
    fi
}

# Function to check if upstream remote exists
check_upstream() {
    if ! git remote get-url upstream > /dev/null 2>&1; then
        print_error "Upstream remote not found!"
        print_error "Please make sure you have 'upstream' pointing to huggingface/lerobot"
        exit 1
    fi
}

# Function to check if origin remote exists
check_origin() {
    if ! git remote get-url origin > /dev/null 2>&1; then
        print_error "Origin remote not found!"
        print_error "Please make sure you have 'origin' pointing to your fork"
        exit 1
    fi
}

# Function to get current branch
get_current_branch() {
    git branch --show-current
}

# Function to stash changes if any
stash_changes() {
    if ! git diff-index --quiet HEAD --; then
        print_warning "You have uncommitted changes. Stashing them..."
        git stash push -m "Auto-stash before sync with upstream $(date)"
        return 0
    fi
    return 1
}

# Function to restore stashed changes
restore_stash() {
    if git stash list | grep -q "Auto-stash before sync with upstream"; then
        print_status "Restoring stashed changes..."
        git stash pop
    fi
}

# Function to sync a branch with upstream
sync_branch() {
    local branch=$1
    local upstream_branch=${2:-$branch}
    
    # For dev branch, always sync with upstream/main since upstream doesn't have dev
    if [ "$branch" = "dev" ]; then
        upstream_branch="main"
    fi
    
    print_status "Syncing branch '$branch' with upstream/$upstream_branch..."
    
    # Switch to the branch
    git checkout "$branch"
    
    # Check if upstream branch exists
    if ! git show-ref --verify --quiet "refs/remotes/upstream/$upstream_branch"; then
        print_error "Upstream branch 'upstream/$upstream_branch' does not exist!"
        print_error "Available upstream branches:"
        git branch -r | grep upstream | sed 's/^/  /'
        return 1
    fi
    
    # Fetch latest from upstream
    git fetch upstream "$upstream_branch"
    
    # Check if there are any new commits
    local behind=$(git rev-list --count HEAD..upstream/"$upstream_branch" 2>/dev/null || echo "0")
    local ahead=$(git rev-list --count upstream/"$upstream_branch"..HEAD 2>/dev/null || echo "0")
    
    if [ "$behind" -eq 0 ]; then
        print_success "Branch '$branch' is already up to date with upstream/$upstream_branch"
        return 0
    fi
    
    print_status "Branch '$branch' is $behind commits behind upstream/$upstream_branch"
    if [ "$ahead" -gt 0 ]; then
        print_warning "Branch '$branch' is $ahead commits ahead of upstream/$upstream_branch"
    fi
    
    # Merge upstream changes
    if git merge upstream/"$upstream_branch" --no-edit; then
        print_success "Successfully merged upstream/$upstream_branch into $branch"
        
        # Push to origin (your fork)
        if git push origin "$branch"; then
            print_success "Successfully pushed $branch to origin"
        else
            print_error "Failed to push $branch to origin"
            return 1
        fi
    else
        print_error "Failed to merge upstream/$upstream_branch into $branch"
        print_error "Please resolve conflicts manually and run: git push origin $branch"
        return 1
    fi
}

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS] [BRANCHES...]"
    echo ""
    echo "Sync your local repository with upstream (huggingface/lerobot)"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help     Show this help message"
    echo "  -a, --all      Sync all branches (main, dev)"
    echo "  -m, --main     Sync only main branch"
    echo "  -d, --dev      Sync only dev branch (syncs with upstream/main)"
    echo "  -f, --force    Force sync even if there are uncommitted changes"
    echo "  -q, --quiet    Quiet mode (less output)"
    echo ""
    echo "EXAMPLES:"
    echo "  $0                    # Sync current branch with upstream"
    echo "  $0 --all             # Sync main and dev branches"
    echo "  $0 --main            # Sync only main branch"
    echo "  $0 main dev          # Sync specific branches"
    echo ""
    echo "BRANCHES:"
    echo "  You can specify which branches to sync as arguments"
    echo "  If no branches are specified, only the current branch is synced"
}

# Main function
main() {
    local sync_all=false
    local sync_main=false
    local sync_dev=false
    local force=false
    local quiet=false
    local branches=()
    local current_branch
    local stashed=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -a|--all)
                sync_all=true
                shift
                ;;
            -m|--main)
                sync_main=true
                shift
                ;;
            -d|--dev)
                sync_dev=true
                shift
                ;;
            -f|--force)
                force=true
                shift
                ;;
            -q|--quiet)
                quiet=true
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                branches+=("$1")
                shift
                ;;
        esac
    done
    
    # Check if we're in a git repository
    check_git_repo
    check_upstream
    check_origin
    
    # Get current branch
    current_branch=$(get_current_branch)
    
    # Handle stashing
    if [ "$force" = false ]; then
        if stash_changes; then
            stashed=true
        fi
    fi
    
    # Determine which branches to sync
    if [ "$sync_all" = true ]; then
        branches=("main" "dev")
    elif [ "$sync_main" = true ]; then
        branches=("main")
    elif [ "$sync_dev" = true ]; then
        branches=("dev")
    elif [ ${#branches[@]} -eq 0 ]; then
        branches=("$current_branch")
    fi
    
    print_status "Starting sync with upstream repository..."
    print_status "Branches to sync: ${branches[*]}"
    
    # Sync each branch
    local success=true
    for branch in "${branches[@]}"; do
        if ! git show-ref --verify --quiet "refs/heads/$branch"; then
            print_warning "Branch '$branch' does not exist locally, skipping..."
            continue
        fi
        
        if ! sync_branch "$branch"; then
            success=false
        fi
    done
    
    # Restore stashed changes
    if [ "$stashed" = true ]; then
        restore_stash
    fi
    
    # Return to original branch if it changed
    if [ "$current_branch" != "$(get_current_branch)" ]; then
        git checkout "$current_branch"
    fi
    
    if [ "$success" = true ]; then
        print_success "Sync completed successfully!"
    else
        print_error "Sync completed with errors. Please check the output above."
        exit 1
    fi
}

# Run main function with all arguments
main "$@"

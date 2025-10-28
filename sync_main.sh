#!/bin/bash

# Simple script to sync only with upstream main branch
# This script only syncs your main branch with upstream/main

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

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Sync your main branch with upstream (huggingface/lerobot)"
    echo ""
    echo "OPTIONS:"
    echo "  -h, --help     Show this help message"
    echo "  -f, --force    Force sync even if there are uncommitted changes"
    echo ""
    echo "This script will:"
    echo "  1. Switch to main branch"
    echo "  2. Fetch latest changes from upstream/main"
    echo "  3. Merge upstream/main into your main branch"
    echo "  4. Push updated main branch to your fork"
}

# Main function
main() {
    local force=false
    local stashed=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                show_help
                exit 0
                ;;
            -f|--force)
                force=true
                shift
                ;;
            -*)
                print_error "Unknown option: $1"
                show_help
                exit 1
                ;;
            *)
                print_error "Unexpected argument: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository!"
        exit 1
    fi
    
    # Check if upstream remote exists
    if ! git remote get-url upstream > /dev/null 2>&1; then
        print_error "Upstream remote not found!"
        print_error "Please make sure you have 'upstream' pointing to huggingface/lerobot"
        exit 1
    fi
    
    # Check if origin remote exists
    if ! git remote get-url origin > /dev/null 2>&1; then
        print_error "Origin remote not found!"
        print_error "Please make sure you have 'origin' pointing to your fork"
        exit 1
    fi
    
    # Handle stashing
    if [ "$force" = false ]; then
        if ! git diff-index --quiet HEAD --; then
            print_warning "You have uncommitted changes. Stashing them..."
            git stash push -m "Auto-stash before sync with upstream $(date)"
            stashed=true
        fi
    fi
    
    print_status "Starting sync with upstream main branch..."
    
    # Switch to main branch
    git checkout main
    
    # Fetch latest from upstream
    print_status "Fetching latest changes from upstream/main..."
    git fetch upstream main
    
    # Check if there are any new commits
    local behind=$(git rev-list --count HEAD..upstream/main 2>/dev/null || echo "0")
    local ahead=$(git rev-list --count upstream/main..HEAD 2>/dev/null || echo "0")
    
    if [ "$behind" -eq 0 ]; then
        print_success "Your main branch is already up to date with upstream/main"
    else
        print_status "Your main branch is $behind commits behind upstream/main"
        if [ "$ahead" -gt 0 ]; then
            print_warning "Your main branch is $ahead commits ahead of upstream/main"
        fi
        
        # Merge upstream changes
        print_status "Merging upstream/main into your main branch..."
        if git merge upstream/main --no-edit; then
            print_success "Successfully merged upstream/main into main"
            
            # Push to origin (your fork)
            print_status "Pushing updated main branch to your fork..."
            if git push origin main; then
                print_success "Successfully pushed main to origin"
            else
                print_error "Failed to push main to origin"
                exit 1
            fi
        else
            print_error "Failed to merge upstream/main into main"
            print_error "Please resolve conflicts manually and run: git push origin main"
            exit 1
        fi
    fi
    
    # Restore stashed changes
    if [ "$stashed" = true ]; then
        print_status "Restoring stashed changes..."
        git stash pop
    fi
    
    print_success "Sync completed successfully!"
}

# Run main function with all arguments
main "$@"

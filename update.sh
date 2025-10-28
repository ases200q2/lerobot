#!/bin/bash

# Simple alias script for syncing with upstream
# This is a shortcut for the main sync script

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the main sync script with all arguments
"$SCRIPT_DIR/sync_upstream.sh" "$@"

#!/bin/bash

# Define the environment and the path to search
ENV_PATH="/cluster/home/t144807uhn/envs/methylvae-env/bin/activate"
SEARCH_PATH="/cluster/home/t144807uhn/tmp/"

# 1. Activate the environment
source "$ENV_PATH"

# 2. Find and sync all offline runs
echo "Starting synchronization..."
for run_dir in $(find "$SEARCH_PATH" -type d -name "offline-run-*"); do
    echo "Syncing: $run_dir"
    wandb sync "$run_dir"
done

echo "All syncs completed."
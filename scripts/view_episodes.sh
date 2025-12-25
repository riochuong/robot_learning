#!/bin/bash
# Simple wrapper for viewing local dataset episodes
# Usage: ./view_episodes.sh [dataset_name] [start_episode]

DATASET=${1:-"data/pick_small_cube_1_20eps"}
START_EP=${2:-0}

echo "🎬 LeRobot Local Episode Viewer"
echo "================================"
echo "Dataset: $DATASET"
echo "Starting from episode: $START_EP"
echo ""
echo "Controls:"
echo "  - [Enter] = Next episode"
echo "  - [s] = Skip to episode number"
echo "  - [q] = Quit"
echo ""
echo "Close Rerun window after each episode to continue"
echo "================================"
echo ""

HF_HUB_OFFLINE=1 uv run python view_dataset_local.py "$DATASET" "$START_EP" --all


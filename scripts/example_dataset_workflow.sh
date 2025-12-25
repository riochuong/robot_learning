#!/bin/bash
# Example workflow: Clean and prepare dataset for training
# 
# This demonstrates a complete workflow from recording to training-ready dataset

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║            DATASET PREPARATION WORKFLOW EXAMPLE                    ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"
echo

# Configuration
DATASET_NAME="data/pick_small_cube_1_20eps"
CLEANED_NAME="data/pick_small_cube_cleaned"
BAD_EPISODES="[5, 12, 18]"  # Example: episodes to remove

# Step 1: Verify original dataset
echo "1️⃣  Verifying original dataset..."
echo "─────────────────────────────────────────────────────────────────────"
uv run python verify_dataset.py "$DATASET_NAME" || {
    echo "❌ Original dataset verification failed!"
    exit 1
}
echo

# Step 2: View episodes to identify bad ones (manual step)
echo "2️⃣  View episodes to identify bad demonstrations"
echo "─────────────────────────────────────────────────────────────────────"
echo "Run this command to review all episodes:"
echo "  uv run python view_dataset_local.py $DATASET_NAME --all"
echo
echo "Press Enter when you've identified bad episodes to remove..."
read -r

# Step 3: Remove bad episodes
echo "3️⃣  Removing bad episodes..."
echo "─────────────────────────────────────────────────────────────────────"
echo "Deleting episodes: $BAD_EPISODES"
lerobot-edit-dataset \
    --repo_id "$DATASET_NAME" \
    --new_repo_id "$CLEANED_NAME" \
    --operation.type delete_episodes \
    --operation.episode_indices "$BAD_EPISODES"
echo "✅ Bad episodes removed"
echo

# Step 4: Verify cleaned dataset
echo "4️⃣  Verifying cleaned dataset..."
echo "─────────────────────────────────────────────────────────────────────"
uv run python verify_dataset.py "$CLEANED_NAME" || {
    echo "❌ Cleaned dataset verification failed!"
    exit 1
}
echo

# Step 5: Split into train/val
echo "5️⃣  Splitting dataset into train (80%) and validation (20%)..."
echo "─────────────────────────────────────────────────────────────────────"
lerobot-edit-dataset \
    --repo_id "$CLEANED_NAME" \
    --operation.type split \
    --operation.splits '{"train": 0.8, "val": 0.2}'
echo "✅ Dataset split complete"
echo

# Step 6: Verify train and val datasets
echo "6️⃣  Verifying split datasets..."
echo "─────────────────────────────────────────────────────────────────────"
echo
echo "Training set:"
uv run python verify_dataset.py "${CLEANED_NAME}_train"
echo
echo "Validation set:"
uv run python verify_dataset.py "${CLEANED_NAME}_val"
echo

# Summary
echo "╔════════════════════════════════════════════════════════════════════╗"
echo "║                                                                    ║"
echo "║                    ✅ WORKFLOW COMPLETE                            ║"
echo "║                                                                    ║"
echo "╠════════════════════════════════════════════════════════════════════╣"
echo "║                                                                    ║"
echo "║  Your datasets are ready for training:                             ║"
echo "║                                                                    ║"
echo "║  📁 Training:   ${CLEANED_NAME}_train"
echo "║  📁 Validation: ${CLEANED_NAME}_val"
echo "║                                                                    ║"
echo "║  Next steps:                                                       ║"
echo "║  1. Review train and val datasets with view_dataset_local.py      ║"
echo "║  2. Configure training script with these paths                     ║"
echo "║  3. Start training!                                                ║"
echo "║                                                                    ║"
echo "╚════════════════════════════════════════════════════════════════════╝"


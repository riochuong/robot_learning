# LeRobot Dataset Editing Guide

> Complete guide to editing LeRobot datasets using built-in tools

**LeRobot Version:** v3.0  
**Tool:** `lerobot-edit-dataset`

---

## Table of Contents

1. [Overview](#overview)
2. [Delete Episodes](#delete-episodes)
3. [Split Datasets](#split-datasets)
4. [Merge Datasets](#merge-datasets)
5. [Remove Features](#remove-features)
6. [Convert Image to Video Format](#convert-image-to-video-format)
7. [Best Practices](#best-practices)
8. [Common Use Cases](#common-use-cases)

---

## Overview

LeRobot provides the `lerobot-edit-dataset` command-line tool for dataset manipulation:

```bash
# General syntax
lerobot-edit-dataset \
    --repo_id dataset/my_dataset \
    --operation.type <operation_type> \
    [operation-specific arguments]
```

### Key Features

✅ **Non-destructive by default** - Creates new datasets, preserves originals  
✅ **Efficient** - Only re-encodes modified video chunks  
✅ **Complete** - Updates all metadata automatically  
✅ **Flexible** - Supports local and HuggingFace Hub datasets  

### Important Notes

- By default, **modifies the original dataset** (creates `_old` backup)
- Use `--new_repo_id` to create a new dataset and preserve the original
- All operations update episode indices, video mappings, and metadata automatically

---

## Delete Episodes

Remove specific episodes from a dataset (e.g., bad demonstrations, corrupted data).

### Basic Usage

```bash
# Delete episodes and modify original (creates backup)
lerobot-edit-dataset \
    --repo_id dataset/pick_and_place_small_cube \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 5, 15]"
```

### Create New Dataset (Preserve Original)

```bash
# Delete episodes and save to new dataset
lerobot-edit-dataset \
    --repo_id dataset/pick_and_place_small_cube \
    --new_repo_id dataset/pick_and_place_cleaned \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 5, 15]"
```

### Local Dataset

```bash
# For local datasets (not on HuggingFace Hub)
lerobot-edit-dataset \
    --repo_id data/my_dataset \
    --root ~/.cache/huggingface/lerobot \
    --operation.type delete_episodes \
    --operation.episode_indices "[1, 3, 7]"
```

### Python API

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes

# Load dataset
dataset = LeRobotDataset("dataset/my_dataset")

# Delete episodes
new_dataset = delete_episodes(
    dataset,
    episode_indices=[0, 5, 15],
    output_dir="path/to/output",
    repo_id="dataset/my_dataset_cleaned"
)

print(f"Original episodes: {dataset.meta.total_episodes}")
print(f"New episodes: {new_dataset.meta.total_episodes}")
```

### What Happens

1. **Creates new dataset** with remaining episodes
2. **Re-indexes episodes** (0, 1, 2, ... for kept episodes)
3. **Copies data parquet files** (filtering deleted episodes)
4. **Re-encodes videos** (only chunks containing deleted episodes)
5. **Updates all metadata** (info.json, episode metadata, stats)
6. **Preserves video quality** (only re-encodes affected chunks)

---

## Split Datasets

Split a dataset into train/val/test sets for model evaluation.

### Split by Fraction

```bash
# Split into 80% train, 20% val
lerobot-edit-dataset \
    --repo_id dataset/my_dataset \
    --operation.type split \
    --operation.splits '{"train": 0.8, "val": 0.2}'
```

### Split into Three Sets

```bash
# 60% train, 20% val, 20% test
lerobot-edit-dataset \
    --repo_id dataset/my_dataset \
    --operation.type split \
    --operation.splits '{"train": 0.6, "val": 0.2, "test": 0.2}'
```

### Split by Episode Indices

```bash
# Specify exact episodes for each split
lerobot-edit-dataset \
    --repo_id dataset/my_dataset \
    --operation.type split \
    --operation.splits '{"train": [0, 1, 2, 3, 4, 5], "val": [6, 7], "test": [8, 9]}'
```

### Python API

```python
from lerobot.datasets.dataset_tools import split_dataset

splits = split_dataset(
    dataset,
    splits={"train": 0.8, "val": 0.2},
    output_dir="path/to/output"
)

train_dataset = splits["train"]
val_dataset = splits["val"]

print(f"Train episodes: {train_dataset.meta.total_episodes}")
print(f"Val episodes: {val_dataset.meta.total_episodes}")
```

### What Happens

1. **Creates multiple new datasets** (one per split)
2. **Saves to**: `dataset/my_dataset_train`, `dataset/my_dataset_val`, etc.
3. **Each split is a complete dataset** with its own metadata
4. **Episodes are re-indexed** starting from 0 in each split

---

## Merge Datasets

Combine multiple datasets into one (e.g., merge data from different recording sessions).

### Basic Usage

```bash
# Merge multiple datasets
lerobot-edit-dataset \
    --repo_id dataset/merged_dataset \
    --operation.type merge \
    --operation.repo_ids "['dataset/session1', 'dataset/session2', 'dataset/session3']"
```

### With Different Roots

```bash
# Merge datasets from different locations
lerobot-edit-dataset \
    --repo_id dataset/merged_dataset \
    --operation.type merge \
    --operation.repo_ids "['data/run1', 'data/run2']" \
    --root ~/.cache/huggingface/lerobot
```

### Python API

```python
from lerobot.datasets.dataset_tools import merge_datasets

merged = merge_datasets(
    repo_ids=["dataset/session1", "dataset/session2"],
    output_dir="path/to/merged",
    output_repo_id="dataset/merged_dataset"
)

print(f"Total episodes: {merged.meta.total_episodes}")
```

### Requirements

✅ **Same robot type** - All datasets must be from same robot  
✅ **Same FPS** - Frame rates must match  
✅ **Compatible features** - Action/observation spaces must align  

### What Happens

1. **Concatenates all episodes** from all datasets
2. **Re-indexes episodes** continuously (0, 1, 2, ...)
3. **Copies all data and videos**
4. **Merges statistics** (min/max/mean/std for normalization)
5. **Preserves tasks** if multi-task datasets

---

## Remove Features

Remove camera feeds or other features from datasets (e.g., to reduce size or remove unused cameras).

### Remove Camera

```bash
# Remove wrist camera
lerobot-edit-dataset \
    --repo_id dataset/my_dataset \
    --new_repo_id dataset/my_dataset_no_wrist \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.wrist']"
```

### Remove Multiple Features

```bash
# Remove multiple cameras or sensors
lerobot-edit-dataset \
    --repo_id dataset/my_dataset \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.wrist', 'observation.images.top']"
```

### Python API

```python
from lerobot.datasets.dataset_tools import remove_feature

new_dataset = remove_feature(
    dataset,
    feature_names=["observation.images.wrist"],
    output_dir="path/to/output",
    repo_id="dataset/my_dataset_modified"
)
```

### What Happens

1. **Removes feature from metadata** (info.json)
2. **Deletes video directories** for removed cameras
3. **Updates episode metadata** (removes video mapping columns)
4. **Data parquet unchanged** (no video columns anyway)
5. **Dataset size reduced** significantly if cameras removed

---

## Convert Image to Video Format

Convert old image-format datasets to modern video format (more efficient storage).

### Basic Conversion

```bash
# Convert and save to new dataset
lerobot-edit-dataset \
    --repo_id dataset/old_image_dataset \
    --new_repo_id dataset/converted_video \
    --operation.type convert_to_video
```

### Custom Output Directory

```bash
# Specify output directory
lerobot-edit-dataset \
    --repo_id dataset/old_image_dataset \
    --operation.type convert_to_video \
    --operation.output_dir /path/to/output
```

### Custom Video Settings

```bash
# Adjust encoding settings
lerobot-edit-dataset \
    --repo_id dataset/old_image_dataset \
    --new_repo_id dataset/converted_video \
    --operation.type convert_to_video \
    --operation.vcodec libx264 \
    --operation.crf 23 \
    --operation.num_workers 8
```

### Convert Specific Episodes

```bash
# Only convert episodes 0-9
lerobot-edit-dataset \
    --repo_id dataset/old_image_dataset \
    --operation.type convert_to_video \
    --operation.episode_indices "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]"
```

### Video Codec Options

- **libsvtav1** (default): Best compression, slower encoding
- **libx264**: Good compression, faster encoding, widely supported
- **libx265**: Better than x264, good for archival

### What Happens

1. **Reads images from parquet** columns
2. **Encodes to MP4 videos** with specified codec
3. **Creates episode metadata** with video mappings
4. **Removes image columns** from data parquet
5. **Significantly reduces dataset size**

---

## Best Practices

### Before Editing

1. **Verify dataset first**
   ```bash
   uv run python verify_dataset.py dataset/my_dataset
   ```

2. **Backup important datasets**
   ```bash
   cp -r ~/.cache/huggingface/lerobot/dataset/my_dataset /path/to/backup/
   ```

3. **Check which episodes to delete**
   ```bash
   # View episodes to identify bad ones
   uv run python view_dataset_local.py dataset/my_dataset --all
   ```

### During Editing

1. **Use `--new_repo_id` for safety** - Preserve original dataset
2. **Test on small datasets first** - Ensure commands work as expected
3. **Monitor disk space** - Editing creates temporary files
4. **Use multiple workers** for large video conversions

### After Editing

1. **Verify the new dataset**
   ```bash
   uv run python verify_dataset.py dataset/my_dataset_modified
   ```

2. **Test loading**
   ```python
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   dataset = LeRobotDataset("dataset/my_dataset_modified")
   sample = dataset[0]
   print(sample.keys())
   ```

3. **Compare statistics**
   ```bash
   # Check that episodes, frames, cameras match expectations
   uv run python verify_dataset.py dataset/my_dataset_modified
   ```

4. **Delete backup only after confirming**
   ```bash
   # After thorough testing
   rm -rf ~/.cache/huggingface/lerobot/dataset/my_dataset_old
   ```

---

## Common Use Cases

### Remove Bad Episodes After Recording

**Scenario:** You recorded 20 episodes but episodes 5, 12, and 18 were bad demonstrations.

```bash
# 1. View episodes to identify bad ones
uv run python view_dataset_local.py data/my_dataset --all

# 2. Delete bad episodes
lerobot-edit-dataset \
    --repo_id data/my_dataset \
    --new_repo_id data/my_dataset_clean \
    --operation.type delete_episodes \
    --operation.episode_indices "[5, 12, 18]"

# 3. Verify cleaned dataset
uv run python verify_dataset.py data/my_dataset_clean
```

### Create Train/Val Split for Training

**Scenario:** You have 100 episodes and want 80 for training, 20 for validation.

```bash
# Split dataset
lerobot-edit-dataset \
    --repo_id dataset/my_full_dataset \
    --operation.type split \
    --operation.splits '{"train": 0.8, "val": 0.2}'

# Results in:
# - dataset/my_full_dataset_train (80 episodes)
# - dataset/my_full_dataset_val (20 episodes)
```

### Merge Multiple Recording Sessions

**Scenario:** You recorded data across 3 sessions and want to combine them.

```bash
# Merge sessions
lerobot-edit-dataset \
    --repo_id dataset/combined_dataset \
    --operation.type merge \
    --operation.repo_ids "['data/session1', 'data/session2', 'data/session3']"

# Verify merged dataset
uv run python verify_dataset.py dataset/combined_dataset
```

### Remove Unused Camera to Save Space

**Scenario:** You recorded with 2 cameras but only need 1 for training.

```bash
# Remove wrist camera (keep scene camera only)
lerobot-edit-dataset \
    --repo_id dataset/my_dataset \
    --new_repo_id dataset/my_dataset_scene_only \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.wrist']"

# Check file sizes
du -sh dataset/my_dataset
du -sh dataset/my_dataset_scene_only
```

### Clean Dataset Before Pushing to Hub

**Scenario:** You want to share your dataset on HuggingFace Hub but need to remove test episodes first.

```bash
# 1. Remove test/debug episodes
lerobot-edit-dataset \
    --repo_id data/my_dataset \
    --new_repo_id dataset/public_dataset \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 1, 2]"  # Remove first 3 test episodes

# 2. Verify cleaned dataset
uv run python verify_dataset.py dataset/public_dataset

# 3. Push to hub (manual step)
# Use HuggingFace Hub CLI or web interface
```

---

## Troubleshooting

### Error: "Cannot delete all episodes"

**Cause:** Trying to delete all episodes from dataset.

**Solution:** Keep at least 1 episode.

### Error: "Invalid episode indices"

**Cause:** Episode indices don't exist in dataset.

**Solution:** Check valid episodes:
```bash
uv run python verify_dataset.py dataset/my_dataset
```

### Error: "Incompatible datasets for merge"

**Cause:** Datasets have different robot types, FPS, or feature structures.

**Solution:** Only merge datasets from same robot with same configuration.

### Operation Very Slow

**Cause:** Re-encoding large video files.

**Solution:** 
- Increase workers: `--operation.num_workers 8`
- Use faster codec: `--operation.vcodec libx264`
- Only edit necessary episodes

### Disk Space Issues

**Cause:** Editing creates temporary files and doesn't delete originals.

**Solution:**
1. Check disk space before editing: `df -h`
2. Delete old backups after confirming new dataset works
3. Use external drive for large operations

---

## Quick Reference

```bash
# DELETE EPISODES
lerobot-edit-dataset --repo_id data/my_dataset \
    --operation.type delete_episodes \
    --operation.episode_indices "[0, 5, 10]"

# SPLIT DATASET (80/20)
lerobot-edit-dataset --repo_id data/my_dataset \
    --operation.type split \
    --operation.splits '{"train": 0.8, "val": 0.2}'

# MERGE DATASETS
lerobot-edit-dataset --repo_id data/merged \
    --operation.type merge \
    --operation.repo_ids "['data/set1', 'data/set2']"

# REMOVE CAMERA
lerobot-edit-dataset --repo_id data/my_dataset \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.wrist']"

# CONVERT TO VIDEO
lerobot-edit-dataset --repo_id data/old_dataset \
    --operation.type convert_to_video \
    --new_repo_id data/video_dataset
```

---

## Additional Resources

- **LeRobot Docs:** https://github.com/huggingface/lerobot
- **Dataset Tools API:** `lerobot/datasets/dataset_tools.py`
- **Edit Script:** `lerobot/scripts/lerobot_edit_dataset.py`
- **Verification Tool:** `verify_dataset.py` (local)

---

**Pro Tip:** Always use `--new_repo_id` when experimenting with dataset edits. You can delete the original later once you've confirmed the edited version works correctly!


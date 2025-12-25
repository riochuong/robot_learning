# Dataset Safety Documentation

## Read-Only Scripts

The following scripts are **completely read-only** and **safe to use** on your training datasets:

### ✅ `view_dataset_local.py`

**Purpose:** Visualize episodes with camera feeds and robot data in Rerun viewer.

**Operations performed:**
- ✅ **READ** JSON metadata files (`meta/info.json`)
- ✅ **READ** Parquet data files (`data/chunk-*/file-*.parquet`)
- ✅ **READ** MP4 video files (`videos/*/chunk-*/file-*.mp4`)
- ✅ **CREATE** in-memory copies for visualization only
- ✅ **DISPLAY** data in Rerun viewer (separate process)

**File system access:**
- Uses `'r'` (read-only) mode for all file opens
- Uses `pandas.read_parquet()` which is read-only by design
- Uses `ffmpeg` with read-only flags (`-i` input only)
- **NEVER** opens files in write mode ('w', 'a', 'w+', etc.)
- **NEVER** modifies, deletes, or creates files in dataset directory

**Memory safety:**
- All pandas DataFrames are copies, not views of original data
- Modifications to in-memory data do not affect disk
- `.reset_index(drop=True)` operates on DataFrame copy only

### ✅ `view_all_episodes.py`

**Purpose:** Browse through multiple episodes sequentially.

**Operations performed:**
- ✅ **READ** JSON metadata to count episodes
- ✅ **CALL** `lerobot-dataset-viz` subprocess (also read-only)

### ✅ `camera_test.py`

**Purpose:** Test camera connections.

**Operations performed:**
- ✅ **READ** camera streams via OpenCV
- ✅ **DISPLAY** live camera feeds
- ⚠️  **WRITE** test images to `outputs/captured_images/` (NOT your dataset directory)

## Training-Safe Guarantee

**Your training data is completely safe because:**

1. ✅ All visualization scripts use read-only operations
2. ✅ No script writes to the dataset directory
3. ✅ pandas and ffmpeg operations create in-memory copies
4. ✅ Rerun visualization is a separate process that doesn't touch source data
5. ✅ Even if a script crashes, your data remains unchanged

## Dataset Directory Structure (Protected)

```
~/.cache/huggingface/lerobot/data/your_dataset/
├── data/           ← READ ONLY by visualization scripts
│   └── chunk-000/
│       └── *.parquet
├── meta/           ← READ ONLY by visualization scripts
│   ├── info.json
│   ├── stats.json
│   └── tasks.parquet
└── videos/         ← READ ONLY by visualization scripts
    ├── observation.images.scene/
    └── observation.images.wrist/
```

## How to Verify Safety

You can verify scripts are read-only by checking:

```bash
# Check for any write operations in the script
grep -i "write\|delete\|remove\|'w'\|'a'" view_dataset_local.py
# Result: Should only find comments, not actual file operations

# Check file permissions (they remain unchanged)
ls -la ~/.cache/huggingface/lerobot/data/your_dataset/data/chunk-000/
# Run visualization
uv run python view_dataset_local.py data/your_dataset 0
# Check permissions again - should be identical
ls -la ~/.cache/huggingface/lerobot/data/your_dataset/data/chunk-000/
```

## Best Practices

1. ✅ Always use `view_dataset_local.py` for visualization
2. ✅ Never manually edit files in the dataset directory
3. ✅ Keep backups of important datasets (copy entire directory)
4. ✅ Use `--dataset.push_to_hub=False` during recording to keep data local
5. ✅ Use separate directories for experiments vs final training data

## Summary

**You can safely use `view_dataset_local.py` on your training datasets.** It is designed from the ground up to be read-only and will never modify, corrupt, or delete your training data.


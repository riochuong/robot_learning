# LeRobot Dataset Knowledge Base

> Comprehensive reference for understanding, creating, verifying, and debugging LeRobot datasets

**Last Updated:** December 14, 2025  
**LeRobot Version:** v3.0

---

## Table of Contents

1. [Dataset Structure](#dataset-structure)
2. [Video Storage Architecture](#video-storage-architecture)
3. [Recording Datasets](#recording-datasets)
4. [Verification and Debugging](#verification-and-debugging)
5. [Common Issues and Solutions](#common-issues-and-solutions)
6. [Tools Reference](#tools-reference)
7. [Key Insights](#key-insights)
8. [Best Practices](#best-practices)

---

## Dataset Structure

### Directory Layout

```
dataset/my_dataset/
├── meta/
│   ├── info.json              # Dataset metadata
│   ├── episodes/
│   │   └── chunk-000/
│   │       └── file-000.parquet  # Episode-level metadata including video mappings
│   └── tasks/
│       └── chunk-000/
│           └── file-000.parquet  # Task descriptions
├── data/
│   └── chunk-000/
│       └── file-*.parquet     # Frame-level data (NO video columns)
└── videos/
    ├── observation.images.scene/
    │   └── chunk-000/
    │       └── file-*.mp4     # Video files for scene camera
    └── observation.images.wrist/
        └── chunk-000/
            └── file-*.mp4     # Video files for wrist camera
```

### Critical Understanding

**Data Parquet Files (data/chunk-000/file-*.parquet):**
- Contains: `action`, `observation.state`, `timestamp`, `frame_index`, `episode_index`, `index`
- **DOES NOT** contain video/image columns
- Frame-level data only

**Episode Metadata Files (meta/episodes/chunk-000/file-*.parquet):**
- Contains: `episode_index`, `length`, `timestamp`
- **MOST IMPORTANT:** Contains video mapping columns:
  - `videos/{video_key}/file_index` - Which MP4 file contains this episode's video
  - `videos/{video_key}/from_timestamp` - Start time in video file
  - `videos/{video_key}/to_timestamp` - End time in video file
- One row per episode

**Video Files (videos/{camera_name}/chunk-000/file-*.mp4):**
- MP4 video files containing actual image data
- Can contain multiple episodes per file
- Frame extraction done on-the-fly during training
- Indexed via episode metadata

---

## Video Storage Architecture

### How LeRobot Links Videos to Episodes

**The Critical Discovery:** LeRobot uses a **timestamp-based indirect mapping** stored in episode metadata, NOT direct image columns in data parquet files.

```
Episode Metadata Entry:
{
  "episode_index": 15,
  "videos/observation.images.scene/file_index": 2,
  "videos/observation.images.scene/from_timestamp": 123.45,
  "videos/observation.images.scene/to_timestamp": 156.78
}

This means:
- Episode 15's scene camera video is in file-002.mp4
- It spans from 123.45s to 156.78s in that file
- At 30 FPS: frame_offset = int(123.45 * 30) = 3703
- Number of frames: int((156.78 - 123.45) * 30) = 999 frames
```

### Why This Architecture?

1. **Efficiency**: Videos are stored once in compressed MP4 format
2. **Flexibility**: Multiple episodes can share the same video file
3. **Performance**: Parquet files stay small and fast to query
4. **Streaming**: Can decode only needed frames on-demand

### Data Loading Flow

```
1. User requests sample index 5000
   ↓
2. LeRobotDataset loads data parquet row 5000
   → Gets: episode_index=15, frame_index=500, action, state
   ↓
3. If images requested, uses episode_index to query episode metadata
   → Gets: file_index=2, from_ts=123.45, to_ts=156.78
   ↓
4. Calculates: video_frame = int(from_ts * fps) + frame_index_within_episode
   ↓
5. Opens video file-002.mp4 and seeks to video_frame
   ↓
6. Decodes and returns image
```

**Important:** `LeRobotDataset.__getitem__()` does this automatically. The `_query_videos()` method handles all video loading logic.

---

## Recording Datasets

### Two-Camera Recording Command

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm_so_101 \
    --robot.cameras='{"scene": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30}}' \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm_so_100 \
    --dataset.repo_id=data/my_dataset \
    --dataset.push_to_hub=False \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=10 \
    --display_data=True \
    --dataset.single_task="Task description"
```

### Key Parameters

- `--robot.cameras`: **MUST** be a single JSON string for complex dict arguments
- `--display_data=True`: Shows live Rerun.io visualization during recording
- `--dataset.reset_time_s`: Time between episodes for manual reset
- `--dataset.push_to_hub=False`: Keep dataset local only

### Camera Timeout Issues

**Problem:** `TimeoutError: Timed out waiting for frame from camera` during recording

**Causes:**
1. USB bandwidth limitations (multiple cameras on same bus)
2. Camera hardware instability
3. Resolution/FPS too high for USB 2.0
4. USB cable quality issues

**Solutions:**
1. Use different USB buses for cameras
2. Reduce resolution or FPS: `--robot.cameras='{"scene": {"...", "fps": 15}}'`
3. Check with `v4l2-ctl --list-devices` and `lsusb -t`
4. Use `--resume=True` to continue recording after interruption

**Data Safety:** Partially recorded episodes up to the crash point are usually valid and usable.

---

## Verification and Debugging

### Quick Verification

```bash
# Comprehensive dataset verification
uv run python verify_dataset.py dataset/my_dataset

# Custom root directory
uv run python verify_dataset.py data/my_dataset --root ~/.cache/huggingface/lerobot

# JSON output for scripting
uv run python verify_dataset.py dataset/my_dataset --json
```

### What Gets Verified

✅ Dataset directory exists  
✅ `meta/info.json` present and valid  
✅ Episode metadata exists with video mappings  
✅ Data parquet files exist  
✅ Video files present (counts and sizes)  
✅ Video mapping columns in episode metadata  
✅ Data structure correctness (no video columns in data parquet)  
✅ Episode count consistency  
✅ Video file accessibility  

### Manual Inspection Tools

```bash
# Inspect parquet files in detail
uv run python inspect_parquet.py dataset/my_dataset/data/chunk-000/file-000.parquet

# Check episode metadata
uv run python inspect_parquet.py dataset/my_dataset/meta/episodes/chunk-000/file-000.parquet

# View dataset episodes with Rerun
uv run python view_dataset_local.py data/my_dataset 0

# Browse all episodes sequentially
uv run python view_dataset_local.py data/my_dataset --all
```

### Testing Data Loading

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Load dataset
dataset = LeRobotDataset(
    repo_id="dataset/my_dataset",
    root="/home/user/.cache/huggingface/lerobot"
)

print(f"Total samples: {len(dataset)}")
print(f"Episodes: {dataset.episodes}")

# Test loading samples
sample = dataset[0]
print(f"Keys: {sample.keys()}")
print(f"Action shape: {sample['action'].shape}")
print(f"State shape: {sample['observation.state'].shape}")

if 'observation.images.scene' in sample:
    print(f"Scene image shape: {sample['observation.images.scene'].shape}")
    # Should be: (3, height, width) in CHW format
```

---

## Common Issues and Solutions

### Issue 1: "Missing video columns in parquet"

**Status:** ❌ **FALSE ALARM** - This is normal!

**Misunderstanding:** Data parquet files should have image columns  
**Reality:** Videos are mapped via episode metadata, NOT stored in data parquet

**Verification:**
```bash
# Check episode metadata (should have video mapping columns)
uv run python -c "
import pandas as pd
df = pd.read_parquet('dataset/my_dataset/meta/episodes/chunk-000/file-000.parquet')
video_cols = [c for c in df.columns if 'video' in c]
print('Video mapping columns:', video_cols)
"
```

**Expected output:**
```
Video mapping columns: [
  'videos/observation.images.scene/file_index',
  'videos/observation.images.scene/from_timestamp',
  'videos/observation.images.scene/to_timestamp',
  'videos/observation.images.wrist/file_index',
  ...
]
```

### Issue 2: "Videos not showing for episodes > 0"

**Cause:** Incorrect assumption about video file chunking

**Wrong assumption:**
```python
# ❌ WRONG: Assumes fixed 1000-frame chunks
file_index = start_index // 1000
frame_offset = start_index % 1000
```

**Correct approach:**
```python
# ✅ CORRECT: Use episode metadata
episodes_meta_df = pd.read_parquet(dataset_path / "meta/episodes/chunk-000/file-000.parquet")
episode_meta = episodes_meta_df[episodes_meta_df['episode_index'] == episode_idx].iloc[0]

video_file_index = int(episode_meta[f"videos/{video_key}/file_index"])
from_timestamp = float(episode_meta[f"videos/{video_key}/from_timestamp"])
to_timestamp = float(episode_meta[f"videos/{video_key}/to_timestamp"])

fps = info["fps"]
frame_offset = int(from_timestamp * fps)
num_frames = int((to_timestamp - from_timestamp) * fps)
```

### Issue 3: "LeRobotDataset.__getitem__ index mismatch with filtered episodes"

**Symptom:** When loading dataset with episode filter, indices don't match expected episodes

**Example:**
```python
# Load only episodes [5, 6, 7]
dataset = LeRobotDataset("dataset/my_dataset", episodes=[5, 6, 7])

# dataset[0] should return first frame of episode 5
# but might return wrong frame if indexing is broken
```

**Root cause:** `__getitem__` needs to map relative indices (0-based in filtered dataset) to absolute indices (original position in full dataset)

**Fix location:** `lerobot/src/lerobot/datasets/lerobot_dataset.py` - line ~500-550 in `__getitem__` method

**Note:** This bug exists in LeRobot library but doesn't affect most use cases because:
1. Training usually uses all episodes
2. The bug only affects explicit episode filtering
3. Visual inspection tools work around it

### Issue 4: "lerobot-dataset-viz doesn't work with local datasets"

**Cause:** `lerobot-dataset-viz` requires HuggingFace Hub access for version control

**Solution:** Use custom viewer instead:
```bash
uv run python view_dataset_local.py data/my_dataset 0
```

**Workaround for official tool:**
```bash
# Set offline mode (will fail for local-only datasets)
HF_HUB_OFFLINE=1 lerobot-dataset-viz --repo-id dataset/my_dataset
# ❌ Will fail: FileNotFoundError or OfflineModeIsEnabled

# Local datasets must be viewed with custom tools
```

### Issue 5: "Videos overlapping in Rerun viewer"

**Cause:** Rerun sessions persist data across multiple `rr.log_*()` calls

**Solution:** Start fresh Rerun session for each episode:
```python
import rerun as rr

for episode_idx in episodes:
    # Unique session for each episode
    session_id = f"dataset_viewer_ep{episode_idx}"
    rr.init(session_id, spawn=True)
    
    # Log episode data
    # ...
    
    # Disconnect before next episode
    rr.disconnect()
    input("Press Enter for next episode...")
```

### Issue 6: "High calibration errors despite correct robot behavior"

**Causes:**
1. **Observation timing:** Taking observation before action is applied
2. **Unit mismatch:** Comparing degrees with radian tolerance
3. **Starting position:** Robot not at same position as recording
4. **Processing mismatch:** Observation processing not applied during replay

**Solutions:**
- Take observation AFTER sending action
- Convert degrees to radians: `np.deg2rad(error)`
- Move robot to initial recorded state before replay
- Apply same observation processing pipeline during replay

---

## Tools Reference

### verify_dataset.py

**Purpose:** Comprehensive dataset structure verification

**Usage:**
```bash
uv run python verify_dataset.py dataset/my_dataset
```

**Checks:**
- Directory structure
- Metadata files
- Episode count and consistency
- Video file presence and mapping
- Data parquet structure
- Video metadata columns

**Output:** Pass/fail verdict with detailed report

**Exit codes:** 0 = valid, 1 = invalid

---

### view_dataset_local.py

**Purpose:** Visualize local-only LeRobot datasets with Rerun

**Features:**
- Works without HuggingFace Hub access
- Uses episode metadata for accurate video loading
- Fresh Rerun window per episode (no overlap)
- Supports all episodes browsing
- Debug mode for frame verification

**Usage:**
```bash
# Single episode
uv run python view_dataset_local.py data/my_dataset 0

# Browse all episodes
uv run python view_dataset_local.py data/my_dataset --all

# Start from episode 10
uv run python view_dataset_local.py data/my_dataset 10 --all

# Debug mode
DEBUG_VIDEO=1 uv run python view_dataset_local.py data/my_dataset 0
```

**Controls:**
- `Enter` - Next episode
- `s` - Skip to specific episode
- `q` - Quit

**How it works:**
1. Reads episode metadata to get video mapping
2. Uses FFmpeg to decode specific frame range
3. Spawns Rerun with unique session ID
4. Logs video frames, joint states, actions
5. Disconnects Rerun before next episode

**Key implementation:**
```python
def get_episode_video_info(dataset_path, episode_idx, video_key):
    """Get video file info from episode metadata."""
    episodes_meta = pd.read_parquet(
        dataset_path / "meta/episodes/chunk-000/file-000.parquet"
    )
    episode_meta = episodes_meta[
        episodes_meta['episode_index'] == episode_idx
    ].iloc[0]
    
    file_index = int(episode_meta[f"videos/{video_key}/file_index"])
    from_ts = float(episode_meta[f"videos/{video_key}/from_timestamp"])
    to_ts = float(episode_meta[f"videos/{video_key}/to_timestamp"])
    
    return file_index, from_ts, to_ts
```

---

### test_calibration.py

**Purpose:** Test robot calibration by recording and replaying trajectories

**Features:**
- Records robot movements
- Replays and compares joint positions
- Handles observation processing pipeline
- Converts units (degrees ↔ radians)
- Moves robot to initial position before replay

**Usage:**
```bash
uv run python test_calibration.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm_so_101 \
    --dataset.repo_id=data/test_calibration \
    --dataset.skip_record=True \
    --dataset.test_episodes=0,1,2
```

**Key insights:**
- Observation must be taken AFTER action is applied
- Use same processing pipeline for recording and replay
- Convert units consistently
- Match initial robot position

---

### inspect_parquet.py

**Purpose:** Detailed parquet file inspection

**Usage:**
```bash
uv run python inspect_parquet.py path/to/file.parquet
```

**Output:**
- Schema (column names and types)
- Row count
- Sample rows
- Episode breakdown
- Column statistics

---

## Key Insights

### 1. Video Storage is Indirect

**LeRobot does NOT store images directly in data parquet files.**

Instead:
- Videos are stored as compressed MP4 files
- Episode metadata contains mappings (file_index, from_timestamp, to_timestamp)
- `LeRobotDataset._query_videos()` handles the linking automatically
- This is **intentional design**, not a bug

### 2. Episode Metadata is Critical

Episode metadata (`meta/episodes/`) is not just episode counts - it contains:
- Video file mappings for each episode
- Timestamp ranges in video files
- Episode lengths and indices
- Task assignments

**Without episode metadata, videos cannot be loaded!**

### 3. Parquet Files Have Different Roles

- **Data parquet:** Frame-level observations, actions, timestamps
- **Episode parquet:** Episode-level metadata, video mappings
- **Tasks parquet:** Task descriptions and assignments

Don't expect all information in one file.

### 4. Resume Mode Has Known Issues

`lerobot-record --resume=True` can cause:
- Duplicate data chunks
- Incomplete episode metadata
- Frame index mismatches

**Best practice:** Record full datasets in one session, or manually merge if interrupted.

### 5. Local Datasets Are Different

Local-only datasets (not pushed to Hub) have limitations:
- `lerobot-dataset-viz` may not work
- Need custom tools for visualization
- No version control or collaboration features

But they're perfectly valid for training!

### 6. Calibration Testing Requires Careful Implementation

When testing robot calibration:
- Timing matters: observation after action
- Units matter: degrees vs radians
- Processing matters: same pipeline for record and replay
- Initial state matters: start from same position

Small bugs can cause huge reported errors.

### 7. Video Chunking is Episode-Based

Video files don't chunk at fixed frame counts (e.g., 1000 frames).

Instead:
- Episodes are recorded continuously
- Multiple episodes can be in one video file
- Video file boundaries don't align with episode boundaries
- Use episode metadata to find where each episode's video is

### 8. FFmpeg is the Video Backend

LeRobot uses FFmpeg (via `torchcodec` or `ffmpeg-python`) for:
- Video encoding during recording
- Video decoding during training
- Frame seeking and extraction

Understanding FFmpeg commands helps debug video issues.

---

## Best Practices

### Recording

1. **Test cameras first:** Use `camera_test.py` or `v4l2-ctl --list-devices`
2. **Check USB bandwidth:** Use `lsusb -t` to ensure cameras on different buses
3. **Enable visualization:** Always use `--display_data=True` to catch issues early
4. **Set appropriate reset time:** `--dataset.reset_time_s=10` or higher for manual resets
5. **Start with fewer episodes:** Test with 3-5 episodes before recording 50+
6. **Keep logs:** Redirect output to log file: `lerobot-record ... 2>&1 | tee recording.log`
7. **Use absolute paths:** Avoid confusion with relative paths

### Verification

1. **Always verify after recording:** Run `verify_dataset.py` immediately
2. **Check first and last episodes:** Use `view_dataset_local.py` to spot-check
3. **Test data loading:** Try loading with `LeRobotDataset` before training
4. **Inspect episode metadata:** Ensure video mappings exist
5. **Check video file sizes:** Abnormally small files indicate recording issues

### Debugging

1. **Start with structure:** Check directory layout and file presence first
2. **Read episode metadata:** Most video issues are visible in episode metadata
3. **Test with LeRobotDataset:** Official loader is ground truth
4. **Check logs:** `lerobot-record` logs often show timeout/error patterns
5. **Isolate variables:** Test one episode, one camera, simplified setup
6. **Compare with working dataset:** Use known-good datasets as reference

### Training

1. **Filter bad episodes:** Use `episodes=[0,1,3,5,...]` to exclude corrupted episodes
2. **Monitor first epoch:** Watch for data loading errors early
3. **Check data augmentation:** Ensure transforms work with your image sizes
4. **Verify action ranges:** Check min/max values match robot capabilities
5. **Start with small batches:** Test data pipeline before full training

### Collaboration

1. **Document your setup:** Robot type, camera indices, calibration files
2. **Keep README updated:** Commands that worked, issues encountered
3. **Version your datasets:** Use descriptive names like `data/pick_cube_v3_good`
4. **Share verification results:** Include `verify_dataset.py` output in reports
5. **Push to Hub if possible:** Enables collaboration and version control

---

## Quick Reference Commands

```bash
# RECORDING
lerobot-record --robot.type=so101_follower --robot.cameras='{"scene": {...}}' ...

# VERIFICATION
uv run python verify_dataset.py dataset/my_dataset

# VIEWING
uv run python view_dataset_local.py data/my_dataset --all

# INSPECTION
uv run python inspect_parquet.py dataset/my_dataset/meta/episodes/chunk-000/file-000.parquet

# TESTING LOADING
uv run python -c "from lerobot.common.datasets.lerobot_dataset import LeRobotDataset; \
    ds = LeRobotDataset('dataset/my_dataset'); \
    print(f'Episodes: {ds.episodes}'); \
    print(f'Samples: {len(ds)}'); \
    sample = ds[0]; \
    print(f'Keys: {sample.keys()}')"

# CAMERA TEST
uv run python camera_test.py 2 4

# CALIBRATION TEST
uv run python test_calibration.py --robot.type=so101_follower --dataset.test_episodes=0,1
```

---

## Troubleshooting Checklist

**Dataset Not Loading:**
- [ ] Does `meta/info.json` exist?
- [ ] Does `meta/episodes/chunk-000/` exist with parquet files?
- [ ] Does `data/chunk-000/` exist with parquet files?
- [ ] Are video directories present if `info.json` claims videos?
- [ ] Run `verify_dataset.py` for full report

**Videos Not Showing:**
- [ ] Check episode metadata has `videos/{key}/file_index` columns
- [ ] Check video files exist in `videos/{camera}/chunk-000/`
- [ ] Check file indices in metadata match actual files
- [ ] Check timestamps are reasonable (not NaN or negative)
- [ ] Use `DEBUG_VIDEO=1 view_dataset_local.py` to debug

**Recording Timeouts:**
- [ ] Check USB bus with `lsusb -t`
- [ ] Test cameras individually with `camera_test.py`
- [ ] Reduce FPS or resolution
- [ ] Check dmesg for USB errors: `dmesg | grep -i usb`
- [ ] Try different USB ports

**High Calibration Errors:**
- [ ] Check observation timing (before vs after action)
- [ ] Check unit consistency (degrees vs radians)
- [ ] Check initial robot position
- [ ] Check observation processing pipeline
- [ ] Add debug prints to compare values

**Episodes Missing:**
- [ ] Check if recording was interrupted (timeout, crash)
- [ ] Check episode metadata file for all expected episodes
- [ ] Check data parquet has rows for all episodes
- [ ] Partially recorded episodes might be excluded from metadata

---

## Advanced Topics

### Custom Dataset Formats

LeRobot can load datasets not in standard format using custom loaders:

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# Convert custom format to LeRobot format
# See: lerobot/common/datasets/push_dataset_to_hub/
```

### Multi-Task Datasets

Store multiple tasks in one dataset:

```bash
# Record with task assignment
lerobot-record --dataset.single_task="Task A" ...

# Filter by task during training
dataset = LeRobotDataset("dataset/multi_task")
task_a_indices = dataset.hf_dataset.filter(lambda x: x['task_index'] == 0)
```

### Delta Actions

LeRobot supports action spaces:
- **Absolute:** Joint positions directly
- **Delta:** Changes from current position

Check `info.json` for `action_is_delta` flag.

### Frame Stacking

Load multiple consecutive frames as input:

```python
dataset = LeRobotDataset(
    "dataset/my_dataset",
    delta_timestamps={
        "observation.images.scene": [-0.033, 0.0],  # 2 frames at 30 FPS
        "observation.state": [0.0],
        "action": [0.0]
    }
)
```

---

## Resources

- **LeRobot Docs:** https://github.com/huggingface/lerobot
- **Rerun.io Docs:** https://www.rerun.io/docs
- **FFmpeg Docs:** https://ffmpeg.org/documentation.html
- **PyArrow/Parquet:** https://arrow.apache.org/docs/python/

---

## Version History

- **v1.0 (Dec 14, 2025):** Initial knowledge base creation
  - Documented video storage architecture
  - Common issues and solutions
  - Tool reference and best practices

---

**End of Knowledge Base**

*This document represents accumulated knowledge from debugging, verifying, and understanding LeRobot datasets in depth. Use it as a reference for future dataset work.*


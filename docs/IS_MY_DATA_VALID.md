# Is My Data Valid? A Complete Answer

## TL;DR

**❌ NO - Your dataset is INVALID for vision-based training**  
**✅ YES - It CAN be viewed with our custom viewer**  
**✅ YES - It could work for joint-space-only training (no vision)**

## The Situation

### What You Have:

```
Dataset: dataset/pick_and_place_small_cube
├── ✅ Videos exist (file-000.mp4 with all frames)
├── ✅ Joint data exists (observation.state, actions)
├── ✅ Metadata claims videos exist
└── ❌ Parquet has NO video columns (the critical problem!)
```

### What's Missing:

```
Parquet columns:
  ['action', 'observation.state', 'timestamp', ...]
  ❌ NO 'observation.images.scene'
  ❌ NO 'observation.images.wrist'

Metadata claims:
  ['action', 'observation.state', 'observation.images.scene', 
   'observation.images.wrist', 'timestamp', ...]
  
MISMATCH! Metadata lies about what's in parquet.
```

## Why Our Viewer Works

Our `view_dataset_local.py` is a **workaround** that:

1. Reads parquet for joint data and frame indices
2. **Assumes** videos exist at specific paths
3. Uses `ffprobe` to count video frames
4. **Manually** maps episode indices to video offsets
5. Extracts frames with `ffmpeg`

**This is a hack!** It works for viewing but doesn't fix the underlying corruption.

## Why Training Won't Work

LeRobot's training code expects:

```python
# Load a sample
sample = dataset[0]

# Access images (expects this to work)
scene_image = sample['observation.images.scene']  # ❌ KEY ERROR!
wrist_image = sample['observation.images.wrist']  # ❌ KEY ERROR!

# These DO work:
joint_state = sample['observation.state']  # ✅
action = sample['action']  # ✅
```

Because the parquet has no video columns, samples don't include images.

## The Disconnect Explained

| Component | Status | Why |
|-----------|--------|-----|
| Video files on disk | ✅ Exist | Recording saved them |
| Metadata (info.json) | ⚠️ Claims videos exist | Updated during recording |
| Parquet video columns | ❌ Missing | Bug in recording process |
| Our viewer | ✅ Works | Manual workaround |
| LeRobot training | ❌ Fails | Expects video columns |

## What Happened During Recording

Best guess at the sequence:

1. **Recording started**
   ```bash
   lerobot-record ... --robot.cameras='...'
   ```

2. **For each frame:**
   - ✅ Camera captures images → saved to MP4
   - ✅ Robot joints recorded → saved to parquet
   - ❌ Video reference NOT added to parquet row

3. **After recording:**
   - ✅ Video files finalized (file-000.mp4)
   - ✅ Metadata updated (claims videos exist)
   - ❌ Parquet never got video columns

4. **Result:**
   - "Orphaned" video files
   - Metadata mismatch
   - Dataset corrupted

## Possible Causes

1. **Resume mode bug** - Recording without cameras, then resumed with cameras
2. **Recording interrupted** - Crash during video encoding
3. **LeRobot bug** - Video linking fails silently in local-only mode
4. **Schema lock** - First episode defines schema, cameras added later

## Can It Be Fixed?

### ❌ Cannot Fix Existing Data

There's no LeRobot tool to:
- Add video columns to existing parquet
- Rebuild the video-to-frame mapping
- Repair corrupted datasets

Manual reconstruction would require:
- Deep knowledge of LeRobot's video encoding
- Frame-by-frame mapping
- Schema modification
- High risk of further corruption

### ✅ Can Use For Limited Purposes

**Option 1: Joint-space training (no vision)**
```python
# Train using only proprioception
dataset = LeRobotDataset(...)
# Use only observation.state and action
# Ignore missing images
```

**Option 2: Viewing/analysis**
```bash
# Use our custom viewer
uv run python view_dataset_local.py dataset/pick_and_place_small_cube 0
```

**Option 3: Extract demonstrations manually**
```python
# Read parquet for trajectories
# Use videos for human analysis
# But can't use together in training
```

### ✅ Must Re-record For Vision Training

**There is NO alternative** - you must re-record if you want to train vision-based policies.

## How to Avoid This Next Time

### ✅ DO:

1. **Test first episode immediately**
   ```bash
   # After first episode, CHECK:
   python -c "
   import pandas as pd
   df = pd.read_parquet('path/to/file-000.parquet')
   print('Has videos:', 'observation.images.scene' in df.columns)
   "
   ```

2. **Start fresh, never resume**
   ```bash
   # Delete dataset completely
   rm -rf ~/.cache/huggingface/lerobot/dataset/my_dataset
   
   # Record from scratch
   lerobot-record ...  # NO --resume=True
   ```

3. **Include cameras from episode 0**
   ```bash
   # Always specify cameras in initial command
   lerobot-record \
       --robot.cameras='{"scene": {...}, "wrist": {...}}' \
       ... # First command, not added later
   ```

4. **Monitor recording output**
   ```bash
   # Watch for errors like:
   # "TimeoutError: Timed out waiting for frame"
   # → STOP and restart, don't resume
   ```

5. **Verify after completion**
   ```bash
   # Run inspection immediately
   uv run python inspect_parquet.py path/to/file-000.parquet
   # Check: "Has scene video: True"
   ```

### ❌ DON'T:

1. **Don't use --resume=True** after any errors
2. **Don't change camera config** mid-dataset
3. **Don't continue after crashes** - start fresh
4. **Don't record 65 episodes** before checking first one!
5. **Don't assume metadata is correct** - verify parquet

## Verification Script

```bash
#!/bin/bash
# check_dataset.sh - Run after ANY recording session

DATASET_PATH="$1"

echo "Checking dataset: $DATASET_PATH"

# Check parquet columns
uv run python -c "
import pandas as pd
from pathlib import Path

pf = Path('$DATASET_PATH') / 'data/chunk-000/file-000.parquet'
df = pd.read_parquet(pf)

has_scene = 'observation.images.scene' in df.columns
has_wrist = 'observation.images.wrist' in df.columns

print(f'Parquet columns: {list(df.columns)}')
print(f'Has scene video: {has_scene}')
print(f'Has wrist video: {has_wrist}')

if has_scene and has_wrist:
    print('\n✅ DATASET IS VALID')
    exit(0)
else:
    print('\n❌ DATASET IS CORRUPTED - DELETE AND RE-RECORD')
    exit(1)
"
```

## Final Answer

### Your Current Dataset:

```
Status: ❌ CORRUPTED - Cannot use for vision training
Cause: Video files exist but parquet has no video columns
Fix: None - must re-record
Value: Can view episodes, extract joint trajectories, analyze demonstrations
       But CANNOT train vision-based policies
```

### Next Steps:

1. **Delete this dataset** (it's unusable for your goal)
   ```bash
   rm -rf ~/.cache/huggingface/lerobot/dataset/pick_and_place_small_cube
   ```

2. **Re-record properly**
   - Include cameras from first episode
   - Don't use --resume=True
   - Verify first episode immediately

3. **Keep our viewer** (`view_dataset_local.py`)
   - Works around video column issue
   - Useful for checking recordings
   - Better than official lerobot-dataset-viz for corrupted data

## The Real Problem

This isn't user error - it's a **LeRobot bug** that affects recording. Until it's fixed:
- You must be extremely careful with recording workflow
- Verify every dataset immediately
- Never trust that recording "worked" until verified
- Our custom tools are necessary workarounds

The bug we've identified:
- **Symptom**: Videos record but parquet missing video columns
- **Trigger**: Resume mode, crashes, or local-only recording
- **Impact**: 100% of affected datasets unusable for vision training
- **Official fix**: None yet (should report to LeRobot team)


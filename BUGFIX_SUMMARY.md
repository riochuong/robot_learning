# Bug Fix Summary: view_dataset_local.py

## Problem
`view_dataset_local.py` could only show videos for episode 0, not episodes 1 and 2.

## Root Cause
The video file lookup logic assumed videos were chunked every 1000 frames:

```python
# OLD (BROKEN) CODE:
file_index = start_index // 1000  # Assumes 1000 frames per file
frame_offset = start_index % 1000
```

For your dataset:
- Episode 0: frames 0-1115 → file_index=0, offset=0 ✅ Works  
- Episode 1: frames 1116-2203 → file_index=1, offset=116 ❌ Looks for file-001.mp4 (doesn't exist!)
- Episode 2: frames 2204-3359 → file_index=2, offset=204 ❌ Looks for file-002.mp4 (doesn't exist!)

**Reality:** ALL 3360 frames are in ONE file (file-000.mp4)!

## Solution
Changed the logic to dynamically discover which video file contains the episode by:
1. Checking which video files actually exist
2. Using `ffprobe` to get frame count of each video
3. Mapping episode frame indices to the correct video file

```python
# NEW (FIXED) CODE:
# Finds the actual video file by checking frame counts
for video_file in sorted(video_dir.glob("file-*.mp4")):
    video_frame_count = get_frame_count(video_file)
    if start_index < video_frame_count:
        file_index = extract_file_number(video_file)
        frame_offset = start_index
        break
    else:
        start_index -= video_frame_count
```

## Testing
```bash
# Now all episodes work:
uv run python view_dataset_local.py dataset/pick_and_place_small_cube 0  # ✅ Works
uv run python view_dataset_local.py dataset/pick_and_place_small_cube 1  # ✅ Fixed!
uv run python view_dataset_local.py dataset/pick_and_place_small_cube 2  # ✅ Fixed!
```

## Why This Bug Existed
The script was written for "normal" LeRobot datasets where:
- Video columns exist in parquet (link data to videos)
- Videos are chunked predictably

But for corrupted datasets (no video columns in parquet):
- We have to guess video file layout
- Old code made wrong assumptions
- New code discovers layout dynamically

## Status
✅ **FIXED** - All episodes now viewable in datasets without video columns in parquet


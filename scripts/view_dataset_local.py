#!/usr/bin/env python
"""View local dataset episodes without HuggingFace Hub dependency.

IMPORTANT: This script is READ-ONLY and does NOT modify your training data.
It only:
- Reads metadata JSON files
- Reads parquet data files
- Reads video MP4 files
- Visualizes in Rerun (separate process)

Your original dataset remains completely unchanged and safe for training.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import rerun as rr
from PIL import Image


def get_episode_video_info(dataset_path: Path, episode_idx: int, video_key: str, debug: bool = False):
    """Get video file and timestamp info for a specific episode.
    
    Uses episode metadata (just like official LeRobot) to get the correct video file
    and timestamp mapping. This is the ONLY correct way to map episodes to videos.
    
    Returns:
        tuple: (file_index, from_timestamp, to_timestamp, num_frames) or (None, None, None, None)
    """
    import pandas as pd
    
    # Load EPISODE METADATA (the source of truth for video mapping)
    episodes_dir = dataset_path / "meta" / "episodes" / "chunk-000"
    if not episodes_dir.exists():
        if debug:
            print(f"   [DEBUG] Episode metadata directory not found: {episodes_dir}")
        return None, None, None, None
    
    episodes_files = sorted(episodes_dir.glob("*.parquet"))
    if not episodes_files:
        if debug:
            print(f"   [DEBUG] No episode metadata files found")
        return None, None, None, None
    
    # Load episode metadata
    episodes_df = pd.concat([pd.read_parquet(f) for f in episodes_files], ignore_index=True)
    
    if episode_idx not in episodes_df['episode_index'].values:
        if debug:
            print(f"   [DEBUG] Episode {episode_idx} not found in metadata")
        return None, None, None, None
    
    episode_meta = episodes_df[episodes_df['episode_index'] == episode_idx].iloc[0]
    
    # Extract video info from episode metadata (exactly like LeRobot does)
    # Column names are like: videos/observation.images.scene/file_index
    file_index_col = f"videos/{video_key}/file_index"
    from_ts_col = f"videos/{video_key}/from_timestamp"
    to_ts_col = f"videos/{video_key}/to_timestamp"
    
    if file_index_col not in episode_meta or pd.isna(episode_meta[file_index_col]):
        if debug:
            print(f"   [DEBUG] Video info for {video_key} not found in episode metadata")
        return None, None, None, None
    
    file_index = int(episode_meta[file_index_col])
    from_timestamp = float(episode_meta[from_ts_col])
    to_timestamp = float(episode_meta[to_ts_col])
    
    # Get number of frames from data
    data_dir = dataset_path / "data" / "chunk-000"
    data_files = sorted(data_dir.glob("*.parquet"))
    for data_file in data_files:
        df = pd.read_parquet(data_file)
        if episode_idx in df['episode_index'].values:
            num_frames = len(df[df['episode_index'] == episode_idx])
            break
    else:
        num_frames = None
    
    if debug:
        print(f"   [DEBUG] Episode {episode_idx} video info from metadata:")
        print(f"           Video key: {video_key}")
        print(f"           Video file: file-{file_index:03d}.mp4")
        print(f"           Timestamp range: {from_timestamp:.2f}s to {to_timestamp:.2f}s")
        print(f"           Duration: {to_timestamp - from_timestamp:.2f}s")
        print(f"           Frames: {num_frames}")
    
    return file_index, from_timestamp, to_timestamp, num_frames


def load_video_frames(dataset_path: Path, video_key: str, episode_idx: int, debug: bool = False):
    """Load video frames for a specific episode using episode metadata.
    
    This mimics LeRobot's approach: uses timestamps from episode metadata
    to extract the correct frames from the video file.
    
    Args:
        dataset_path: Path to the dataset directory
        video_key: Video feature key (e.g., 'observation.images.scene')
        episode_idx: Episode index
        debug: Print debug information
    
    Returns:
        List of frames (numpy arrays) or None if video not found
    """
    import pandas as pd
    
    # Get video info from EPISODE METADATA (the correct way)
    result = get_episode_video_info(dataset_path, episode_idx, video_key, debug=debug)
    if result[0] is None:
        print(f"   ⚠️  Could not find video metadata for episode {episode_idx}")
        return None
    
    file_index, from_timestamp, to_timestamp, num_frames = result
    
    video_dir = dataset_path / "videos" / video_key / "chunk-000"
    video_file = video_dir / f"file-{file_index:03d}.mp4"
    
    if not video_file.exists():
        print(f"   ⚠️  Video file not found: {video_file}")
        return None
    
    # Load timestamps from data to get exact frame timing
    data_dir = dataset_path / "data" / "chunk-000"
    data_files = sorted(data_dir.glob("*.parquet"))
    
    episode_timestamps = None
    for data_file in data_files:
        df = pd.read_parquet(data_file)
        if episode_idx in df['episode_index'].values:
            episode_data = df[df['episode_index'] == episode_idx]
            episode_timestamps = episode_data['timestamp'].values
            break
    
    if episode_timestamps is None:
        print(f"   ⚠️  Could not find episode data")
        return None
    
    try:
        # Use ffmpeg to extract frames using TIMESTAMPS (like LeRobot does)
        # We need to extract frames at: from_timestamp + each episode timestamp
        
        # Method: Use select filter with timestamps
        # This matches LeRobot's decode_video_frames approach
        fps = 30  # Standard FPS for LeRobot datasets
        
        # Calculate which frames we need from the video
        # Video is recorded at 30fps, timestamps tell us the exact timing
        start_frame = int(from_timestamp * fps)
        end_frame = int(to_timestamp * fps)
        num_frames_to_extract = len(episode_timestamps)
        
        cmd = [
            'ffmpeg',
            '-loglevel', 'error',
            '-i', str(video_file),
            '-vf', f'select=gte(n\\,{start_frame})',  # Start at from_timestamp
            '-frames:v', str(num_frames_to_extract),   # Extract exact count
            '-f', 'image2pipe',
            '-pix_fmt', 'rgb24',
            '-vcodec', 'rawvideo',
            '-'
        ]
        
        if debug:
            print(f"   [DEBUG] Extracting from video file: {video_file.name}")
            print(f"   [DEBUG] Video frame range: {start_frame} to {end_frame}")
            print(f"   [DEBUG] Frames to extract: {num_frames_to_extract}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            check=True,
            timeout=60
        )
        
        # Parse raw video output
        width, height = 640, 480
        frame_size = width * height * 3
        raw_data = result.stdout
        expected_size = num_frames_to_extract * frame_size
        
        if debug:
            print(f"   [DEBUG] Raw data size: {len(raw_data)} bytes")
            print(f"   [DEBUG] Expected size: {expected_size} bytes")
        
        if len(raw_data) < expected_size:
            actual_frames = len(raw_data) // frame_size
            print(f"   ⚠️  Warning: Only received {actual_frames}/{num_frames_to_extract} frames")
            num_frames_to_extract = actual_frames
        
        frames = []
        for i in range(num_frames_to_extract):
            start = i * frame_size
            end = start + frame_size
            if end > len(raw_data):
                break
            frame_data = raw_data[start:end]
            frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((height, width, 3))
            frames.append(frame)
        
        if debug and len(frames) > 0:
            print(f"   [DEBUG] Successfully loaded {len(frames)} frames")
            print(f"   [DEBUG] Frame shape: {frames[0].shape}")
        
        return frames if frames else None
                
    except subprocess.TimeoutExpired:
        print(f"   ⚠️  FFmpeg timeout - video extraction taking too long")
        return None
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode() if e.stderr else "Unknown error"
        print(f"   ⚠️  FFmpeg error: {stderr[:200]}")
        return None
    except Exception as e:
        print(f"   ⚠️  Error loading video: {e}")
        import traceback
        if debug:
            traceback.print_exc()
        return None


def visualize_episode(dataset_path: Path, episode_idx: int, auto_continue: bool = False):
    """Visualize a single episode using Rerun.
    
    READ-ONLY OPERATION: This function only reads data, never writes or modifies.
    
    Args:
        dataset_path: Path to the dataset directory
        episode_idx: Episode index to visualize
        auto_continue: If True, automatically wait for user input after loading
    """
    # Load metadata (READ-ONLY)
    with open(dataset_path / "meta" / "info.json", 'r') as f:
        info = json.load(f)
    
    fps = info["fps"]
    features = info["features"]
    
    # Load parquet files directly (READ-ONLY - pandas.read_parquet never modifies source files)
    import pandas as pd
    data_dir = dataset_path / "data" / "chunk-000"
    parquet_files = sorted(data_dir.glob("*.parquet"))
    
    if not parquet_files:
        print(f"❌ No parquet files found in {data_dir}")
        return False
    
    # Load all parquet files and concatenate (creates in-memory copy, doesn't touch disk)
    dfs = [pd.read_parquet(f) for f in parquet_files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Filter to requested episode (operates on in-memory DataFrame copy)
    episode_data = df[df["episode_index"] == episode_idx]
    
    if len(episode_data) == 0:
        print(f"❌ Episode {episode_idx} not found")
        return False
    
    num_frames = len(episode_data)
    print(f"📊 Episode {episode_idx}: {num_frames} frames")
    episode_data = episode_data.reset_index(drop=True)  # In-memory operation only
    
    # Load video frames for each camera
    video_frames = {}
    debug_mode = os.environ.get('DEBUG_VIDEO', '0') == '1'
    
    for key in features.keys():
        if features[key].get("dtype") == "video":
            print(f"📹 Loading video: {key}...")
            frames = load_video_frames(dataset_path, key, episode_idx, debug=debug_mode)
            if frames:
                video_frames[key] = frames
                print(f"   ✅ Loaded {len(frames)} frames")
                if len(frames) != num_frames:
                    print(f"   ⚠️  WARNING: Expected {num_frames} frames but got {len(frames)}")
            else:
                print(f"   ⚠️  Failed to load video")
    
    # Initialize Rerun with unique app_id for each episode (prevents data mixing)
    # Use a timestamped or random session ID to ensure completely fresh viewer
    import time
    session_id = f"episode_{episode_idx}_{int(time.time())}"
    
    print(f"\n📺 Opening fresh Rerun viewer for Episode {episode_idx}...")
    rr.init(session_id, spawn=True)
    
    # Log each frame
    print(f"🎬 Logging {num_frames} frames to Rerun...")
    for idx in range(len(episode_data)):
        frame = episode_data.iloc[idx]
        timestamp = idx / fps
        
        # Set time using Rerun 0.26 API - timeline name is first positional arg
        rr.set_time_sequence("frame", idx)
        rr.set_time_seconds("timestamp", timestamp)
        
        # Log camera images
        for video_key, frames in video_frames.items():
            if idx < len(frames):
                # Extract camera name from key (e.g., 'observation.images.scene' -> 'scene')
                camera_name = video_key.split('.')[-1]
                rr.log(f"cameras/{camera_name}", rr.Image(frames[idx]))
        
        # Log robot state (joint positions)
        if "observation.state" in frame:
            state = np.array(frame["observation.state"])
            state_names = features.get("observation.state", {}).get("names", [])
            
            # Log as scalars (Rerun 0.26 API uses Scalars)
            for i, name in enumerate(state_names):
                if i < len(state):
                    rr.log(f"robot/joints/{name}", rr.Scalars(float(state[i])))
        
        # Log actions
        if "action" in frame:
            action = np.array(frame["action"])
            action_names = features.get("action", {}).get("names", [])
            
            for i, name in enumerate(action_names):
                if i < len(action):
                    rr.log(f"robot/actions/{name}", rr.Scalars(float(action[i])))
    
    print(f"✅ Data loaded into Rerun viewer")
    
    # Disconnect to ensure clean separation between episodes
    rr.disconnect()
    
    if auto_continue:
        print(f"💡 Close the Rerun window when done reviewing, then press Enter here")
        input()  # Wait for user to acknowledge
    else:
        print(f"💡 Review the episode in Rerun, then close the viewer to continue")
    
    return True


def main():
    """Main entry point."""
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("Usage: python view_dataset_local.py [repo_id] [episode_idx] [root] [--all]")
        print("\nExamples:")
        print("  python view_dataset_local.py data/pick_small_cube_1_20eps 0")
        print("  python view_dataset_local.py data/my_dataset 5 /path/to/datasets")
        print("  python view_dataset_local.py data/my_dataset --all  # View all episodes")
        print("  python view_dataset_local.py data/my_dataset 10 --all  # Start from ep 10")
        print("\nDefaults:")
        print("  repo_id: data/pick_small_cube_1_20eps")
        print("  episode_idx: 0")
        print("  root: ~/.cache/huggingface/lerobot")
        print("\nOptions:")
        print("  --all: Browse through all episodes sequentially")
        sys.exit(0)
    
    # Parse arguments
    browse_all = "--all" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--all"]
    
    repo_id = args[0] if len(args) > 0 else "data/pick_small_cube_1_20eps"
    episode_idx = int(args[1]) if len(args) > 1 else 0
    root = args[2] if len(args) > 2 else str(Path.home() / ".cache" / "huggingface" / "lerobot")
    
    # Construct dataset path
    dataset_path = Path(root) / repo_id
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print(f"\nAvailable datasets:")
        root_path = Path(root)
        if root_path.exists():
            for item in sorted(root_path.glob("data/*")):
                if item.is_dir():
                    print(f"  - {item.relative_to(root_path)}")
        sys.exit(1)
    
    # Load metadata to check episode count
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"❌ Dataset metadata not found at: {info_path}")
        sys.exit(1)
    
    with open(info_path) as f:
        info = json.load(f)
    
    total_episodes = info.get("total_episodes", 0)
    total_frames = info.get("total_frames", 0)
    
    print(f"📁 Dataset: {repo_id}")
    print(f"   Path: {dataset_path}")
    print(f"   Episodes: {total_episodes}")
    print(f"   Frames: {total_frames}")
    print()
    
    if episode_idx >= total_episodes:
        print(f"❌ Episode {episode_idx} out of range (0-{total_episodes-1})")
        sys.exit(1)
    
    # Browse through episodes
    if browse_all:
        print(f"🎬 Browse mode: Will show episodes {episode_idx} to {total_episodes-1}")
        print("   Each episode opens in a fresh Rerun window")
        print("   Close each window when done reviewing\n")
        
        current_ep = episode_idx
        while current_ep < total_episodes:
            print(f"\n{'='*70}")
            print(f"📼 Episode {current_ep}/{total_episodes-1}")
            print(f"{'='*70}")
            
            try:
                # Pass auto_continue=True to wait for user after each episode
                visualize_episode(dataset_path, current_ep, auto_continue=False)
            except Exception as e:
                print(f"❌ Error visualizing episode {current_ep}: {e}")
                import traceback
                traceback.print_exc()
            
            # Prompt for next episode
            if current_ep < total_episodes - 1:
                print(f"\n{'─'*70}")
                response = input(f"▶  Ready for next episode?\n   [Enter]=Load ep {current_ep+1}  [s]=Skip to...  [q]=Quit: ").strip().lower()
                
                if response == 'q':
                    print("👋 Exiting browse mode")
                    break
                elif response == 's':
                    skip_to = input(f"Skip to episode (0-{total_episodes-1}): ").strip()
                    try:
                        current_ep = int(skip_to)
                        if current_ep < 0 or current_ep >= total_episodes:
                            print(f"Invalid episode, continuing from {current_ep}")
                            current_ep = min(current_ep, total_episodes - 1)
                    except ValueError:
                        print(f"Invalid input, moving to next episode")
                        current_ep += 1
                else:
                    current_ep += 1
            else:
                print(f"\n✅ Finished viewing all episodes!")
                break
    else:
        # Single episode mode
        try:
            visualize_episode(dataset_path, episode_idx)
        except Exception as e:
            print(f"❌ Error visualizing episode: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
        # Ask if user wants to view another episode
        if episode_idx < total_episodes - 1:
            print(f"\n{'─'*70}")
            response = input(f"▶  View next episode ({episode_idx + 1})? [y/n]: ")
            if response.lower() == 'y':
                # Update args and recursively call
                if len(args) > 1:
                    sys.argv[2] = str(episode_idx + 1)
                else:
                    sys.argv.append(str(episode_idx + 1))
                main()


if __name__ == "__main__":
    main()


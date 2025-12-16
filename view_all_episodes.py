#!/usr/bin/env python
"""Visualize all episodes in a dataset."""
import os
import subprocess
import sys
from pathlib import Path

# Force offline mode to prevent HF hub access - MUST be set before any imports
os.environ["HF_HUB_OFFLINE"] = "1"

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from datasets import Dataset as HFDataset
from datasets import load_from_disk

def view_all_episodes(repo_id: str, root: str = None):
    """View all episodes in a local dataset.
    
    Args:
        repo_id: Dataset identifier (e.g., 'data/my_dataset')
        root: Root directory where datasets are stored (default: ~/.cache/huggingface/lerobot)
    """
    # Set default root if not provided
    if root is None:
        root = Path.home() / ".cache" / "huggingface" / "lerobot"
    
    root = Path(root)
    
    # Check if dataset exists locally
    dataset_path = root / repo_id
    if not dataset_path.exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print(f"\nAvailable datasets in {root}:")
        if root.exists():
            for item in sorted(root.glob("data/*")):
                if item.is_dir():
                    print(f"  - {item.relative_to(root)}")
        return
    
    print(f"Loading dataset from: {dataset_path}")
    
    # Read metadata directly to get episode count without loading full dataset
    import json
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"❌ Dataset metadata not found at: {info_path}")
        return
    
    with open(info_path) as f:
        info = json.load(f)
    
    num_episodes = info.get("total_episodes", 0)
    num_frames = info.get("total_frames", 0)
    
    print(f"✅ Dataset info: {num_episodes} episodes, {num_frames} frames")
    
    if num_episodes == 0:
        print("❌ No episodes found in dataset")
        return
    
    for ep_idx in range(num_episodes):
        print(f"\n{'='*60}")
        print(f"Viewing episode {ep_idx}/{num_episodes-1}")
        print(f"{'='*60}")
        
        cmd = [
            "lerobot-dataset-viz",
            "--repo-id", repo_id,
            "--episode-index", str(ep_idx),
            "--root", str(root),
        ]
        
        # Set offline mode for the subprocess too
        env = os.environ.copy()
        env["HF_HUB_OFFLINE"] = "1"
        
        try:
            subprocess.run(cmd, env=env, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Failed to view episode {ep_idx}: {e}")
            response = input(f"\nContinue to next episode? (y/n): ")
            if response.lower() != 'y':
                break
            continue
        
        if ep_idx < num_episodes - 1:
            response = input(f"\nContinue to next episode? (y/n): ")
            if response.lower() != 'y':
                break
        else:
            print(f"\n✅ Finished viewing all episodes!")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["-h", "--help"]:
        print("Usage: python view_all_episodes.py [repo_id] [root]")
        print("\nExamples:")
        print("  python view_all_episodes.py data/pick_small_cube_1_20eps")
        print("  python view_all_episodes.py data/my_dataset /path/to/lerobot/datasets")
        print("\nDefaults:")
        print("  repo_id: data/test_calibration_teleop_1")
        print("  root: ~/.cache/huggingface/lerobot")
        sys.exit(0)
    
    repo_id = sys.argv[1] if len(sys.argv) > 1 else "data/test_calibration_teleop_1"
    root = sys.argv[2] if len(sys.argv) > 2 else None
    
    view_all_episodes(repo_id, root)


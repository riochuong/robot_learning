#!/usr/bin/env python
"""Test video loading to verify frames are correct for each episode."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"

from pathlib import Path
import numpy as np
from view_dataset_local import get_episode_video_info, load_video_frames

def test_video_loading():
    """Test that different episodes load different video frames."""
    dataset_path = Path.home() / ".cache/huggingface/lerobot/data/pick_small_cube_1_20eps"
    video_key = "observation.images.scene"
    
    print("Testing video frame loading consistency...")
    print("="*70)
    
    episodes_to_test = [0, 1, 2]
    episode_first_frames = {}
    
    for ep in episodes_to_test:
        print(f"\nTesting Episode {ep}:")
        
        # Load frames
        frames = load_video_frames(dataset_path, video_key, ep, debug=True)
        
        if frames and len(frames) > 0:
            # Store first frame for comparison
            first_frame = frames[0]
            episode_first_frames[ep] = first_frame
            
            # Compute frame statistics
            mean_val = first_frame.mean()
            std_val = first_frame.std()
            
            print(f"  First frame stats: mean={mean_val:.2f}, std={std_val:.2f}")
            print(f"  Total frames loaded: {len(frames)}")
        else:
            print(f"  ❌ Failed to load frames")
    
    # Compare first frames between episodes
    print("\n" + "="*70)
    print("Comparing first frames between episodes:")
    print("="*70)
    
    if len(episode_first_frames) >= 2:
        ep_keys = list(episode_first_frames.keys())
        for i in range(len(ep_keys)):
            for j in range(i+1, len(ep_keys)):
                ep1, ep2 = ep_keys[i], ep_keys[j]
                frame1 = episode_first_frames[ep1]
                frame2 = episode_first_frames[ep2]
                
                # Calculate difference
                diff = np.abs(frame1.astype(float) - frame2.astype(float))
                mean_diff = diff.mean()
                max_diff = diff.max()
                
                # Check if frames are identical (they shouldn't be for different episodes)
                identical = np.array_equal(frame1, frame2)
                
                print(f"\nEpisode {ep1} vs Episode {ep2}:")
                print(f"  Mean pixel difference: {mean_diff:.2f}")
                print(f"  Max pixel difference: {max_diff:.0f}")
                print(f"  Frames identical: {identical}")
                
                if identical:
                    print(f"  ⚠️  WARNING: Episodes {ep1} and {ep2} have identical first frames!")
                    print(f"  This suggests video loading may be incorrect.")
                else:
                    print(f"  ✅ Frames are different (as expected)")
    
    print("\n" + "="*70)
    print("Test complete!")

if __name__ == "__main__":
    test_video_loading()


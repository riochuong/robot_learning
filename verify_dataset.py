#!/usr/bin/env python
"""Verify LeRobot dataset structure and metadata.

Usage:
    python verify_dataset.py <repo_id> [--root path/to/root]
    
Examples:
    python verify_dataset.py dataset/pick_and_place_small_cube
    python verify_dataset.py data/my_dataset --root ~/.cache/huggingface/lerobot
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def check_video_file(video_path: Path) -> dict:
    """Get video file info using ffprobe."""
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-count_packets',
            '-show_entries', 'stream=nb_read_packets,duration',
            '-of', 'json',
            str(video_path)
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            data = json.loads(result.stdout)
            if 'streams' in data and len(data['streams']) > 0:
                stream = data['streams'][0]
                return {
                    'frames': int(stream.get('nb_read_packets', 0)),
                    'duration': float(stream.get('duration', 0)),
                    'exists': True
                }
    except Exception:
        pass
    
    return {'exists': False, 'frames': 0, 'duration': 0}


def verify_dataset(repo_id: str, root: Path) -> dict:
    """Verify dataset and return detailed information."""
    dataset_path = root / repo_id
    
    result = {
        'repo_id': repo_id,
        'path': str(dataset_path),
        'exists': dataset_path.exists(),
        'valid': False,
        'issues': [],
        'warnings': [],
        'info': {},
    }
    
    if not dataset_path.exists():
        result['issues'].append(f"Dataset not found at {dataset_path}")
        return result
    
    # Check info.json
    info_file = dataset_path / 'meta/info.json'
    if info_file.exists():
        with open(info_file) as f:
            info = json.load(f)
        
        result['info'] = {
            'total_episodes': info.get('total_episodes', 0),
            'total_frames': info.get('total_frames', 0),
            'fps': info.get('fps', 30),
            'codebase_version': info.get('codebase_version', 'unknown'),
            'robot_type': info.get('robot_type', 'unknown'),
        }
        
        # Check for video features
        video_features = {k: v for k, v in info.get('features', {}).items() 
                         if v.get('dtype') == 'video'}
        result['info']['video_features'] = list(video_features.keys())
        result['info']['has_videos'] = len(video_features) > 0
    else:
        result['issues'].append("Missing meta/info.json")
        return result
    
    # Check episode metadata
    episodes_dir = dataset_path / 'meta/episodes/chunk-000'
    if episodes_dir.exists():
        try:
            ep_files = sorted(episodes_dir.glob('*.parquet'))
            if ep_files:
                ep_df = pd.concat([pd.read_parquet(f) for f in ep_files], ignore_index=True)
                
                result['info']['episodes_in_metadata'] = sorted(ep_df['episode_index'].unique().tolist())
                result['info']['num_episodes_metadata'] = len(ep_df)
                
                # Check for video mapping columns
                video_cols = [col for col in ep_df.columns if 'video' in col.lower() and 'observation' in col]
                result['info']['has_video_metadata'] = len(video_cols) > 0
                result['info']['video_metadata_columns'] = len(video_cols)
                
                if result['info']['has_videos'] and not result['info']['has_video_metadata']:
                    result['issues'].append("info.json claims videos exist but episode metadata has no video mapping columns")
                
                # Check first episode for video info
                if result['info']['has_video_metadata']:
                    first_ep = ep_df.iloc[0]
                    result['info']['sample_video_mapping'] = {}
                    
                    for video_key in result['info']['video_features']:
                        file_col = f'videos/{video_key}/file_index'
                        from_col = f'videos/{video_key}/from_timestamp'
                        to_col = f'videos/{video_key}/to_timestamp'
                        
                        if file_col in ep_df.columns and pd.notna(first_ep[file_col]):
                            result['info']['sample_video_mapping'][video_key] = {
                                'file_index': int(first_ep[file_col]),
                                'from_timestamp': float(first_ep[from_col]),
                                'to_timestamp': float(first_ep[to_col]),
                                'duration': float(first_ep[to_col] - first_ep[from_col])
                            }
            else:
                result['issues'].append("No episode metadata files found")
        except Exception as e:
            result['issues'].append(f"Error reading episode metadata: {e}")
    else:
        result['issues'].append("Missing meta/episodes directory")
    
    # Check data parquet
    data_dir = dataset_path / 'data/chunk-000'
    if data_dir.exists():
        try:
            data_files = sorted(data_dir.glob('*.parquet'))
            if data_files:
                # Just read first file to check structure
                df = pd.read_parquet(data_files[0])
                
                result['info']['data_columns'] = list(df.columns)
                result['info']['episodes_in_data'] = sorted(df['episode_index'].unique().tolist())
                
                # Check if video columns are in data parquet (they shouldn't be)
                has_video_in_data = any('image' in col.lower() for col in df.columns)
                result['info']['has_video_columns_in_data'] = has_video_in_data
                
                if has_video_in_data:
                    result['warnings'].append("Data parquet has video columns (unusual for modern LeRobot datasets)")
            else:
                result['issues'].append("No data parquet files found")
        except Exception as e:
            result['issues'].append(f"Error reading data parquet: {e}")
    else:
        result['issues'].append("Missing data directory")
    
    # Check video files
    if result['info'].get('has_videos'):
        result['info']['video_files'] = {}
        
        for video_key in result['info']['video_features']:
            video_dir = dataset_path / 'videos' / video_key / 'chunk-000'
            
            if video_dir.exists():
                video_files = sorted(video_dir.glob('*.mp4'))
                
                total_frames = 0
                total_size = 0
                
                for vf in video_files:
                    total_size += vf.stat().st_size
                    # Only check first video file for speed
                    if vf == video_files[0]:
                        info = check_video_file(vf)
                        if info['exists']:
                            total_frames = info['frames']  # Approximate from first file
                
                result['info']['video_files'][video_key] = {
                    'count': len(video_files),
                    'total_size_mb': total_size / 1024 / 1024,
                    'directory': str(video_dir),
                    'sample_frame_count': total_frames
                }
            else:
                result['issues'].append(f"Missing video directory for {video_key}")
    
    # Determine if dataset is valid
    critical_issues = [
        'Missing meta/info.json',
        'Missing meta/episodes directory',
        'Missing data directory',
        'No episode metadata files found',
        'No data parquet files found'
    ]
    
    has_critical_issue = any(issue in result['issues'] for issue in critical_issues)
    
    if result['info'].get('has_videos'):
        # For video datasets, also need video metadata
        if not result['info'].get('has_video_metadata'):
            has_critical_issue = True
    
    result['valid'] = not has_critical_issue
    
    return result


def print_report(result: dict):
    """Print formatted verification report."""
    print("=" * 70)
    print("LEROBOT DATASET VERIFICATION REPORT")
    print("=" * 70)
    print(f"\nDataset: {result['repo_id']}")
    print(f"Path: {result['path']}")
    print(f"Exists: {'✅ Yes' if result['exists'] else '❌ No'}")
    
    if not result['exists']:
        return
    
    # Overall status
    print(f"\nStatus: ", end='')
    if result['valid']:
        print("✅ VALID - Ready for training")
    else:
        print("❌ INVALID - Has issues")
    
    # Dataset info
    if result['info']:
        print(f"\n{'─' * 70}")
        print("DATASET INFORMATION")
        print('─' * 70)
        
        info = result['info']
        print(f"  Episodes: {info.get('total_episodes', 'unknown')}")
        print(f"  Frames: {info.get('total_frames', 'unknown')}")
        print(f"  FPS: {info.get('fps', 'unknown')}")
        print(f"  Robot: {info.get('robot_type', 'unknown')}")
        print(f"  Version: {info.get('codebase_version', 'unknown')}")
        
        # Video features
        if info.get('has_videos'):
            print(f"\n  Video Features:")
            for vf in info.get('video_features', []):
                print(f"    • {vf}")
        else:
            print(f"\n  Video Features: None (joint-space only dataset)")
        
        # Episode metadata
        if 'episodes_in_metadata' in info:
            episodes = info['episodes_in_metadata']
            print(f"\n  Episodes in metadata: {len(episodes)}")
            if len(episodes) <= 10:
                print(f"    Episodes: {episodes}")
            else:
                print(f"    Episodes: {episodes[:5]} ... {episodes[-2:]}")
        
        # Video metadata
        if info.get('has_video_metadata'):
            print(f"\n  ✅ Video Mapping: Present ({info.get('video_metadata_columns')} columns)")
            
            if 'sample_video_mapping' in info:
                print(f"\n  Sample Episode 0 Video Mapping:")
                for key, mapping in info['sample_video_mapping'].items():
                    camera_name = key.split('.')[-1]
                    print(f"    {camera_name}:")
                    print(f"      file: file-{mapping['file_index']:03d}.mp4")
                    print(f"      time: {mapping['from_timestamp']:.2f}s - {mapping['to_timestamp']:.2f}s")
                    print(f"      duration: {mapping['duration']:.2f}s")
        elif info.get('has_videos'):
            print(f"\n  ❌ Video Mapping: Missing")
        
        # Data structure
        if 'data_columns' in info:
            print(f"\n  Data Parquet Columns: {len(info['data_columns'])}")
            print(f"    {', '.join(info['data_columns'][:6])}", end='')
            if len(info['data_columns']) > 6:
                print(", ...")
            else:
                print()
            
            has_video_in_data = info.get('has_video_columns_in_data', False)
            if has_video_in_data:
                print(f"    ⚠️  Has video columns in data parquet (unusual)")
            else:
                print(f"    ✅ No video columns in data (correct - mapped via episode metadata)")
        
        # Video files
        if 'video_files' in info:
            print(f"\n  Video Files:")
            for key, vinfo in info['video_files'].items():
                camera_name = key.split('.')[-1]
                print(f"    {camera_name}:")
                print(f"      files: {vinfo['count']}")
                print(f"      size: {vinfo['total_size_mb']:.1f} MB")
                if vinfo.get('sample_frame_count'):
                    print(f"      frames (est): {vinfo['sample_frame_count']}")
    
    # Issues
    if result['issues']:
        print(f"\n{'─' * 70}")
        print("❌ ISSUES")
        print('─' * 70)
        for issue in result['issues']:
            print(f"  • {issue}")
    
    # Warnings
    if result['warnings']:
        print(f"\n{'─' * 70}")
        print("⚠️  WARNINGS")
        print('─' * 70)
        for warning in result['warnings']:
            print(f"  • {warning}")
    
    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print('=' * 70)
    
    if result['valid']:
        print("\n✅ Dataset is valid and ready for use!")
        print("\nYou can:")
        print("  • Train policies with this dataset")
        print("  • View with: lerobot-dataset-viz --repo-id", result['repo_id'])
        print("  • View with: python view_dataset_local.py", result['repo_id'], "0")
        
        if result['info'].get('has_videos'):
            print("\n  Training will have access to:")
            print("    ✅ Joint positions and actions")
            print("    ✅ Camera images")
        else:
            print("\n  Training will have access to:")
            print("    ✅ Joint positions and actions")
            print("    ❌ No camera images (joint-space only)")
    else:
        print("\n❌ Dataset has issues and may not work correctly!")
        print("\nRecommended actions:")
        print("  1. Fix the issues listed above")
        print("  2. Or re-record the dataset")
        print("  3. Verify metadata structure matches LeRobot requirements")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Verify LeRobot dataset structure and metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python verify_dataset.py dataset/pick_and_place_small_cube
  python verify_dataset.py data/my_dataset --root ~/.cache/huggingface/lerobot
  python verify_dataset.py lerobot/pusht --root /path/to/datasets
        """
    )
    
    parser.add_argument(
        'repo_id',
        help='Dataset repository ID (e.g., dataset/my_dataset)'
    )
    
    parser.add_argument(
        '--root',
        default=str(Path.home() / '.cache/huggingface/lerobot'),
        help='Root directory where datasets are stored (default: ~/.cache/huggingface/lerobot)'
    )
    
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON instead of formatted report'
    )
    
    args = parser.parse_args()
    
    root = Path(args.root)
    
    if not root.exists():
        print(f"❌ Root directory not found: {root}")
        print(f"\nAvailable datasets:")
        # Try to find common locations
        for common_root in [
            Path.home() / '.cache/huggingface/lerobot',
            Path('/tmp/lerobot'),
        ]:
            if common_root.exists():
                print(f"\n  In {common_root}:")
                for item in sorted(common_root.rglob('*/meta/info.json')):
                    dataset_path = item.parent.parent
                    rel_path = dataset_path.relative_to(common_root)
                    print(f"    • {rel_path}")
        sys.exit(1)
    
    result = verify_dataset(args.repo_id, root)
    
    if args.json:
        # Output as JSON
        import json
        print(json.dumps(result, indent=2, default=str))
    else:
        # Output formatted report
        print_report(result)
    
    # Exit code: 0 if valid, 1 if invalid
    sys.exit(0 if result['valid'] else 1)


if __name__ == '__main__':
    main()


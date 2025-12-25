## ROBOT LEARNING EXPERIMENTS

### RECORD TELEOP NO CAMERAS
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm_so_101 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm_so_100 \
    --dataset.repo_id=data/test_calibration_teleop \
    --dataset.push_to_hub=False \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=30 \
    --dataset.single_task="Calibration test trajectory"

### VERIFY DATASET METADATA
# Quick check if your dataset is valid and ready for training
uv run python verify_dataset.py dataset/pick_and_place_small_cube

# Options:
#   --root /path/to/datasets  # Custom dataset root
#   --json                    # Output as JSON

### EDIT DATASETS (Remove bad episodes, split, merge, etc.)
# See DATASET_EDITING_GUIDE.md for full documentation

# Delete bad episodes (e.g., episodes 5, 12, 18 were bad demonstrations)
lerobot-edit-dataset \
    --repo_id data/pick_small_cube_1_20eps \
    --new_repo_id data/pick_small_cube_cleaned \
    --operation.type delete_episodes \
    --operation.episode_indices "[5, 12, 18]"

# Split into train/val (80%/20%)
lerobot-edit-dataset \
    --repo_id data/pick_small_cube_cleaned \
    --operation.type split \
    --operation.splits '{"train": 0.8, "val": 0.2}'

# Merge multiple recording sessions
lerobot-edit-dataset \
    --repo_id data/merged_dataset \
    --operation.type merge \
    --operation.repo_ids "['data/session1', 'data/session2']"

# Remove wrist camera to save space
lerobot-edit-dataset \
    --repo_id data/my_dataset \
    --new_repo_id data/my_dataset_scene_only \
    --operation.type remove_feature \
    --operation.feature_names "['observation.images.wrist']"

### RECORD TELEOP WITH TWO CAMERAS
# --display_data=True enables Rerun.io live visualization during recording!
# This shows: camera feeds, joint positions, actions in real-time 3D view
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm_so_101 \
    --robot.cameras='{"scene": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30}, "wrist": {"type": "opencv", "index_or_path": "/dev/video4", "width": 640, "height": 480, "fps": 30}}' \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm_so_100 \
    --dataset.repo_id=data/pick_small_cube_1_20eps \
    --dataset.push_to_hub=False \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=10 \
    --display_data=True \
    --dataset.single_task="Pick and place demonstration"


### TEST CALIBRATION

uv run python test_calibration.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm_so_101 \
    --dataset.repo_id=data/test_calibration_teleop_1 \
    --dataset.skip_record=True \
    --dataset.test_episodes=0,1,2


### CAMERA TEST
uv run python camera_test.py 2 4

### VISUALIZE RECORDED DATASET (LOCAL ONLY)
# Easy wrapper script (recommended):
uv run lerobot-dataset-viz --repo-id=dataset/pick_and_place_small_cube --episode-index 23

# Or use Python directly:
# Browse through ALL episodes sequentially (recommended for quality checking):
uv run python view_dataset_local.py data/pick_small_cube_1_20eps --all

# Start browsing from episode 10:
uv run python view_dataset_local.py data/pick_small_cube_1_20eps 10 --all

# View single episode:
uv run python view_dataset_local.py data/pick_small_cube_1_20eps 0

# Enable debug mode to verify frame loading:
DEBUG_VIDEO=1 uv run python view_dataset_local.py data/pick_small_cube_1_20eps 0

### TELEOP


sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm_so_101 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm_so_100


### VERIFY DATASET 
uv run python verify_dataset.py dataset/pick_and_place_small_cube


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


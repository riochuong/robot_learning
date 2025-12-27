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
    --teleop.id=my_leader_arm_so_100 \
    --robot.cameras='{"scene": {"type": "intelrealsense", "serial_number_or_name": "247122073488", "width": 848, "height": 480, "fps": 30, use_depth: true}, "wrist": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30}}' \
    --display_data=true


### VERIFY DATASET 
uv run python verify_dataset.py dataset/pick_and_place_small_cube

### FIND CAMERAS 
```
lerobot-find-cameras realsense
lerobot-find-cameras
```


### CALIBRATE LEADER ARM
# Calibration happens automatically when you first use the leader arm
# It will prompt you through the process when you run teleoperation or recording

# Method 1: Via teleoperation (recommended for first-time setup)
sudo chmod 666 /dev/ttyACM1
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_follower_arm_so_101 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm_so_100

# Method 2: Calibration triggered during recording
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_leader_arm_so_100 \
    --dataset.repo_id=data/test \
    --dataset.num_episodes=1

# Calibration file location:
# ~/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader/my_leader_arm_so_100.json

# Recalibrate: Delete the file above and run teleoperation again
# Or: Type 'c' when prompted to use existing calibration

# See LEADER_ARM_CALIBRATION.md for detailed guide

### CONFIGURE MOTOR IDs
# Reconfigure motor IDs if motors were replaced or IDs are conflicting

# For leader arm
sudo chmod 666 /dev/ttyACM1
lerobot-setup-motors \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM1

# For follower arm
sudo chmod 666 /dev/ttyACM0
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0

# IMPORTANT: Connect ONE motor at a time during setup!
# The script will prompt you for each motor in reverse order:
# gripper → wrist_roll → wrist_flex → elbow_flex → shoulder_lift → shoulder_pan

# After configuring motor IDs, recalibrate:
rm ~/.cache/huggingface/lerobot/calibration/teleoperators/so100_leader/my_leader_arm_so_100.json
# Then run teleoperation to trigger calibration

# See MOTOR_ID_CONFIGURATION.md for detailed guide

### CAMERA SETUP
```
 8996  v4l2-ctl -d /dev/video4 --list-ctrls
 9003  v4l2-ctl -d /dev/video4 -c focus_absolute=0
 9004  v4l2-ctl -d /dev/video4 --list-ctrls
 9014  v4l2-ctl -d /dev/video4 --list-ctrls
 9015  v4l2-ctl -d /dev/video4 -c exposure_dynamic_framerate=0
 9016  v4l2-ctl -d /dev/video4 -c auto_exposure=1
 9021  v4l2-ctl -d /dev/video4 -c exposure_time_absolute=300

 8997  v4l2-ctl -d /dev/video2 --list-ctrls
 8998  v4l2-ctl -d /dev/video2 -c focus_automatic_continuous=0
 9000  v4l2-ctl -d /dev/video2 -c focus_absolute=0
 9005  v4l2-ctl -d /dev/video2 -c exposure_dynamic_framerate=0
 9006  v4l2-ctl -d /dev/video2 -c auto_exposure=1
 9007  v4l2-ctl -d /dev/video2 -c exposure_time_absolute=150
 9009  v4l2-ctl -d /dev/video2 -c exposure_time_absolute=200
 9011  v4l2-ctl -d /dev/video2 -c exposure_time_absolute=300
 9013  v4l2-ctl -d /dev/video2 --list-ctrls
 9017  v4l2-ctl -d /dev/video2 -c exposure_time_absolute=150
 9019  v4l2-ctl -d /dev/video2 -c exposure_time_absolute=300
 ```

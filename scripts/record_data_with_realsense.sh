set -x 
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

# Parse arguments:
# $1 = task instruction (required)
# $2 = resume flag (optional, "--resume" or empty)
TASK_INSTRUCTION="$1"
RESUME_FLAG=""

# Check if task instruction was provided
if [ -z "$TASK_INSTRUCTION" ]; then
    echo "Error: Task instruction is required as first argument"
    echo "Usage: $0 \"task instruction\" [--resume]"
    echo "Example: $0 \"Pick the black cube and place it in the box\" --resume"
    exit 1
fi

# Check if --resume was passed as second argument
if [ "$2" = "--resume" ]; then
    RESUME_FLAG="--resume=True"
fi

uv run lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=my_follower_arm_so_101 \
    --robot.cameras='{"scene": {"type": "intelrealsense", "serial_number_or_name": "247122073488", "width": 848, "height": 480, "fps": 30, use_depth: false}, "wrist": {"type": "opencv", "index_or_path": "/dev/video2", "width": 640, "height": 480, "fps": 30}}' \
    --teleop.type=so100_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=my_leader_arm_so_100 \
    --dataset.repo_id=dataset/pick_and_place_realsense_cam \
    --dataset.push_to_hub=False \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --display_data=True \
    --dataset.single_task="$TASK_INSTRUCTION" \
    $RESUME_FLAG

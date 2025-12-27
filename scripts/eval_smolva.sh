OUTPUT_DIR=$1
DATASET_ROOT=/home/cd105-dgx/workspace/robot_training/dataset/smolvla_so101_eval
rm -rf $DATASET_ROOT
sudo chmod 666 /dev/ttyACM0
lerobot-record \
  --policy.path=/home/cd105-dgx/workspace/robot_learning/robot_training/models/smolvla_pick_small_cube/checkpoints/020000/pretrained_model \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.cameras='{
    camera1: {type: intelrealsense, serial_number_or_name: 247122073488, width: 848, height: 480, fps: 30, use_depth: false, reset_device: true},
    camera2: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: MJPG}
  }' \
  --robot.id=my_follower_arm_so_101 \
  --dataset.repo_id=chuongdao/eval_smolvla_so101 \
  --dataset.root=$DATASET_ROOT \
  --dataset.num_episodes=10 \
  --dataset.single_task="Pick the black cube and place it in the box" \
  --dataset.episode_time_s=300 \
  --policy.device=cuda \
  --display_data=true \
  --dataset.push_to_hub=false

                                 

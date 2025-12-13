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


### TEST CALIBRATION 

uv run python test_calibration.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --dataset.repo_id=data/test_calibration_teleop_1 \
    --dataset.skip_record=True \
    --dataset.test_episodes=0,1,2
 

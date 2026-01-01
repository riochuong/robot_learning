set -x 

huggingface-cli login --token $(cat HF_TOKEN)
wandb login $(cat WANDB_TOKEN)

OUTPUT_DIR=./models/smolvla_pick_and_place_realsense_cam
#rm -rf ${OUTPUT_DIR}

lerobot-train \
  --dataset.repo_id=pick_and_place_realsense_cam_2 \
  --dataset.root=/home/ubuntu/robot_learning/dataset/pick_and_place_realsense_cam_2 \
  --steps=20000 \
  --resume=true \
  --output_dir=${OUTPUT_DIR} \
  --config_path=models/smolvla_pick_and_place_realsense_cam/checkpoints/last/pretrained_model/train_config.json





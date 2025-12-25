set -x 

huggingface-cli login --token $(cat HF_TOKEN)
wandb login $(cat WANDB_TOKEN)

OUTPUT_DIR=outputs/smolvla_pick_small_cube
rm -rf ${OUTPUT_DIR}

lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=pick_small_cube \
  --dataset.video_backend=torchcodec \
  --dataset.root=/home/cd105-dgx/workspace/robot_training/dataset/pick_small_cube \
  --batch_size=128 \
  --steps=10000 \
  --save_freq=5000 \
  --output_dir=${OUTPUT_DIR} \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --rename_map="{\"observation.images.scene\": \"observation.images.camera1\", \"observation.images.wrist\": \"observation.images.camera2\"}"



set -x 

export TORCHINDUCTOR_CUDAGRAPHS=0
huggingface-cli login --token $(cat HF_TOKEN)
wandb login $(cat WANDB_TOKEN)

# CHANGE lerobot_train.py  TO THIS:
#dataloader = torch.utils.data.DataLoader(
#    dataset,
#...    
#    prefetch_factor=4,            # Force 4 (only works if num_workers > 0)
#    persistent_workers=True,      # Crucial for CPU bottleneck!
#)

OUTPUT_DIR=outputs/pi05_pick_and_place
rm -rf ${OUTPUT_DIR}

lerobot-train \
  --policy.path=lerobot/pi05_base \
  --dataset.repo_id=pick_and_place_realsense_cam \
  --dataset.video_backend=torchcodec \
  --dataset.root=/home/ubuntu/robot_learning/dataset/pick_and_place_realsense_cam \
  --dataset.video_backend=torchcodec \
  --batch_size=32 \
  --steps=20000 \
  --num_workers=24 \
  --save_freq=1000 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --output_dir=${OUTPUT_DIR} \
  --policy.device=cuda \
  --policy.dtype=bfloat16 \
  --policy.push_to_hub=false \
  --wandb.enable=true \
  --rename_map="{\"observation.images.scene\": \"observation.images.base_0_rgb\", \"observation.images.wrist\": \"observation.images.left_wrist_0_rgb\"}"



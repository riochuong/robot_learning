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

OUTPUT_DIR=outputs/groot_1_5
rm -rf ${OUTPUT_DIR}


lerobot-train \
  --policy.type=groot \
  --dataset.root=./dataset/pick_and_place_realsense_combined_dataset \
  --dataset.repo_id=pick_and_place_realsense_combined_dataset \
  --output_dir=${OUTPUT_DIR} \
  --dataset.video_backend=torchcodec \
  --batch_size=64 \
  --steps=40000 \
  --save_freq=5000 \
  --policy.push_to_hub=false \
  --log_freq=500 \
  --policy.tune_projector=true \
  --policy.tune_diffusion_model=true \
  --policy.tune_llm=true \
  --policy.tune_visual=true \
  --wandb.enable=true \
  --policy.device=cuda



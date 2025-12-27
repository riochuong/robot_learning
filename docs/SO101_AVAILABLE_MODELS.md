# Available Models/Policies for SO101 Arm

All policies in LeRobot are **robot-agnostic** and automatically adapt to your robot's action and observation space. This means any policy type can be used with SO101, as long as the dataset format matches.

## ✅ All Available Policy Types

Based on the codebase, here are all the policy types available in LeRobot that can work with SO101:

### 1. **ACT** (Action Chunking Transformer) ⭐
- **Policy Type**: `act`
- **Status**: ✅ Fully supported
- **Example Usage**: Used in SO101 tutorials
- **Best For**: Fast single-step inference, proven performance on manipulation tasks
- **Training Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=act \
    --output_dir=outputs/train/act_so101_test \
    --job_name=act_so101_test \
    --policy.device=cuda
  ```

### 2. **Diffusion Policy** ⭐
- **Policy Type**: `diffusion`
- **Status**: ✅ Fully supported  
- **Example Usage**: Tutorial examples use `lerobot/svla_so101_pickplace` dataset
- **Best For**: Multi-modal action distributions, versatile and proven
- **Training Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=diffusion \
    --output_dir=outputs/train/diffusion_so101_test \
    --job_name=diffusion_so101_test \
    --policy.device=cuda
  ```

### 3. **X-VLA** (Cross-Embodiment VLA)
- **Policy Type**: `xvla`
- **Status**: ✅ Supported (special action mode for bimanual SO101)
- **Performance**: **93% LIBERO**, **100% real-world cloth folding**
- **Special Features**: 
  - Has `so101_bimanual` action mode (pads 12D real → 20D model)
  - Cross-embodiment via soft prompts (rapid adaptation)
  - Florence-2 backbone with pluggable action spaces
  - Multi-view camera support
- **Best For**: Cross-robot transfer, language-conditioned tasks, bimanual SO101
- **Training Command**:
  ```bash
  # Phase II: Fine-tune on SO101 (after pretraining)
  lerobot-train \
    --dataset.repo_id=${HF_USER}/bimanual_so101 \
    --policy.type=xvla \
    --policy.path="lerobot/xvla-base" \
    --policy.action_mode=so101_bimanual \
    --output_dir=outputs/train/xvla_so101 \
    --job_name=xvla_so101_training \
    --policy.device=cuda \
    --policy.freeze_vision_encoder=false \
    --policy.freeze_language_encoder=false \
    --policy.train_soft_prompts=true \
    --steps=3000
  ```

### 4. **SmolVLA**
- **Policy Type**: `smolvla`
- **Status**: ✅ Supported
- **Best For**: Language-conditioned tasks with limited compute, lightweight
- **Training Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=smolvla \
    --output_dir=outputs/train/smolvla_so101_test \
    --job_name=smolvla_so101_test \
    --policy.device=cuda
  ```

### 5. **Pi0** (PaliGemma-based)
- **Policy Type**: `pi0`
- **Status**: ✅ Supported
- **Best For**: High-quality VLA with strong generalization, complex manipulation
- **Training Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=pi0 \
    --output_dir=outputs/train/pi0_so101_test \
    --job_name=pi0_so101_test \
    --policy.device=cuda
  ```

### 6. **Pi0.5** (Extended Pi0)
- **Policy Type**: `pi05`
- **Status**: ✅ Supported
- **Best For**: Open-world generalization, 200-token prompts, AdaRMS conditioning
- **Training Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=pi05 \
    --output_dir=outputs/train/pi05_so101_test \
    --job_name=pi05_so101_test \
    --policy.device=cuda
  ```

### 7. **Groot** (NVIDIA GR00T Integration)
- **Policy Type**: `groot`
- **Status**: ✅ Supported
- **Best For**: Multi-embodiment support, transferring policies across robots
- **Training Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=groot \
    --output_dir=outputs/train/groot_so101_test \
    --job_name=groot_so101_test \
    --policy.device=cuda
  ```

### 8. **TDMPC** (Temporal Difference Model Predictive Control)
- **Policy Type**: `tdmpc`
- **Status**: ✅ Supported
- **Best For**: Model-based reinforcement learning
- **Training Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=tdmpc \
    --output_dir=outputs/train/tdmpc_so101_test \
    --job_name=tdmpc_so101_test \
    --policy.device=cuda
  ```

### 9. **VQBeT** (Vector Quantized Behavior Transformer)
- **Policy Type**: `vqbet`
- **Status**: ✅ Supported
- **Best For**: Discrete action spaces, behavior cloning
- **Training Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=vqbet \
    --output_dir=outputs/train/vqbet_so101_test \
    --job_name=vqbet_so101_test \
    --policy.device=cuda
  ```

### 10. **Wall-X**
- **Policy Type**: `wall_x`
- **Status**: ✅ Supported
- **Best For**: Vision-language-action models with Qwen2.5-VL
- **Training Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=wall_x \
    --output_dir=outputs/train/wall_x_so101_test \
    --job_name=wall_x_so101_test \
    --policy.device=cuda
  ```

### 11. **SARM** (Sparse Action Reward Model)
- **Policy Type**: `sarm`
- **Status**: ✅ Supported
- **Best For**: Reward learning, sparse reward scenarios
- **Training Command**:
  ```bash
  lerobot-train \
    --dataset.repo_id=${HF_USER}/so101_test \
    --policy.type=sarm \
    --output_dir=outputs/train/sarm_so101_test \
    --job_name=sarm_so101_test \
    --policy.device=cuda
  ```

## 🎯 Recommended Models for SO101

### For Maximum Generalization:
1. **Pi0.5** (`pi05`) - 97.5% LIBERO benchmark, best overall generalization
2. **X-VLA** (`xvla`) - 93% LIBERO, cross-embodiment, has SO101 bimanual mode
3. **SmolVLA** (`smolvla`) - Lightweight VLA, proven on SO101 real-world

### For Fast Training/Quick Start:
1. **ACT** (`act`) - Most commonly used, well-documented, fast inference
2. **Diffusion** (`diffusion`) - Versatile, proven performance, good for complex tasks

### For Multi-Embodiment Research:
1. **Groot** (`groot`) - 87% LIBERO, NVIDIA's multi-embodiment foundation model

See `MODELS_WITH_STRONG_GENERALIZATION.md` for detailed performance comparisons.

## 📚 Example Datasets

- `lerobot/svla_so101_pickplace` - Example SO101 dataset used in tutorials

## 💡 Key Points

- **All policies automatically adapt** to SO101's action/observation space
- **Action space**: 6 motors (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
- **Observation space**: Motor positions + cameras (if configured)
- Policies extract the correct dimensions from your dataset automatically

## 📖 Documentation Links

- ACT: `docs/source/act.mdx`
- Diffusion: `docs/source/policy_diffusion_README.md`
- X-VLA: `docs/source/xvla.mdx`
- SmolVLA: `docs/source/smolvla.mdx`
- Pi0: `docs/source/pi0.mdx`
- SO101 Robot: `docs/source/so101.mdx`
- General IL Tutorial: `docs/source/il_robots.mdx`


# 🤖 LeRobot Training Pipeline Deep Dive

> Understanding how data flows from dataset to model for SmolVLA, Pi0, ACT, Groot, and Diffusion policies

## Table of Contents

1. [Overview: The Training Loop](#1-overview-the-training-loop)
2. [DataLoader & Dataset Loading](#2-dataloader--dataset-loading)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [SmolVLA: Vision-Language-Action Model](#4-smolvla-vision-language-action-model)
5. [Pi0: PaliGemma + Expert Model](#5-pi0-paligemma--expert-model)
6. [Pi0.5: No State, AdaRMS Conditioning](#6-pi05-no-state-adarms-conditioning)
7. [Pi0.5-Fast: Optimized for Speed](#7-pi05-fast-optimized-for-speed)
8. [ACT: Action Chunking Transformer](#8-act-action-chunking-transformer)
9. [Groot: NVIDIA GR00T Integration](#9-groot-nvidia-groot-integration)
10. [Diffusion Policy: Denoising Actions](#10-diffusion-policy-denoising-actions)
11. [Model Comparison Summary](#11-model-comparison-summary)

---

## 1. Overview: The Training Loop

The LeRobot training pipeline follows a consistent pattern across all policies:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           LEROBOT TRAINING PIPELINE                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Dataset ──► DataLoader ──► Preprocessor ──► Policy.forward() ──► Loss         │
│  (Parquet +   (Batching +   (Normalize +   (Model-specific)      (L1/MSE/      │
│   Videos)      Shuffling)    Tokenize)                            Flow Match)   │
│                                                                       │         │
│                                                                       ▼         │
│  Metrics ◄──── Optimizer ◄──── Backward Pass ◄────────────────────────┘         │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Training Script Entry Point

The main training loop is in `src/lerobot/scripts/lerobot_train.py`:

```python
# Simplified training loop
for _ in range(step, cfg.steps):
    # 1. Load batch from DataLoader
    batch = next(dl_iter)
    
    # 2. Apply preprocessing (normalize, tokenize, move to device)
    batch = preprocessor(batch)
    
    # 3. Forward pass through policy
    loss, output_dict = policy.forward(batch)
    
    # 4. Backward pass + optimizer step
    accelerator.backward(loss)
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    
    # 5. Logging and checkpointing
    if step % cfg.save_freq == 0:
        save_checkpoint(...)
```

---

## 2. DataLoader & Dataset Loading

### LeRobotDataset.__getitem__

When the DataLoader requests a sample, `LeRobotDataset.__getitem__` is called:

```python
def __getitem__(self, idx) -> dict:
    # 1. Load tabular data from HuggingFace dataset (Parquet)
    item = self.hf_dataset[idx]
    ep_idx = item["episode_index"].item()

    # 2. Handle temporal sequences (delta_indices for past/future frames)
    if self.delta_indices is not None:
        query_indices, padding = self._get_query_indices(idx, ep_idx)
        query_result = self._query_hf_dataset(query_indices)
        item = {**item, **padding}  # Add padding masks

    # 3. Decode video frames for visual observations
    if len(self.meta.video_keys) > 0:
        current_ts = item["timestamp"].item()
        query_timestamps = self._get_query_timestamps(current_ts, query_indices)
        video_frames = self._query_videos(query_timestamps, ep_idx)
        item = {**video_frames, **item}

    # 4. Apply image transforms (augmentation)
    if self.image_transforms is not None:
        for cam in self.meta.camera_keys:
            item[cam] = self.image_transforms(item[cam])

    return item
```

### What the DataLoader Returns

| Key | Shape | Description |
|-----|-------|-------------|
| `observation.state` | `(B, n_obs_steps, state_dim)` | Robot joint positions, velocities |
| `observation.images.cam1` | `(B, n_obs_steps, C, H, W)` | Camera images (normalized to [0, 1]) |
| `action` | `(B, chunk_size, action_dim)` | Target action sequence (ground truth) |
| `action_is_pad` | `(B, chunk_size)` | Boolean mask for padded actions |
| `task` | `List[str]` of length B | Language task description |
| `timestamp` | `(B,)` | Frame timestamp |
| `episode_index` | `(B,)` | Episode identifier |

---

## 3. Preprocessing Pipeline

Each policy has a `make_*_pre_post_processors()` function that creates a preprocessing pipeline:

```
Raw Batch from DataLoader
         │
         ▼
┌────────────────────────────────────┐
│ 1. RenameObservationsProcessorStep │  Rename keys to match pretrained config
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ 2. AddBatchDimensionProcessorStep  │  Ensure batch dim exists (for inference)
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ 3. TokenizerProcessorStep          │  Convert task string → token IDs (VLA models)
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ 4. DeviceProcessorStep             │  Move tensors to GPU (cuda/mps)
└────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────┐
│ 5. NormalizerProcessorStep         │  Normalize features based on dataset stats
│    - MEAN_STD: (x - mean) / std    │
│    - MIN_MAX: (x - min) / (max-min)│
│    - IDENTITY: no normalization    │
└────────────────────────────────────┘
         │
         ▼
    Preprocessed Batch
```

### Normalization Modes

```python
normalization_mapping = {
    "VISUAL": NormalizationMode.IDENTITY,     # Images: no normalization (already [0,1])
    "STATE": NormalizationMode.MEAN_STD,       # State: (x - μ) / σ
    "ACTION": NormalizationMode.MEAN_STD,      # Actions: (x - μ) / σ
}
```

---

## 4. SmolVLA: Vision-Language-Action Model

### Architecture Overview

SmolVLA embeds robot state in the **PREFIX** (not suffix like Pi0), concatenated with image and language embeddings.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              SmolVLA Architecture                            │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    Images ──► SigLIP Vision Encoder ──► Vision Projection ──┐               │
│                                                              │               │
│    Language ──► SmolVLM Tokenizer ──► Embedding Layer ───────┼──► PREFIX    │
│                                                              │    (embed_   │
│    State ──► Linear Projection (state_proj) ─────────────────┘     prefix)  │
│                                                                              │
│                              PREFIX EMBEDDINGS                               │
│         [img_emb₁, img_emb₂, ..., lang_emb, STATE_emb]                       │
│                              │                                               │
│                              ▼                                               │
│                     SmolVLM Transformer (Frozen)                             │
│                              │                                               │
│                              │ KV Cache                                      │
│                              ▼                                               │
│                 SUFFIX: [noisy_actions + timestep]                           │
│                              │        (embed_suffix - NO state)              │
│                              ▼                                               │
│                     ACTION EXPERT (Trainable)                                │
│                              │                                               │
│                              ▼                                               │
│                     Output Projection → Predicted Actions                    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Step-by-Step Forward Pass

#### Step 1: Prepare Images
```python
def prepare_images(self, batch):
    """Resize to 512x512 with padding, normalize to [-1, 1] for SigLIP."""
    for key in self.config.image_features:
        img = batch[key][:, -1, :, :, :]  # Take last observation step: (B, C, H, W)
        img = resize_with_pad(img, 512, 512, pad_value=0)
        img = img * 2.0 - 1.0  # Normalize from [0,1] to [-1,1]
```

#### Step 2: Prepare State
```python
def prepare_state(self, batch):
    """Pad state vector to max_state_dim (default: 32)."""
    state = batch["observation.state"][:, -1, :]  # (B, state_dim)
    state = pad_vector(state, self.config.max_state_dim)  # (B, 32)
```

#### Step 3: Embed Prefix (Images + Language + State)

> **Note:** Unlike Pi0 which puts state in SUFFIX, SmolVLA includes state in PREFIX.

```python
def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state):
    embs = []
    
    # 3a. Embed images with SigLIP vision encoder + normalize
    for img in images:
        img_emb = self.vlm_with_expert.embed_image(img)  # (B, num_patches, hidden_dim)
        img_emb = img_emb * math.sqrt(hidden_dim)  # Scale normalization
        embs.append(img_emb)
    
    # 3b. Embed language tokens + normalize
    lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
    lang_emb = lang_emb * math.sqrt(hidden_dim)  # Scale normalization
    embs.append(lang_emb)
    
    # 3c. Embed state with linear projection (STATE IS IN PREFIX!)
    state_emb = self.state_proj(state)  # Linear(max_state_dim=32, hidden_dim)
    embs.append(state_emb[:, None, :])  # Add sequence dimension
    
    return torch.cat(embs, dim=1)  # (B, prefix_len, hidden_dim)
```

#### Step 3b: Embed Suffix (Actions + Timestep - NO State)
```python
def embed_suffix(self, noisy_actions, timestep):
    """Note: SmolVLA suffix does NOT include state (unlike Pi0)."""
    # Fuse timestep + action information using an MLP
    action_emb = self.action_in_proj(noisy_actions)
    time_emb = create_sinusoidal_pos_embedding(timestep, hidden_dim, ...)
    action_time_emb = MLP([action_emb, time_emb])
    
    return action_time_emb  # (B, chunk_size, hidden_dim)
```

#### Step 4: Flow Matching Forward Pass
```python
def forward(self, batch, noise=None, time=None):
    # Prepare inputs
    images, img_masks = self.prepare_images(batch)
    state = self.prepare_state(batch)
    actions = self.prepare_action(batch)  # (B, chunk_size, max_action_dim)
    
    # Sample noise and time for flow matching
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)  # N(0, 1)
    if time is None:
        time = self.sample_time(batch_size, device)  # Beta distribution
    
    # Flow matching interpolation: x_t = t * noise + (1-t) * actions
    time_expanded = time[:, None, None]  # (B, 1, 1)
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions  # Target velocity field
    
    # Forward through VLM + Expert
    pred_velocity = self.vlm_with_expert(prefix_embs, suffix_embs, ...)
    
    # Compute loss: MSE between predicted and target velocity
    loss = F.mse_loss(pred_velocity, u_t)
    
    return loss
```

### 🔑 Key Concept: Flow Matching

Flow Matching is a generative technique that learns to map noise to data by predicting velocity fields:

- **Interpolation:** `x_t = t × noise + (1-t) × action` (t ∈ [0, 1])
- **Target velocity:** `u_t = noise - action` (straight line from action to noise)
- **Training:** Network predicts `v_θ(x_t, t)`, minimize `||v_θ - u_t||²`
- **Inference:** Integrate from noise using Euler: `x_{t+dt} = x_t + dt × v_θ(x_t, t)`

---

## 5. Pi0: PaliGemma + Expert Model

### Architecture Overview

Pi0 uses PaliGemma (SigLIP + Gemma 2B) as VLM backbone with a Gemma 300M action expert. Unlike SmolVLA, Pi0 embeds state in the **SUFFIX** (with actions), not in the prefix.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Pi0 Architecture                                │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    Images ──► SigLIP Vision Tower (224×224) ──► Multi-modal Projector ──┐   │
│                                                                          │   │
│    Language ──► PaliGemma Tokenizer ──► Gemma Embedding ─────────────────┼──►│
│                                                                          │   │
│                              PREFIX: [images, language]                      │
│                              (embed_prefix - NO state!)                      │
│                                        │                                     │
│                                        ▼                                     │
│                     PaliGemma Language Model (Frozen)                        │
│                                        │                                     │
│    ┌───────────────────────────────────┼───────────────────────────────────┐ │
│    │                                   │ KV Cache                          │ │
│    ▼                                   ▼                                   ▼ │
│  state_proj(state)      action_in_proj(noisy_actions)    time_embedding     │
│    │                         │                           │                   │
│    └─────────────────────────┼───────────────────────────┘                   │
│                              ▼                                               │
│                 SUFFIX: [STATE_emb, noisy_act₁+time, ..., noisy_actₙ+time]   │
│                              (embed_suffix - STATE is here!)                 │
│                              │                                               │
│                              ▼                                               │
│                 Gemma 300M Expert (Trainable) - Joint Attention              │
│                              │                                               │
│                              ▼                                               │
│                 Predicted Action Chunk (chunk_size=50)                       │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Differences from SmolVLA

- **VLM Backbone:** PaliGemma (Gemma 2B + SigLIP) vs SmolVLM2
- **Image Resolution:** 224×224 (default) vs 512×512
- **Expert Architecture:** Gemma 300M decoder vs lightweight expert
- **Joint Attention:** Expert attends to both VLM output and its own tokens
- **Robot State Position:** Pi0 embeds state in SUFFIX; SmolVLA embeds state in PREFIX

> **⚠️ Important: State Handling Across Models**
> - **SmolVLA:** State embedded via `state_proj` in **PREFIX**: `[images, language, state]`
> - **Pi0:** State embedded via `state_proj` in **SUFFIX**: `[state_emb, noisy_actions, timestep]`
> - **Pi0.5:** State is **discretized into 256 bins and tokenized into the language prompt** as text: `"Task: {task}, State: {bin1} {bin2} ...;\nAction: "`. No separate state embedding layer. Uses AdaRMS conditioning for timestep instead.

---

## 6. Pi0.5: Discretized State in Language Prompt + AdaRMS Conditioning

### Architecture Overview

Pi0.5 is similar to Pi0 but with key differences in how state is handled:

- **Discretized State in Language:** Pi0.5 discretizes robot state into 256 bins and includes it as text in the language prompt: `"Task: {task}, State: {bin1} {bin2} ...;\nAction: "`. No separate `state_proj` layer.
- **AdaRMS Conditioning:** Uses Adaptive RMSNorm (AdaRMS) for timestep conditioning in the action expert.
- **Longer Language:** Supports 200 tokens (vs 48 in Pi0) to accommodate the discretized state in the prompt.
- **Quantiles Normalization:** Uses QUANTILES normalization (vs MEAN_STD in Pi0) before discretization.

### Key Differences from Pi0

- **Robot State:** Pi0.5 tokenizes state into the language prompt (discretized to 256 bins). Pi0 uses a separate `state_proj` embedding.
- **AdaRMS Conditioning:** Uses Adaptive RMSNorm (AdaRMS) for timestep conditioning instead of concatenating timestep with action embeddings.
- **Tokenizer Length:** 200 tokens (vs 48 in Pi0) to fit the discretized state string.
- **Expert Size:** Same as Pi0 (gemma_300m default for both).
- **Normalization:** Uses QUANTILES normalization for state/action (vs MEAN_STD in Pi0) - state is normalized to [-1, 1] before discretization.
- **Time Conditioning:** Uses `time_mlp_*` for AdaRMS conditioning (vs `action_time_mlp_*` in Pi0).

### Pi0.5 Forward Pass

```python
def forward(self, images, img_masks, tokens, masks, actions, noise=None, time=None):
    # Note: NO state parameter!
    # Sample noise and time
    if noise is None:
        noise = self.sample_noise(actions.shape, actions.device)
    if time is None:
        time = self.sample_time(batch_size, device)
    
    # Flow matching interpolation
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1 - time_expanded) * actions
    u_t = noise - actions
    
    # Embed prefix (images + language)
    prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
        images, img_masks, tokens, masks
    )
    
    # Embed suffix (NO state - only noisy actions + timestep)
    suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(
        x_t, time  # NO state parameter!
    )
    
    # Combine prefix and suffix
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
    
    # Joint forward through PaliGemma + Expert with AdaRMS
    prefix_output, suffix_output = self.paligemma_with_expert(
        inputs_embeds=[prefix_embs, suffix_embs],
        attention_mask=att_2d_masks,
        adarms_cond=[None, adarms_cond],  # AdaRMS conditioning
        ...
    )
    
    # Project to action space
    pred_velocity = self.action_out_proj(suffix_output[:, -chunk_size:])
    
    # Flow matching loss
    loss = F.mse_loss(pred_velocity, u_t)
    
    return loss
```

### 🔑 Key Concept: Discretized State in Language Prompt

Pi0.5 incorporates robot state into the language prompt as discretized tokens:

```python
# From Pi05PrepareStateTokenizerProcessorStep
# 1. State is normalized to [-1, 1] by NormalizerProcessorStep (QUANTILES mode)
# 2. Discretize into 256 bins
discretized_states = np.digitize(state_np, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
# 3. Convert to string and append to prompt
state_str = " ".join(map(str, discretized_states[i]))
full_prompt = f"Task: {cleaned_text}, State: {state_str};\nAction: "
# Example: "Task: pick up the cube, State: 128 130 127 125 129 128;\nAction: "
```

### 🔑 Key Concept: AdaRMS Conditioning

Pi0.5 uses Adaptive RMSNorm (AdaRMS) for timestep conditioning:

- **Pi0:** Concatenates timestep embedding with action embeddings: `[action_emb, time_emb] → MLP`
- **Pi0.5:** Uses timestep embedding as AdaRMS condition: `adarms_cond = time_emb`, passed separately to expert layers
- **Benefit:** More efficient conditioning, allows model to adapt normalization based on timestep

---

## 7. Pi0.5-Fast: Optimized for Speed

### Overview

Pi0.5-Fast is the same architecture as Pi0.5, but optimized for faster inference:

- **Fewer Inference Steps:** Uses `num_inference_steps=5` (vs 10 in standard Pi0.5)
- **Same Architecture:** Identical to Pi0.5 (no state, AdaRMS conditioning)
- **Trade-off:** Slightly lower action quality for ~2x faster inference
- **Use Case:** Real-time robot control where speed is critical

### Configuration Differences

| Parameter | Pi0.5 | Pi0.5-Fast |
|-----------|-------|------------|
| `num_inference_steps` | 10 | 5 |
| `action_expert_variant` | gemma_300m | gemma_300m |
| `paligemma_variant` | gemma_2b | gemma_2b |
| `chunk_size` | 50 | 50 |
| **Inference Speed** | Baseline | ~2x faster |

### Usage Example

```bash
# Standard Pi0.5
lerobot-train \
    --policy.path=lerobot/pi05_base \
    --policy.num_inference_steps=10 \
    ...

# Pi0.5-Fast (faster inference)
lerobot-train \
    --policy.path=lerobot/pi05_base \
    --policy.num_inference_steps=5 \
    ...
```

> **Note:** Pi0.5-Fast is not a separate model variant, but rather Pi0.5 configured with fewer inference steps. The architecture is identical to Pi0.5. You can achieve the same result by setting `--policy.num_inference_steps=5` when using Pi0.5.

---

## 8. ACT: Action Chunking Transformer

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              ACT Architecture                                │
│                    (Action Chunking Transformer)                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                            Training Mode (with VAE)                          │
│                                                                              │
│    Actions + Robot State ──► VAE Encoder ──► μ, log(σ²) ──► Reparameterize  │
│                                                                    │         │
│                                                              Latent z        │
│                                                                    │         │
│    ┌───────────────────────────────────────────────────────────────┘         │
│    │                                                                         │
│    ▼                                                                         │
│    TRANSFORMER ENCODER                                                       │
│    Input: [latent_proj(z), state_proj(state), cam_features...]               │
│                                                                              │
│    Camera Features:                                                          │
│    Images ──► ResNet18 Backbone ──► Spatial Features ──► Projection          │
│                              │                                               │
│                              ▼                                               │
│                     Encoder attention layers                                 │
│                              │                                               │
│                              │ Encoder output                                │
│                              ▼                                               │
│    TRANSFORMER DECODER                                                       │
│    Query: [chunk_size] learnable position embeddings                         │
│    Cross-attention to encoder output                                         │
│                              │                                               │
│                              ▼                                               │
│    Action Head (Linear) ──► (B, chunk_size, action_dim)                      │
│                                                                              │
│   Loss = L1(predicted, target) + β × KL(q(z|x) || p(z))                      │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### ACT Key Features

- **Vision Backbone:** ResNet18 (pretrained on ImageNet) for image encoding
- **VAE Objective:** Optional variational objective for diverse action generation
- **Action Chunking:** Predicts entire action sequence (100 steps) at once
- **Temporal Ensembling:** Optional exponential weighted averaging of predictions

### ACT Loss Computation

```python
# L1 reconstruction loss
l1_loss = F.l1_loss(batch["action"], actions_hat, reduction="none")
l1_loss = (l1_loss * ~batch["action_is_pad"].unsqueeze(-1)).mean()

# KL divergence (if VAE)
if self.config.use_vae:
    mean_kld = (-0.5 * (1 + log_sigma_x2 - mu.pow(2) - log_sigma_x2.exp())).sum(-1).mean()
    loss = l1_loss + self.config.kl_weight * mean_kld
else:
    loss = l1_loss
```

---

## 9. Groot: NVIDIA GR00T Integration

### Architecture Overview

Groot uses NVIDIA's Isaac GR00T N1.5 with Eagle Vision-Language encoding:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              Groot (GR00T N1.5)                              │
│                         NVIDIA Isaac Generalist Robot                        │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    Video Frames + Language ──► Eagle VL Processor ──► Eagle VL Model         │
│                                        │                                     │
│                              Vision-Language Embeddings                      │
│                                        │                                     │
│    ┌───────────────────────────────────┼───────────────────────────────────┐ │
│    │                                   │                                   │ │
│    ▼                                   ▼                                   ▼ │
│  State Encoder          Action Encoder               Embodiment ID           │
│    │                         │                           │                   │
│    └─────────────────────────┼───────────────────────────┘                   │
│                              ▼                                               │
│                 Flow Matching Action Head (DiT)                              │
│                 Cross-attention to VL embeddings                             │
│                              │                                               │
│                              ▼                                               │
│                 Action Decoder (embodiment-specific)                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Groot Preprocessing Pipeline

```python
input_steps = [
    RenameObservationsProcessorStep(rename_map={}),
    AddBatchDimensionProcessorStep(),
    
    # Pack inputs: state, action, language, embodiment
    GrootPackInputsStep(
        state_horizon=1,
        action_horizon=16,
        max_state_dim=64,
        max_action_dim=32,
        normalize_min_max=True,  # Min-max normalize
    ),
    
    # Eagle encode: convert images to Eagle format
    GrootEagleEncodeStep(tokenizer_assets_repo="nvidia/GR00T-N1.5-3B"),
    
    # Collate Eagle content → tensors
    GrootEagleCollateStep(tokenizer_assets_repo="nvidia/GR00T-N1.5-3B"),
    
    DeviceProcessorStep(device="cuda"),
]
```

---

## 10. Diffusion Policy: Denoising Actions

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           Diffusion Policy                                   │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    Images ──► ResNet Encoder ──┐                                             │
│                                ├──► Global Conditioning Vector               │
│    State ──────────────────────┘    (B, global_cond_dim)                     │
│                                        │                                     │
│    ┌───────────────────────────────────┼───────────────────────────────────┐ │
│    │                                   │                                   │ │
│    ▼                                   ▼                                   ▼ │
│  Noise Sample ε        Random Timestep t           Global Condition          │
│    │                         │                           │                   │
│    └─────────────────────────┼───────────────────────────┘                   │
│                              ▼                                               │
│    TRAINING: noisy_trajectory = scheduler.add_noise(actions, ε, t)           │
│                              │                                               │
│                              ▼                                               │
│                     Conditional UNet 1D                                      │
│                     - Down blocks (Conv1d + Attention)                       │
│                     - Mid block (Self-attention)                             │
│                     - Up blocks (Conv1d + Skip connections)                  │
│                              │                                               │
│                              ▼                                               │
│    Loss = MSE(predicted_noise, ε)  or  MSE(predicted_sample, actions)        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘

INFERENCE: Iterative Denoising
┌──────────────────────────────────────────────────────────────────────────────┐
│   sample = randn(B, horizon, action_dim)  # Start from pure noise            │
│                                                                              │
│   for t in scheduler.timesteps:  # T → 0                                     │
│       model_output = unet(sample, t, global_cond)                            │
│       sample = scheduler.step(model_output, t, sample)  # Denoise step       │
│                                                                              │
│   return sample  # Clean action trajectory                                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Diffusion Policy Forward Pass (Training)

```python
def compute_loss(self, batch):
    # 1. Prepare global conditioning
    global_cond = self._prepare_global_conditioning(batch)
    
    # 2. Get target action trajectory
    trajectory = batch["action"]  # (B, horizon, action_dim)
    
    # 3. Sample random noise
    eps = torch.randn(trajectory.shape, device=trajectory.device)
    
    # 4. Sample random timesteps
    timesteps = torch.randint(0, num_train_timesteps, (batch_size,)).long()
    
    # 5. Add noise to trajectory
    noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
    
    # 6. Predict noise with UNet
    pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)
    
    # 7. Compute loss
    loss = F.mse_loss(pred, eps)  # or pred vs trajectory
    
    return loss
```

---

## 11. Model Comparison Summary

| Feature | SmolVLA | Pi0 | Pi0.5 | Pi0.5-Fast | ACT | Groot | Diffusion |
|---------|---------|-----|-------|------------|-----|-------|-----------|
| **Vision Encoder** | SigLIP (SmolVLM) | SigLIP (PaliGemma) | SigLIP (PaliGemma) | SigLIP (PaliGemma) | ResNet18 | Eagle VL | ResNet |
| **Language Model** | SmolVLM2 LLM | Gemma 2B | Gemma 2B | Gemma 2B | None | Eagle LLM | None |
| **Action Generation** | Flow Matching | Flow Matching | Flow Matching | Flow Matching | Encoder-Decoder | Flow Matching (DiT) | DDPM/DDIM |
| **Image Resolution** | 512×512 | 224×224 | 224×224 | 224×224 | 96×96 | Variable | 96×96 |
| **Default chunk_size** | 50 | 50 | 50 | 50 | 100 | 16 | Variable |
| **Loss Function** | MSE (velocity) | MSE (velocity) | MSE (velocity) | MSE (velocity) | L1 + KL | MSE (velocity) | MSE (noise) |
| **Normalization** | Mean-Std | Mean-Std | Quantiles | Quantiles | Mean-Std | Min-Max | Mean-Std |
| **Language Conditioning** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| **Robot State Input** | ✅ PREFIX (state_proj) | ✅ SUFFIX (state_proj) | ⚠️ Tokenized in prompt | ⚠️ Tokenized in prompt | ✅ Yes | ✅ Yes | ✅ Yes |
| **Expert Size** | Lightweight | Gemma 300M | Gemma 300M | Gemma 300M | N/A | DiT | UNet |
| **Tokenizer Length** | 48 tokens | 48 tokens | 200 tokens | 200 tokens | N/A | Variable | N/A |
| **Finetuning Strategy** | Expert only | Expert only | Expert only | Expert only | Full model | Configurable | Full model |
| **Inference Steps** | 10 | 10 | 10 | 5 | 1 | 10 | 100 |

### 🎯 Choosing the Right Model

- **SmolVLA:** Best for language-conditioned tasks with limited compute. Lightweight and fast. State in PREFIX.
- **Pi0:** High-quality VLA with strong generalization. Explicit state embedding in SUFFIX. Good for complex manipulation.
- **Pi0.5:** Open-world generalization with state as tokenized text. 200-token prompts. Uses AdaRMS conditioning.
- **Pi0.5-Fast:** Pi0.5 with 5 inference steps instead of 10. ~2x faster inference, slightly lower quality.
- **ACT:** Best for bimanual manipulation (Aloha). Fast single-step inference, simple encoder-decoder architecture.
- **Groot:** Multi-embodiment support. NVIDIA GR00T integration. Good for transferring policies across robots.
- **Diffusion:** Versatile and proven. Good for multi-modal action distributions. 100 inference steps.

---

*Generated for LeRobot Training Pipeline Understanding*
*https://github.com/huggingface/lerobot*


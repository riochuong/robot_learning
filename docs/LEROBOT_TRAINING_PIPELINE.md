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
11. [X-VLA: Cross-Embodiment VLA](#11-x-vla-cross-embodiment-vla)
12. [Model Comparison Summary](#12-model-comparison-summary)

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

## 11. X-VLA: Cross-Embodiment VLA

### Architecture Overview

X-VLA (Cross-Embodiment Vision-Language-Action) uses Florence-2 as VLM backbone with a soft-prompted transformer for cross-robot generalization:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              X-VLA Architecture                              │
│                      (Cross-Embodiment VLA)                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    Multi-View Images ──► Florence-2 Vision Tower ──► Image Features ──┐     │
│                                                                        │     │
│    Language ──► Florence-2 BART Tokenizer ──► Token Embeddings ───────┼──►  │
│                                                                        │     │
│                         FLORENCE-2 ENCODER                                   │
│               (Vision + Language merged via multimodal projector)            │
│                              │                                               │
│                              ▼                                               │
│                    VLM Features (encoder output)                             │
│                              │                                               │
│    ┌───────────────────────────────┴───────────────────────────────┐        │
│    │                                                               │        │
│    ▼                                                               ▼        │
│  Domain ID (embodiment identifier)                    Aux Visual Inputs     │
│    │                                                               │        │
│    │          Proprioception State ──► Linear Projection          │        │
│    │                                   │                           │        │
│    └──────────────────────┬────────────┘                           │        │
│                           ▼                                        │        │
│               SOFT PROMPT HUB (Per-Domain Embediments)             │        │
│             - 30 domains × 32 learnable prompt tokens              │        │
│             - Rapid adaptation to new robots                       │        │
│                           │                                        │        │
│                           ▼                                        │        │
│               SOFT-PROMPTED TRANSFORMER                            │        │
│             - 24 layers × 16 heads × 1024 hidden                   │        │
│             - Flow matching for action generation                  │        │
│             - Heterogeneous input projections                      │        │
│                           │                                        │        │
│                           ▼                                        │        │
│            Action Denoising (Flow Matching)                        │        │
│            x_t = t * noise + (1-t) * action                        │        │
│            Predict velocity field: noise - action                  │        │
│                           │                                        │        │
│                           ▼                                        │        │
│          ACTION HEAD (with Action Space Registry)                  │        │
│          - ee6d: End-effector 6D + gripper (20D)                   │        │
│          - so101_bimanual: SO101 bimanual (12D real → 20D model)   │        │
│          - auto: Auto-detect from dataset                          │        │
│                           │                                        │        │
│                           ▼                                        │        │
│              Predicted Action Chunk (chunk_size=32)                │        │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Key Differences from Other VLAs

- **VLM Backbone:** Florence-2 (DaViT vision + BART language) vs PaliGemma/SmolVLM
- **Cross-Embodiment:** Soft prompts per domain (30 robot types) for rapid adaptation
- **Action Registry:** Pluggable action spaces (ee6d, so101_bimanual, auto)
- **Multi-View:** Supports multiple camera views with auxiliary visual inputs
- **Freezing Strategy:** Freeze VLM encoders, train only soft prompts + transformer
- **Training:** Two-phase (pretraining on 290K episodes, then domain adaptation)

> **⚠️ Key Feature: Action Space Registry**
> - X-VLA uses an **Action Registry** system to handle different robots
> - Each action mode defines its own loss (MSE for joints, BCE for grippers)
> - `so101_bimanual` mode: pads 12D real actions → 20D model actions
> - `auto` mode: auto-detects action dim from dataset
> - Enables single pretrained model to work across diverse robots

### Step-by-Step Forward Pass

#### Step 1: Encode Vision-Language via Florence-2

```python
def forward_vlm(self, input_ids, pixel_values, image_mask):
    """Encode text and multi-view images via Florence-2 encoder."""
    batch_size, num_views = pixel_values.shape[:2]
    
    # 1a. Flatten and filter valid images
    flat_images = pixel_values.flatten(0, 1)  # (B*V, C, H, W)
    flat_mask = image_mask.view(-1).to(dtype=torch.bool)
    valid_images = flat_images[flat_mask]
    
    # 1b. Encode images with Florence-2 vision tower
    valid_feats = self.vlm._encode_image(valid_images)  # (N_valid, tokens, hidden)
    
    # 1c. Reconstruct to (B, V, tokens, hidden)
    image_features = valid_feats.new_zeros((batch_size * num_views, tokens_per_view, hidden_dim))
    image_features[flat_mask] = valid_feats
    image_features = image_features.view(batch_size, num_views, tokens_per_view, hidden_dim)
    
    # 1d. Embed language tokens
    inputs_embeds = self.vlm.get_input_embeddings()(input_ids)
    
    # 1e. Merge primary view with text via multimodal projector
    merged_embeds, attention_mask = self.vlm._merge_input_ids_with_image_features(
        image_features[:, 0],  # Primary view
        inputs_embeds,
    )
    
    # 1f. Pass through Florence-2 BART encoder
    enc_out = self.vlm.language_model.model.encoder(
        attention_mask=attention_mask,
        inputs_embeds=merged_embeds,
    )[0]
    
    # 1g. Prepare auxiliary views (views 1+)
    aux_visual_inputs = image_features[:, 1:].reshape(batch_size, -1, hidden_dim)
    
    return {"vlm_features": enc_out, "aux_visual_inputs": aux_visual_inputs}
```

#### Step 2: Flow Matching Training

```python
def forward(self, input_ids, image_input, image_mask, domain_id, proprio, action):
    """Forward pass for X-VLA model."""
    # 2a. Encode vision-language
    enc = self.forward_vlm(input_ids, image_input, image_mask)
    
    # 2b. Sample random timestep for flow matching
    batch_size = input_ids.shape[0]
    t = (torch.rand(1, device=device, dtype=dtype)
         + torch.arange(batch_size, device=device, dtype=dtype) / batch_size) % (1 - 1e-5)
    
    # 2c. Flow matching interpolation: x_t = t * noise + (1-t) * action
    action_noisy = torch.randn_like(action) * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
    
    # 2d. Preprocess via action space (e.g., zero out grippers, pad dimensions)
    proprio_m, action_noisy_m = self.action_space.preprocess(proprio, action_noisy)
    
    # 2e. Pass through soft-prompted transformer
    pred_action = self.transformer(
        domain_id=domain_id,          # Which robot embodiment
        action_with_noise=action_noisy_m,
        t=t,                          # Flow matching timestep
        proprio=proprio_m,
        **enc,                        # VLM features + aux visuals
    )
    
    # 2f. Compute action space-specific loss
    return self.action_space.compute_loss(pred_action, action)
```

#### Step 3: Inference with Flow Matching

```python
@torch.no_grad()
def generate_actions(self, input_ids, image_input, image_mask, domain_id, proprio, steps):
    """Generate actions via flow matching denoising."""
    self.eval()
    
    # 3a. Encode vision-language (once)
    enc = self.forward_vlm(input_ids, image_input, image_mask)
    
    # 3b. Start from random noise
    batch_size = input_ids.shape[0]
    x1 = torch.randn(batch_size, self.chunk_size, action_dim, device=device)
    action = torch.zeros_like(x1)
    
    # 3c. Iterative denoising (default: 10 steps)
    steps = max(1, int(steps))
    for i in range(steps, 0, -1):
        t = torch.full((batch_size,), i / steps, device=device)
        
        # Interpolate: x_t = t * x1 + (1-t) * action
        x_t = x1 * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
        
        # Preprocess and predict
        proprio_m, x_t_m = self.action_space.preprocess(proprio, x_t)
        action = self.transformer(
            domain_id=domain_id,
            action_with_noise=x_t_m,
            proprio=proprio_m,
            t=t,
            **enc,
        )
    
    # 3d. Postprocess (e.g., apply sigmoid to grippers, trim padding)
    return self.action_space.postprocess(action)
```

### 🔑 Key Concept: Soft Prompt Hub (Cross-Embodiment)

X-VLA achieves cross-embodiment generalization via **soft prompt hub**:

```python
# Soft Prompt Hub: 30 domains × 32 learnable tokens per domain
self.soft_prompt_hub = nn.Parameter(torch.zeros(num_domains, len_soft_prompts, hidden_size))

# At runtime, select prompts for specific embodiment
domain_prompts = self.soft_prompt_hub[domain_id]  # (B, 32, 1024)

# These prompts are prepended to transformer input
# They encode embodiment-specific information (kinematics, workspace, etc.)
```

**Benefits:**
- **Rapid Adaptation:** Fine-tune only 32 prompts (32K params) for new robot
- **Frozen Backbone:** VLM encoders stay frozen, preserving pretrained knowledge
- **Data Efficient:** 3K steps sufficient to adapt to new embodiment
- **Scalability:** Support 30+ robot types with single model

### 🔑 Key Concept: Action Space Registry

X-VLA uses an **Action Registry** to handle different action spaces:

```python
# Example: SO101 Bimanual (12D real → 20D model)
@register_action("so101_bimanual")
class BimanualSO101ActionSpace(BaseActionSpace):
    dim_action = 20          # Model output dimension
    REAL_DIM = 12           # Real robot dimension
    gripper_idx = (5, 11)   # Gripper indices
    
    def preprocess(self, proprio, action):
        """Pad 12D real actions to 20D for model."""
        action = pad_to_20d(action)
        # Zero out gripper channels during training
        action[:, :, self.gripper_idx] = 0
        return proprio, action
    
    def compute_loss(self, pred, target):
        """MSE for joints, BCE for grippers."""
        joint_loss = F.mse_loss(pred[:, :, joint_idxs], target[:, :, joint_idxs])
        gripper_loss = F.binary_cross_entropy_with_logits(
            pred[:, :, gripper_idx], target[:, :, gripper_idx]
        )
        return {"joint_loss": joint_loss, "gripper_loss": gripper_loss}
    
    def postprocess(self, action):
        """Trim 20D model output to 12D real, apply sigmoid to grippers."""
        action = action[:, :self.REAL_DIM]  # Trim padding
        action[:, self.gripper_idx] = torch.sigmoid(action[:, self.gripper_idx])
        return action
```

### X-VLA Preprocessing Pipeline

```python
input_steps = [
    RenameObservationsProcessorStep(rename_map={}),
    AddBatchDimensionProcessorStep(),
    
    # Tokenize language task description
    TokenizerProcessorStep(
        tokenizer_name="facebook/bart-large",
        max_length=64,
        padding="max_length",
    ),
    
    # Convert images from [0, 255] to [0, 1]
    XVLAImageToFloatProcessorStep(),
    
    # Normalize with ImageNet stats (for Florence-2)
    XVLAImageNetNormalizeProcessorStep(),
    
    # Add domain_id for embodiment selection
    XVLAAddDomainIdProcessorStep(domain_id=0),
    
    DeviceProcessorStep(device="cuda"),
    
    # X-VLA uses IDENTITY normalization (no mean-std)
    NormalizerProcessorStep(
        features=features,
        norm_map={"VISUAL": "IDENTITY", "STATE": "IDENTITY", "ACTION": "IDENTITY"},
        stats=dataset_stats,
    ),
]
```

### Training Configuration

```python
# Two-phase training strategy
# Phase I: Pretraining (290K episodes, multi-embodiment)
lerobot-train \
    --policy.type=xvla \
    --policy.path=lerobot/xvla-base \
    --dataset.repo_id=multi_embodiment_dataset \
    --policy.freeze_vision_encoder=true \
    --policy.freeze_language_encoder=true \
    --policy.train_soft_prompts=true \
    --steps=100000

# Phase II: Domain Adaptation (3K steps, target robot)
lerobot-train \
    --policy.type=xvla \
    --policy.path=lerobot/xvla-base \
    --dataset.repo_id=bimanual_so101_dataset \
    --policy.action_mode=so101_bimanual \
    --policy.freeze_vision_encoder=false \  # Unfreeze for best performance
    --policy.freeze_language_encoder=false \
    --policy.train_soft_prompts=true \
    --steps=3000
```

### Differential Learning Rates

X-VLA uses **XVLAAdamW** optimizer with differential LRs:

```python
# Optimizer applies different learning rates based on parameter names
optimizer = XVLAAdamW(
    model.get_optim_params(),
    lr=1e-4,                                    # Base LR
    betas=(0.9, 0.99),
    weight_decay=0.0,
)

# Learning rate scheme:
# - VLM parameters (vision/language encoders): lr / 10 = 1e-5
# - Soft prompts: lr * soft_prompt_lr_scale = 1e-4 (default)
# - Transformer/action head: lr = 1e-4

# This ensures stable optimization of pretrained VLM while allowing
# rapid adaptation of task-specific components
```

---

## 12. Model Comparison Summary

| Feature | SmolVLA | Pi0 | Pi0.5 | Pi0.5-Fast | ACT | Groot | Diffusion | X-VLA |
|---------|---------|-----|-------|------------|-----|-------|-----------|-------|
| **Vision Encoder** | SigLIP (SmolVLM) | SigLIP (PaliGemma) | SigLIP (PaliGemma) | SigLIP (PaliGemma) | ResNet18 | Eagle VL | ResNet | DaViT (Florence-2) |
| **Language Model** | SmolVLM2 LLM | Gemma 2B | Gemma 2B | Gemma 2B | None | Eagle LLM | None | BART (Florence-2) |
| **Action Generation** | Flow Matching | Flow Matching | Flow Matching | Flow Matching | Encoder-Decoder | Flow Matching (DiT) | DDPM/DDIM | Flow Matching |
| **Image Resolution** | 512×512 | 224×224 | 224×224 | 224×224 | 96×96 | Variable | 96×96 | Variable (resizable) |
| **Default chunk_size** | 50 | 50 | 50 | 50 | 100 | 16 | Variable | 32 |
| **Loss Function** | MSE (velocity) | MSE (velocity) | MSE (velocity) | MSE (velocity) | L1 + KL | MSE (velocity) | MSE (noise) | Action-space specific |
| **Normalization** | Mean-Std | Mean-Std | Quantiles | Quantiles | Mean-Std | Min-Max | Mean-Std | Identity (ImageNet for vision) |
| **Language Conditioning** | ✅ | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ | ✅ |
| **Robot State Input** | ✅ PREFIX (state_proj) | ✅ SUFFIX (state_proj) | ⚠️ Tokenized in prompt | ⚠️ Tokenized in prompt | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Linear projection |
| **Expert Size** | Lightweight | Gemma 300M | Gemma 300M | Gemma 300M | N/A | DiT | UNet | Transformer (24L, 16H, 1024D) |
| **Tokenizer Length** | 48 tokens | 48 tokens | 200 tokens | 200 tokens | N/A | Variable | N/A | 64 tokens (BART) |
| **Finetuning Strategy** | Expert only | Expert only | Expert only | Expert only | Full model | Configurable | Full model | Soft prompts + Transformer |
| **Inference Steps** | 10 | 10 | 10 | 5 | 1 | 10 | 100 | 10 |
| **Cross-Embodiment** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ (Soft prompts) |
| **Action Space Registry** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ (Pluggable) |
| **Multi-View Support** | ❌ | ❌ | ❌ | ❌ | ✅ | ✅ | ❌ | ✅ |

### 🎯 Choosing the Right Model

- **SmolVLA:** Best for language-conditioned tasks with limited compute. Lightweight and fast. State in PREFIX.
- **Pi0:** High-quality VLA with strong generalization. Explicit state embedding in SUFFIX. Good for complex manipulation.
- **Pi0.5:** Open-world generalization with state as tokenized text. 200-token prompts. Uses AdaRMS conditioning. **97.5% LIBERO benchmark.**
- **Pi0.5-Fast:** Pi0.5 with 5 inference steps instead of 10. ~2x faster inference, slightly lower quality.
- **ACT:** Best for bimanual manipulation (Aloha). Fast single-step inference, simple encoder-decoder architecture.
- **Groot:** Multi-embodiment support. NVIDIA GR00T integration. Good for transferring policies across robots. **87% LIBERO benchmark.**
- **Diffusion:** Versatile and proven. Good for multi-modal action distributions. 100 inference steps.
- **X-VLA:** **Cross-embodiment champion.** Florence-2 backbone + soft prompts for rapid adaptation. Pluggable action spaces (ee6d, so101_bimanual, auto). **93% LIBERO, 100% cloth folding.** Best for transferring across diverse robots or bimanual SO101.

---

*Generated for LeRobot Training Pipeline Understanding*
*https://github.com/huggingface/lerobot*


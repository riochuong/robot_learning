# Models with Strong Generalization Capability

Based on recent research (2024-2025) and documented performance results, here are the robotics models that have demonstrated exceptional generalization capabilities:

---

## 🏆 Top Tier: Strongest Generalization (LeRobot Available)

### 1. **Pi0.5** (π₀.₅) ⭐⭐⭐⭐⭐
- **Policy Type**: `pi05`
- **Developer**: Physical Intelligence (2024)
- **Generalization Score**: **97.5% average** on LIBERO benchmark
- **Key Strengths**:
  - Open-world generalization with state tokenized as text
  - 200-token prompts for rich task specification
  - AdaRMS conditioning for better control
  - Cross-task transfer without task-specific training
  
**Benchmark Results (LIBERO)**:
| Task             | Success Rate |
|------------------|--------------|
| Libero Spatial   | 97.0%        |
| Libero Object    | 99.0%        |
| Libero Goal      | 98.0%        |
| Libero 10        | 96.0%        |
| **Average**      | **97.5%**    |

**Why it generalizes well**:
- Pretrained on massive PaliGemma 300M VLM
- Explicit state embedding in SUFFIX position
- Strong language grounding for task understanding
- Handles novel objects and spatial configurations

---

### 2. **X-VLA** (Florence-2 based) ⭐⭐⭐⭐⭐
- **Policy Type**: `xvla`
- **Developer**: Microsoft/Hugging Face (2024)
- **Generalization Score**: **93%** on LIBERO, **100%** on real-world cloth folding
- **Key Strengths**:
  - **Cross-embodiment generalization**: Trained on 290K episodes across 7 platforms
  - **Embodiment-agnostic via soft prompts** (30 domains × 32 learnable tokens)
  - **Strong few-shot adaptation** (3K steps for new robots)
  - **Real-world continuous operation** (2 hours cloth folding)
  - **Pluggable action spaces** (ee6d, so101_bimanual, auto)

**Benchmark Results**:
| Task                    | Success Rate |
|-------------------------|--------------|
| LIBERO (simulation)     | 93%          |
| Cloth Folding (real)    | 100%         |
| WidowX pick-place       | High         |
| Cross-embodiment adapt  | Strong       |
| SO101 bimanual support  | Native       |

**Why it generalizes well**:
- **Two-phase training**: Phase I pretraining (290K episodes, multi-embodiment) + Phase II domain adaptation (3K steps, target robot)
- **Soft prompts** absorb embodiment variations (32K params per robot)
- **Florence-2 backbone**: DaViT vision + BART language with strong vision-language understanding
- **Trained on multi-embodiment data**: Droid, Robomind, Agibot (7 platforms, 5 robot types)
- **Freezes VLM backbone**: Adapts only soft prompts + transformer for new robots
- **Action space registry**: Each robot gets custom preprocessing/postprocessing logic
- **Differential learning rates**: VLM at lr/10, soft prompts at full lr for stable optimization

---

### 3. **OpenVLA** ⭐⭐⭐⭐⭐
- **Status**: Not directly in LeRobot (can be integrated)
- **Developer**: Stanford (2024)
- **Parameters**: 7B
- **Generalization Score**: **Outperforms RT-2** on manipulation tasks
- **Key Strengths**:
  - Trained on Open X-Embodiment dataset (massive scale)
  - Open-source, supports efficient fine-tuning
  - Cross-task and cross-embodiment transfer
  - Smaller than RT-2 (7B vs larger) but better performance

**Why it generalizes well**:
- Open X-Embodiment: diverse robots, tasks, environments
- Vision-language pretraining for semantic understanding
- Efficient LoRA fine-tuning for new tasks
- Web-scale knowledge transfer to robotics

---

### 4. **Groot (GR00T)** ⭐⭐⭐⭐
- **Policy Type**: `groot`
- **Developer**: NVIDIA (2024)
- **Generalization Score**: **87% average** on LIBERO
- **Key Strengths**:
  - Multi-embodiment support (humanoid, arms, mobile manipulators)
  - Foundation model approach
  - Policy transfer across different robot morphologies

**Benchmark Results (LIBERO)**:
| Task             | Success Rate |
|------------------|--------------|
| Libero Spatial   | 82.0%        |
| Libero Object    | 99.0%        |
| Libero Long      | 82.0%        |
| **Average**      | **87.0%**    |

**Why it generalizes well**:
- Designed for multi-embodiment from ground up
- Leverages NVIDIA's Isaac Sim ecosystem
- Large-scale pretraining on diverse robot data
- DiT-based action head with flow matching

---

### 5. **SmolVLA** ⭐⭐⭐⭐
- **Policy Type**: `smolvla`
- **Developer**: Hugging Face (2024)
- **Key Strengths**:
  - Lightweight (small model size)
  - Language-conditioned for task variations
  - Good zero-shot transfer on similar tasks
  - Real-world SO101 demonstrated

**Performance**:
- Pick-place with counting variations
- Generalizes under perturbations
- Real-world SO101 pick-and-place
- Resource-efficient deployment

**Why it generalizes well**:
- Vision-language grounding
- State in PREFIX position for quick adaptation
- Lightweight enables fast fine-tuning
- Balanced compute vs performance

---

## 🥈 Strong Generalization (Proven)

### 6. **Octo (Diffusion-based)** ⭐⭐⭐⭐
- **Status**: Related to LeRobot Diffusion policy
- **Developer**: UC Berkeley (2024)
- **Key Strengths**:
  - Open-source generalist policy
  - Diffusion for smooth trajectories
  - Rapid task adaptation
  - Trained on Open X-Embodiment

**Why it generalizes well**:
- Diffusion outputs handle multi-modal distributions
- Open X-Embodiment diversity
- Continuous trajectory modeling
- Fast fine-tuning (few demonstrations)

---

### 7. **RT-2 (Robotic Transformer 2)** ⭐⭐⭐⭐
- **Status**: Not in LeRobot (proprietary)
- **Developer**: Google DeepMind (2023)
- **Key Strengths**:
  - Web knowledge transfer to robotics
  - Chain-of-thought reasoning
  - Enhanced zero-shot generalization
  - Multi-step task decomposition

**Why it generalizes well**:
- Built on PaLM-E and PaLI-X (massive VLMs)
- Internet-scale vision-language pretraining
- Semantic understanding from web data
- Can reason about novel tasks

---

## 🥉 Good Generalization (Task-Specific)

### 8. **ACT (Action Chunking Transformer)** ⭐⭐⭐
- **Policy Type**: `act`
- **Generalization**: Task-specific, less cross-task
- **Key Strengths**:
  - **Best for bimanual manipulation**
  - Data-efficient (50 demos often sufficient)
  - Fast training (~few hours)
  - Lightweight (80M params)

**Performance**:
- Originally designed for ALOHA bimanual tasks
- High success rates within task domain
- Less generalization to novel objects/tasks
- Excellent for contact-rich manipulation

**When to use**:
- You have task-specific demonstrations
- Need fast training and inference
- Contact-rich, fine-grained manipulation
- Don't need language conditioning

---

### 9. **Diffusion Policy** ⭐⭐⭐
- **Policy Type**: `diffusion`
- **Generalization**: Moderate, multi-modal actions
- **Key Strengths**:
  - Handles multi-modal action distributions
  - Smooth trajectory generation
  - Good for complex, ambiguous tasks
  - Versatile architecture

**Why it generalizes (moderately)**:
- Denoising process handles uncertainty
- Multi-modal outputs for ambiguous situations
- UNet architecture captures spatial features
- 100 diffusion steps for refinement

---

## 📊 Generalization Ranking Summary

| Rank | Model      | Cross-Task | Cross-Embodiment | Language | LIBERO Avg | Use Case                    |
|------|------------|------------|------------------|----------|------------|-----------------------------|
| 🥇   | **Pi0.5**  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐         | ✅       | 97.5%      | Best overall generalization |
| 🥇   | **X-VLA**  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐       | ✅       | 93%        | Cross-embodiment champion   |
| 🥇   | **OpenVLA**| ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐       | ✅       | N/A        | Open-source VLA leader      |
| 🥈   | **Groot**  | ⭐⭐⭐⭐   | ⭐⭐⭐⭐⭐       | ✅       | 87%        | Multi-embodiment            |
| 🥈   | **SmolVLA**| ⭐⭐⭐⭐   | ⭐⭐⭐           | ✅       | N/A        | Lightweight VLA             |
| 🥈   | **Octo**   | ⭐⭐⭐⭐   | ⭐⭐⭐⭐         | ❌       | N/A        | Diffusion generalist        |
| 🥈   | **RT-2**   | ⭐⭐⭐⭐   | ⭐⭐⭐           | ✅       | N/A        | Web knowledge transfer      |
| 🥉   | **ACT**    | ⭐⭐       | ⭐               | ❌       | N/A        | Task-specific, bimanual     |
| 🥉   | **Diffusion**| ⭐⭐⭐   | ⭐⭐             | ❌       | N/A        | Multi-modal actions         |

---

## 🎯 Recommendations by Use Case

### For SO101 Arm - Maximum Generalization:

1. **Pi0.5** (`pi05`) - Best documented generalization (97.5% LIBERO avg)
2. **X-VLA** (`xvla`) - Best for bimanual SO101 with `so101_bimanual` action mode
3. **SmolVLA** (`smolvla`) - Good balance, proven on SO101 real-world

### For Quick Start (Less Generalization Focus):

1. **ACT** (`act`) - Fast training, data-efficient
2. **Diffusion** (`diffusion`) - Versatile, proven

### For Research/Maximum Generalization:

1. **Pi0.5** or **X-VLA** - State-of-the-art results
2. **Groot** - If exploring multi-embodiment
3. **OpenVLA** (custom integration) - Open-source foundation

---

## 🔬 Key Factors for Generalization

Based on the research, models with strong generalization share:

1. **Vision-Language Pretraining**: All top models use VLM backbones
2. **Large-Scale Diverse Data**: Open X-Embodiment, multi-robot datasets
3. **Language Conditioning**: Task specification via natural language
4. **Soft Prompts/Adapters**: Efficient embodiment-specific adaptation
5. **Multi-Modal Reasoning**: Handle visual, language, and proprioceptive inputs

---

## 📚 References

- Pi0.5: Physical Intelligence OpenPI (2024)
- X-VLA: Microsoft Florence-2 VLA (2024)
- OpenVLA: Stanford Open X-Embodiment (2024)
- Groot: NVIDIA GR00T (2024)
- RT-2: Google DeepMind (2023)
- Octo: UC Berkeley (2024)
- ACT: Zhao et al., ALOHA (2023)
- Diffusion: Chi et al., Columbia (2024)

**Last Updated**: December 2025


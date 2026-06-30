# Available Training Configuration Parameters

This document lists the key training config parameters for Cosmos Reason
fine-tuning that are relevant to SOP monitoring RCA.

Source: `cosmos-rl/cosmos_rl/policy/config/__init__.py`

## Training Parameters (train section)

| Parameter | Type | Default | Description | RCA Relevance |
|-----------|------|---------|-------------|---------------|
| `epoch` | int | 1 | Number of training epochs | Too high → overfitting, too low → underfitting. Empirical SOP fine-tuning range: **5–20** |
| `train_batch_per_replica` | int | 8 | Local batch size per replica per gradient-accumulation step | Affects effective batch size and `steps_per_epoch` |
| `max_num_steps` | int \| None | None | Optional upper bound on total training steps. Training stops when either this OR the epoch-based limit is reached, whichever is first. | If set, the loss curve may end mid-epoch — useful for smoke tests but uncommon in production runs |
| `resume` | bool \| str | False | Resume training from a checkpoint. `True` → latest checkpoint in `output_dir`; string → specific checkpoint path. |  |
| `optm_name` | str | "AdamW" | Optimizer name. Choices: `AdamW`, `Adam`. | |
| `optm_lr` | float \| list[float] | 1e-6 | Per-component learning rates as `[llm_lr, vlm_lr, linear_lr]`. Each value controls adaptation rate for that model component. When a single value is given, it applies to all components. | See "Learning Rate Analysis" below |
| `optm_warmup_steps` | int \| float | 20 | LR warmup steps. If a float in `[0.0, 1.0]`, multiplied by total steps to get the step count. | Short warmup + aggressive LR causes early-training gradient shock; assess in conjunction with `optm_lr` when diagnosing Pattern 9 (model collapse). The `optm_warmup_steps / total_steps` ratio is a primary diagnostic. |
| `optm_decay_ratio` | float \| None | None | Fraction of total steps over which LR decays, range `[0.0, 1.0]`. 0 means no decay. | If None or 0, constant LR after warmup |
| `optm_decay_type` | str \| None | None | LR decay schedule. Choices: `sqrt`, `cosine`, `linear`, `none`. | If None, constant LR after warmup |
| `optm_min_lr_factor` | float | 0.0 | Floor LR factor at end of decay, range `[0.0, 1.0]`. Final LR = `optm_lr * optm_min_lr_factor`. | A non-zero floor prevents LR from collapsing entirely; matters when diagnosing late-training learning capacity |
| `optm_weight_decay` | float | 0.01 | Weight decay (L2 regularization) | |
| `optm_betas` | (float, float) | (0.9, 0.999) | AdamW betas | |
| `optm_grad_norm_clip` | float | 1.0 | Gradient norm clipping threshold | |
| `optm_impl` | str \| list[str] | "fused" | Optimizer implementation. Choices: `fused`, `foreach`, `for-loop`. | |

> The `[llm_lr, vlm_lr, linear_lr]` semantics for list-form `optm_lr` are a Cosmos Reason2 convention for LLM, Vision Tower and Linear Projector learning rate.

## Checkpoint Parameters (train.ckpt section)

| Parameter | Type | Default | Description | RCA Relevance |
|-----------|------|---------|-------------|---------------|
| `max_keep` | int | 5 | Max checkpoints to keep | Low value loses potentially better earlier checkpoints |
| `save_freq` | int | 20 | Save checkpoint every N steps | |

## Validation Parameters (validation section)

| Parameter | Type | Default | Description | RCA Relevance |
|-----------|------|---------|-------------|---------------|
| `enable` | bool | False | Enable validation during training | CRITICAL: should be True for convergence monitoring |
| `freq` | int | 20 | Validation frequency (steps) | |

## Dataset Parameters (dataset section)

| Parameter | Type | Default | Description | RCA Relevance |
|-----------|------|---------|-------------|---------------|
| `name` | str | "" | JSON string of dataset file paths | Lists all QA JSON files used |

## Vision Parameters (from VisionConfig)

| Parameter | Type | Default | Description | RCA Relevance |
|-----------|------|---------|-------------|---------------|
| `total_pixels` | int | 16572416 | Total pixel budget for vision input | Higher = better resolution, more compute |
| `max_pixels` | int | 1003520 | Max pixels per frame | |
| `min_pixels` | int | 100352 | Min pixels per frame | |
| `fps` | float | 8.0 | Frame rate for video processing | Must match training |
| `max_frames` | int | 40 | Maximum frames extracted | Must match training |
| `use_fps_or_nframes` | str | "fps" | Whether to use fps-based or fixed frame count | |

## Key Relationships for RCA

### Training Convergence
- **Overfitting signs:** Loss converges very early, flat for majority of training. Often caused by too many epochs for dataset size, no LR decay.
- **Underfitting signs:** Loss still decreasing when training ends. Often caused by too few epochs, LR too low, or premature termination.
- **Key factors:** `epoch`, `optm_lr`, `optm_decay_type`, dataset size

### Learning Rate Analysis
- `optm_lr` is a per-component list: `[llm_lr, vlm_lr, linear_lr]`. The three components have different roles:
  - **LLM (language model):** Adapts the language understanding and generation capabilities
  - **VLM (vision encoder):** Adapts the pretrained visual feature extraction. This is a pretrained component — aggressive fine-tuning risks destroying visual features that the model relies on for action recognition.
  - **Linear (projection head):** Adapts the vision-language projection layer
- There is no single correct LR ratio between components — different datasets succeed with different configurations (e.g., VLM LR higher, lower, or equal to LLM LR). The key factor is that the **absolute magnitude** of each LR should be appropriate for fine-tuning a pretrained model.
- **Warmup interaction:** `optm_warmup_steps` controls how quickly the LR ramps up. Short warmup with an aggressive LR causes a gradient shock early in training that can permanently damage pretrained features. Assess LR aggressiveness in context of warmup — the same LR value is safer with longer warmup.
- **When diagnosing model collapse (Pattern 9):** Assess LR aggressiveness from the training run itself — check how quickly the loss converges and whether the warmup ratio (`optm_warmup_steps / total_steps`) is adequate. The prediction distribution and loss convergence speed are the primary diagnostics.

### Frame Sampling
- With `fps=8` and `max_frames=40`, the model can see up to 5 seconds of video at full resolution
- Chunks longer than `max_frames/fps` seconds get subsampled (frames skipped)
- A 25s chunk at 8fps = 200 frames, but only 40 are selected → 0.2 effective FPS

### Training-Evaluation Consistency
The following parameters MUST match between training and evaluation:
- `fps`
- `max_frames`  
- `total_pixels` / `max_pixels` / `min_pixels`

## LoRA Configuration (PolicyConfig.lora)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `r` | int | 8 | LoRA rank |
| `lora_alpha` | float | 8.0 | LoRA scaling factor |
| `lora_dropout` | float | 0.0 | Dropout for LoRA layers |
| `target_modules` | list/str | None | Modules to apply LoRA (or "all-linear") |
| `use_rslora` | bool | False | Use rank-stabilized LoRA |
| `modules_to_save` | list | None | Modules to save fully (not LoRA) |

## SFT Data Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `mini_batch` | int | 2 | Mini-batch size per optimization step |
| `dataloader_shuffle` | bool | True | Shuffle training data |
| `balance_dp_token` | bool | True | Balance token counts across DP ranks |
| `conversation_column_name` | str | "conversations" | Column name in JSON dataset |
| `system_prompt` | str | "" | System prompt (overridden by CustomConfig) |

## GRPO Configuration (if using RL fine-tuning)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `variant` | str | "grpo" | Algorithm variant (grpo/gspo/dapo) |
| `reward_function` | str/list/dict | ["single_choice"] | Reward function(s) |
| `temperature` | float | 1.0 | Sampling temperature during rollout |
| `epsilon_low/high` | float | 0.2 | GRPO clipping bounds |
| `kl_beta` | float | 0.0 | KL penalty coefficient (0 = no reference model) |

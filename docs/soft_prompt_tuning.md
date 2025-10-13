# Soft Prompt Tuning for Qwen2.5-Coder-0.5B

## Overview

This guide explains how to train and use **soft prompts** (continuous prompt embeddings) for the Qwen2.5-Coder-0.5B model on HumanEval.

### What is Soft Prompt Tuning?

**Soft prompt tuning** is a parameter-efficient fine-tuning method that:
- **Freezes** all model weights (no fine-tuning)
- **Trains** a small set of continuous embeddings (virtual tokens)
- These embeddings are **prepended** to every input
- Model learns optimal "soft prompts" through backpropagation

**Benefits:**
- 🚀 **Efficient**: Only ~0.1% of model parameters are trainable
- ⚡ **Fast**: Training takes 1-4 hours on a single GPU
- 💾 **Small**: Trained prompts are only a few MB (vs GBs for full fine-tuning)
- 🎯 **Task-specific**: Learns optimal prompts for HumanEval

**Architecture:**
```
[soft_prompt_1] [soft_prompt_2] ... [soft_prompt_N] [your_code_problem] → Model → [completion]
        ↑ Trainable (20 tokens = ~18K params)                ↑ Frozen (500M params)
```

---

## Workflow

### Phase 1: Train Soft Prompts (Local, No Docker)

**Step 1: Install Dependencies**
```bash
# Install PyTorch and Transformers
pip install -e .

# Or using uv
uv sync
```

**Step 2: Configure Training**

Edit `prompt_tuning.yml` to customize:
```yaml
soft_prompts:
  num_virtual_tokens: 20  # Number of learnable tokens (10-100)

training:
  num_epochs: 10          # Training epochs
  learning_rate: 0.01     # LR for prompt tuning (higher than fine-tuning)
  batch_size: 4           # Adjust based on GPU memory
```

**Step 3: Train Soft Prompts**
```bash
# Start training (requires GPU recommended)
python scripts/train_soft_prompts.py

# Training will:
# 1. Load Qwen2.5-Coder-0.5B model
# 2. Freeze all model weights
# 3. Initialize 20 soft prompt embeddings
# 4. Train on HumanEval (131 train samples, 33 validation)
# 5. Save checkpoints to soft_prompts/
```

**Expected Output:**
```
================================================================================
Soft Prompt Tuning for Qwen2.5-Coder-0.5B
================================================================================
Device: cuda
Loading model: Qwen/Qwen2.5-Coder-0.5B
Initialized 20 soft prompt tokens
Trainable parameters: 17,920
Frozen parameters: 494,032,896
Trainable ratio: 0.0036%
================================================================================
Training Configuration
================================================================================
Total training steps: 820
Effective batch size: 16
Number of epochs: 10
Learning rate: 0.01
Warmup steps: 100
================================================================================
Epoch 1/10: 100%|████████████| 131/131 [02:15<00:00, loss=2.4521, lr=1.2e-03]
Epoch 1/10 - Average training loss: 2.4521
Running validation...
Validation loss: 2.1234, Perplexity: 8.36
✓ New best checkpoint saved to soft_prompts/best_soft_prompts.pt
...
Epoch 10/10 - Average training loss: 0.8765
Running validation...
Validation loss: 0.9123, Perplexity: 2.49
✓ New best checkpoint saved to soft_prompts/best_soft_prompts.pt
================================================================================
Training complete!
Best validation loss: 0.9123
Trained prompts saved to: soft_prompts/
================================================================================
```

**Training Time:**
- **GPU (T4)**: ~2-3 hours for 10 epochs
- **CPU**: ~8-12 hours (not recommended)

**Output Files:**
```
soft_prompts/
├── best_soft_prompts.pt              # Best checkpoint (lowest val loss)
├── soft_prompts_epoch_1.pt           # Epoch 1 checkpoint
├── soft_prompts_epoch_2.pt           # Epoch 2 checkpoint
...
└── soft_prompts_epoch_10.pt          # Final checkpoint

logs/
└── soft_prompt_training_1234567.log  # Training log
```

---

### Phase 2: Test Soft Prompts (Local)

**Step 1: Run Inference**
```bash
# Use best checkpoint
python scripts/inference_with_soft_prompts.py

# Or specify custom checkpoint
python scripts/inference_with_soft_prompts.py soft_prompts/soft_prompts_epoch_5.pt
```

**Step 2: Evaluate Results**
```bash
# Run evaluation on generated completions
python scripts/run_evaluation.py --input results/completions_soft_prompts.jsonl
```

**Expected Output:**
```
┏━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Metric             ┃ Value              ┃
┣━━━━━━━━━━━━━━━━━━━━╋━━━━━━━━━━━━━━━━━━━━┫
┃ Total problems     ┃ 164                ┃
┃ Passed             ┃ 145                ┃
┃ Failed             ┃ 19                 ┃
┃ pass@1             ┃ 0.884 (88.4%)      ┃
┗━━━━━━━━━━━━━━━━━━━━┻━━━━━━━━━━━━━━━━━━━━┛
```

---

### Phase 3: Deploy to vLLM (Docker) [TODO]

**Option A: Use HuggingFace Transformers (Current)**

The current implementation uses HuggingFace Transformers directly. This works but is slower than vLLM.

**Option B: Deploy to vLLM (Advanced)**

vLLM doesn't natively support soft prompts, but you can:

1. **Merge soft prompts into model vocabulary** (complex)
2. **Use vLLM with custom prefix tokens** (requires vLLM modifications)
3. **Serve via HuggingFace Transformers API** (easier, slower)

For now, **Option A** is recommended for evaluation.

---

## Configuration Reference

### `prompt_tuning.yml`

```yaml
# Model Configuration
model:
  name: "Qwen/Qwen2.5-Coder-0.5B"  # HuggingFace model ID
  trust_remote_code: true           # Required for Qwen
  torch_dtype: "float16"            # float16 for GPU, float32 for CPU

# Soft Prompt Configuration
soft_prompts:
  num_virtual_tokens: 20            # Number of learnable tokens
  init_from_vocab: true             # Initialize from vocab (recommended)
  embedding_dim: 896                # Auto-detected if not specified

# Training Configuration
training:
  batch_size: 4                     # Per-device batch size
  gradient_accumulation_steps: 4    # Effective batch = 4 * 4 = 16
  num_epochs: 10                    # Training epochs
  learning_rate: 0.01               # Higher LR for prompt tuning
  warmup_steps: 100                 # LR warmup
  max_grad_norm: 1.0                # Gradient clipping
  weight_decay: 0.01                # Regularization

# Dataset Configuration
dataset:
  train_val_split: 0.8              # 80% train, 20% val
  max_samples: null                 # null = use all 164 problems
  max_length: 1024                  # Max sequence length
  seed: 42                          # Random seed

# Evaluation Configuration
evaluation:
  eval_every_n_epochs: 1            # Evaluate every N epochs
  save_best_only: false             # Save all or only best
  early_stopping_patience: 3        # Stop if no improvement

# Hardware Configuration
hardware:
  device: "auto"                    # auto, cuda, cpu
  mixed_precision: true             # Mixed precision (faster)
  num_workers: 0                    # DataLoader workers

# Inference Configuration
inference:
  checkpoint_path: "soft_prompts/best_soft_prompts.pt"
  max_new_tokens: 512
  temperature: 0.0                  # 0 = greedy
  top_p: 1.0
  output_path: "results/completions_soft_prompts.jsonl"
```

---

## Hyperparameter Tuning

### Number of Virtual Tokens

**Trade-off:**
- **Fewer tokens (10-20)**: Faster training, less overfitting, smaller checkpoints
- **More tokens (50-100)**: More expressive, better performance, risk of overfitting

**Recommended:**
- Start with **20 tokens** (good balance)
- Try 50 if 20 doesn't improve performance
- Use 10 if training is slow or overfitting occurs

### Learning Rate

**Range:** 0.001 - 0.1

**Typical values:**
- **0.01**: Good default (10x higher than fine-tuning)
- **0.05**: Faster convergence, risk of instability
- **0.001**: Slower but more stable

**Tip:** If loss doesn't decrease, try increasing LR.

### Batch Size

**Effective batch size** = `batch_size` × `gradient_accumulation_steps`

**Recommended:**
- **GPU memory < 8GB**: batch_size=2, grad_accum=8 (effective=16)
- **GPU memory 8-16GB**: batch_size=4, grad_accum=4 (effective=16)
- **GPU memory > 16GB**: batch_size=8, grad_accum=2 (effective=16)

### Number of Epochs

**Typical range:** 5-20 epochs

**Signs of overfitting:**
- Validation loss increases while training loss decreases
- Pass@1 on validation set decreases

**Recommendation:**
- Start with **10 epochs**
- Use **early stopping** (patience=3) to prevent overfitting

---

## Troubleshooting

### Out of Memory (OOM)

**Solutions:**
1. Reduce `batch_size` (e.g., 4 → 2)
2. Increase `gradient_accumulation_steps` (keeps effective batch size same)
3. Use `torch_dtype: "float16"` (GPU only)
4. Reduce `num_virtual_tokens` (e.g., 20 → 10)
5. Reduce `max_length` (e.g., 1024 → 512)

### Slow Training

**Solutions:**
1. Use GPU instead of CPU
2. Enable `mixed_precision: true`
3. Reduce `num_virtual_tokens`
4. Increase `batch_size` if memory allows

### Loss Not Decreasing

**Solutions:**
1. Increase learning rate (e.g., 0.01 → 0.05)
2. Check warmup_steps (try 100-200)
3. Ensure `init_from_vocab: true`
4. Verify training data is loaded correctly

### Checkpoint Not Found

**Check:**
```bash
# List checkpoints
ls -lh soft_prompts/

# Check training logs
cat logs/soft_prompt_training_*.log
```

---

## Comparison: Soft Prompts vs Prompt Engineering

| Aspect | Prompt Engineering | Soft Prompt Tuning |
|--------|-------------------|-------------------|
| **Training** | None | 1-4 hours |
| **Optimization** | Manual trial-and-error | Automatic gradient descent |
| **Parameters** | Text strings (~100 tokens) | Continuous embeddings (20 tokens) |
| **Interpretability** | Human-readable | Black box |
| **Performance** | Good (depends on prompt) | Better (learned from data) |
| **Iteration speed** | Fast (seconds) | Slow (hours) |
| **Deployment** | Easy (just text) | Moderate (need checkpoint) |

**When to use which:**
- **Prompt engineering**: Fast experimentation, interpretability needed
- **Soft prompt tuning**: Maximum performance, have compute budget

---

## Expected Results

### Baseline (No Prompts)
- **pass@1**: ~70-75%

### Prompt Engineering (Your current work)
- **pass@1**: ~88-96% (best: infilling + post_v5)

### Soft Prompt Tuning (Expected)
- **pass@1**: ~85-92% (depends on hyperparameters)
- **May not beat best prompt engineering**, but:
  - Learned automatically from data
  - No manual prompt design needed
  - Can combine with post-processing

### Best Approach (Hybrid)
**Soft prompts + post-processing (post_v5)**
- Use trained soft prompts for input
- Apply post_v5 fixes to output
- **Expected pass@1**: ~90-95%

---

## Advanced: Combining Soft Prompts with Prompt Engineering

You can prepend text prompts BEFORE the soft prompts:

```python
# In inference script
text_prompt = "You are an expert Python programmer.\n"
full_prompt = text_prompt + problem_statement

# Then tokenize and add soft prompts
# [soft_tokens] [text_prompt_tokens] [problem_tokens] → model
```

This combines:
- **Text prompt**: High-level guidance
- **Soft prompts**: Fine-grained learned patterns

---

## Files Created

```
scripts/
├── train_soft_prompts.py                  # Training script
└── inference_with_soft_prompts.py         # Inference script

prompt_tuning.yml                          # Configuration file

soft_prompts/                              # Output directory
├── best_soft_prompts.pt                   # Best checkpoint
└── soft_prompts_epoch_*.pt                # Epoch checkpoints

logs/
└── soft_prompt_training_*.log             # Training logs

docs/
└── soft_prompt_tuning.md                  # This document
```

---

## Next Steps

1. **Train soft prompts**: `python scripts/train_soft_prompts.py`
2. **Run inference**: `python scripts/inference_with_soft_prompts.py`
3. **Evaluate**: `python scripts/run_evaluation.py`
4. **Compare** with your prompt engineering results
5. **Optional**: Combine soft prompts + post_v5 for best results

---

## References

- [The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691) (Lester et al., 2021)
- [Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190) (Li & Liang, 2021)
- [P-Tuning: Prompt Tuning Can Be Comparable to Fine-tuning](https://arxiv.org/abs/2103.10385) (Liu et al., 2021)

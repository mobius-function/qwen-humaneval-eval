#!/usr/bin/env python3
"""
Soft Prompt Tuning for Qwen2.5-Coder-0.5B on HumanEval.

This script trains continuous soft prompt embeddings that are prepended to every
input. The model weights remain frozen - only the prompt embeddings are updated.

Key features:
- Freezes base model weights (efficient training)
- Learns N virtual tokens (configurable) as soft prompts
- Trains on HumanEval with code completion objective
- Saves trained prompt embeddings for deployment
- Supports gradient accumulation for large batches
- Implements early stopping based on validation loss

Training approach:
1. Initialize soft prompt embeddings randomly or from vocabulary
2. Freeze all model parameters except prompt embeddings
3. For each HumanEval problem:
   - Prepend soft prompts to problem statement
   - Generate completion
   - Compute loss against ground truth
   - Update only prompt embeddings via backprop
4. Evaluate on validation set periodically
5. Save best prompts based on validation loss

Usage:
    python scripts/train_soft_prompts.py
    python scripts/train_soft_prompts.py --config custom_config.yml
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class SoftPromptTuning(nn.Module):
    """
    Soft prompt tuning wrapper for causal language models.

    Adds learnable prompt embeddings that are prepended to the input.
    The base model weights remain frozen during training.
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_virtual_tokens: int = 20,
        embedding_dim: int = 896,  # Qwen2.5-Coder-0.5B hidden size
        init_from_vocab: bool = True,
        tokenizer=None,
    ):
        """
        Initialize soft prompt tuning.

        Args:
            base_model: Pre-trained language model
            num_virtual_tokens: Number of virtual tokens to prepend
            embedding_dim: Dimension of token embeddings
            init_from_vocab: If True, initialize from random vocab tokens
            tokenizer: Tokenizer for vocab initialization
        """
        super().__init__()
        self.base_model = base_model
        self.num_virtual_tokens = num_virtual_tokens
        self.embedding_dim = embedding_dim

        # Freeze base model
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Create soft prompt embeddings
        if init_from_vocab and tokenizer is not None:
            # Initialize from random vocabulary embeddings
            # Get the device of the model embeddings
            embedding_device = base_model.get_input_embeddings().weight.device
            init_ids = torch.randint(0, len(tokenizer), (num_virtual_tokens,)).to(embedding_device)
            with torch.no_grad():
                init_embeds = base_model.get_input_embeddings()(init_ids)
            self.soft_prompts = nn.Parameter(init_embeds.clone())
        else:
            # Random initialization
            self.soft_prompts = nn.Parameter(
                torch.randn(num_virtual_tokens, embedding_dim)
            )

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in self.base_model.parameters())

        logging.info(f"Initialized {num_virtual_tokens} soft prompt tokens")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Frozen parameters: {frozen_params:,}")
        logging.info(
            f"Trainable ratio: {trainable_params / (trainable_params + frozen_params) * 100:.4f}%"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with soft prompts prepended.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            labels: Target token IDs for loss computation [batch, seq_len]

        Returns:
            Model outputs with loss if labels provided
        """
        batch_size = input_ids.shape[0]

        # Get input embeddings
        input_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Expand soft prompts for batch
        # [num_virtual_tokens, embedding_dim] -> [batch, num_virtual_tokens, embedding_dim]
        soft_prompt_embeds = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate soft prompts with input embeddings
        # [batch, num_virtual_tokens + seq_len, embedding_dim]
        inputs_embeds = torch.cat([soft_prompt_embeds, input_embeds], dim=1)

        # Update attention mask to account for soft prompts
        # [batch, num_virtual_tokens + seq_len]
        prompt_attention = torch.ones(
            batch_size,
            self.num_virtual_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

        # Update labels if provided
        if labels is not None:
            # Add -100 (ignore index) for soft prompt positions
            prompt_labels = torch.full(
                (batch_size, self.num_virtual_tokens),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([prompt_labels, labels], dim=1)

        # Forward through base model
        outputs = self.base_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels
        )

        return outputs

    def generate(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ):
        """
        Generate text with soft prompts prepended.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            attention_mask: Attention mask [batch, seq_len]
            **kwargs: Additional generation arguments (max_new_tokens, temperature, etc.)

        Returns:
            Generated token IDs
        """
        batch_size = input_ids.shape[0]

        # Get input embeddings
        input_embeds = self.base_model.get_input_embeddings()(input_ids)

        # Prepend soft prompts
        soft_prompt_embeds = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        inputs_embeds = torch.cat([soft_prompt_embeds, input_embeds], dim=1)

        # Update attention mask
        prompt_attention = torch.ones(
            batch_size,
            self.num_virtual_tokens,
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        attention_mask = torch.cat([prompt_attention, attention_mask], dim=1)

        # Generate
        outputs = self.base_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
        )

        return outputs


class HumanEvalDataset(Dataset):
    """Dataset wrapper for HumanEval with prompt/completion pairs."""

    def __init__(
        self, split: str = "train", max_samples: Optional[int] = None, seed: int = 42
    ):
        """
        Initialize HumanEval dataset.

        Args:
            split: "train" or "test" (we'll split HumanEval for validation)
            max_samples: Limit number of samples (for debugging)
            seed: Random seed for train/val split
        """
        # Load full HumanEval (it's all "test" split originally)
        dataset = load_dataset("openai_humaneval", split="test")

        # HumanEval has 164 problems - split 80/20 for train/val
        # Use deterministic split based on seed
        torch.manual_seed(seed)
        indices = list(range(len(dataset)))
        shuffled = torch.randperm(len(indices)).tolist()

        split_idx = int(0.8 * len(dataset))
        if split == "train":
            self.indices = [indices[i] for i in shuffled[:split_idx]]
        else:
            self.indices = [indices[i] for i in shuffled[split_idx:]]

        self.dataset = dataset

        if max_samples is not None:
            self.indices = self.indices[:max_samples]

        logging.info(f"Loaded {len(self.indices)} {split} samples")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        """
        Get a single training example.

        Returns:
            Dict with 'prompt' (problem statement) and 'canonical_solution' (ground truth)
        """
        real_idx = self.indices[idx]
        item = self.dataset[real_idx]

        return {
            "task_id": item["task_id"],
            "prompt": item["prompt"],  # Function signature + docstring
            "canonical_solution": item[
                "canonical_solution"
            ],  # Ground truth implementation
            "test": item["test"],  # Unit tests
            "entry_point": item["entry_point"],
        }


def collate_fn(batch: List[Dict], tokenizer, max_length: int = 1024):
    """
    Collate function for DataLoader.

    Tokenizes prompts and solutions, creating input_ids and labels.
    """
    prompts = [item["prompt"] for item in batch]
    solutions = [item["canonical_solution"] for item in batch]

    # Create full text: prompt + solution
    full_texts = [p + s for p, s in zip(prompts, solutions)]

    # Tokenize
    encoded = tokenizer(
        full_texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # For labels, we want to predict the solution tokens
    # Mask out prompt tokens (set to -100)
    labels = encoded["input_ids"].clone()

    for i, prompt in enumerate(prompts):
        prompt_tokens = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        prompt_len = len(prompt_tokens)
        labels[i, :prompt_len] = -100  # Don't compute loss on prompt

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
        "task_ids": [item["task_id"] for item in batch],
    }


def evaluate_model(
    model: SoftPromptTuning,
    tokenizer,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluate model on validation set.

    Returns:
        Dict with metrics (loss, perplexity)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            # Accumulate loss
            loss = outputs.loss
            if loss is not None:
                num_tokens = (labels != -100).sum().item()
                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {"loss": avg_loss, "perplexity": perplexity}


def load_config(config_path: str = "prompt_tuning.yml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train_soft_prompts(config_path: str = "prompt_tuning.yml"):
    """
    Main training function for soft prompt tuning.

    Args:
        config_path: Path to YAML configuration file
    """
    # Load configuration
    config = load_config(config_path)

    # Extract config sections
    model_config = config["model"]
    prompt_config = config["soft_prompts"]
    train_config = config["training"]
    dataset_config = config["dataset"]
    eval_config = config["evaluation"]
    output_config = config["output"]
    hardware_config = config["hardware"]
    log_config = config["logging"]

    # Set random seeds
    set_seed(dataset_config["seed"])

    # Setup logging
    log_dir = Path(output_config["log_dir"])
    log_dir.mkdir(exist_ok=True, parents=True)

    logging.basicConfig(
        level=logging.INFO if log_config["verbose"] else logging.WARNING,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                log_dir / f"soft_prompt_training_{int(time.time())}.log"
            ),
            logging.StreamHandler(),
        ],
    )

    logging.info("=" * 80)
    logging.info("Soft Prompt Tuning for Qwen2.5-Coder-0.5B")
    logging.info("=" * 80)

    # Create output directory
    output_path = Path(output_config["output_dir"])
    output_path.mkdir(exist_ok=True, parents=True)

    # Determine device
    if hardware_config["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(hardware_config["device"])

    logging.info(f"Device: {device}")

    # Load tokenizer and model
    logging.info(f"Loading model: {model_config['name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"], trust_remote_code=model_config["trust_remote_code"]
    )

    # Add padding token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Determine dtype
    dtype_map = {"float16": torch.float16, "float32": torch.float32, "auto": "auto"}
    dtype = dtype_map.get(model_config["torch_dtype"], torch.float32)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        trust_remote_code=model_config["trust_remote_code"],
        torch_dtype=dtype if dtype != "auto" else (torch.float16 if device.type == "cuda" else torch.float32),
        device_map=device,
    )

    # Get embedding dimension
    embedding_dim = (
        prompt_config.get("embedding_dim")
        or base_model.get_input_embeddings().embedding_dim
    )

    # Wrap with soft prompt tuning
    model = SoftPromptTuning(
        base_model=base_model,
        num_virtual_tokens=prompt_config["num_virtual_tokens"],
        embedding_dim=embedding_dim,
        init_from_vocab=prompt_config["init_from_vocab"],
        tokenizer=tokenizer,
    )
    model = model.to(device)

    # Create datasets
    train_dataset = HumanEvalDataset(
        split="train",
        max_samples=dataset_config["max_samples"],
        seed=dataset_config["seed"],
    )
    val_dataset = HumanEvalDataset(
        split="test",
        max_samples=dataset_config["max_samples"],
        seed=dataset_config["seed"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config["batch_size"],
        shuffle=True,
        collate_fn=lambda batch: collate_fn(
            batch, tokenizer, max_length=dataset_config["max_length"]
        ),
        num_workers=hardware_config["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config["batch_size"],
        shuffle=False,
        collate_fn=lambda batch: collate_fn(
            batch, tokenizer, max_length=dataset_config["max_length"]
        ),
        num_workers=hardware_config["num_workers"],
    )

    # Setup optimizer (only optimize soft prompts)
    optimizer = torch.optim.AdamW(
        [model.soft_prompts],
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    # Setup learning rate scheduler
    total_steps = (
        len(train_loader)
        * train_config["num_epochs"]
        // train_config["gradient_accumulation_steps"]
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=train_config["warmup_steps"],
        num_training_steps=total_steps,
    )

    # Training loop
    logging.info("=" * 80)
    logging.info("Training Configuration")
    logging.info("=" * 80)
    logging.info(f"Total training steps: {total_steps}")
    logging.info(
        f"Effective batch size: {train_config['batch_size'] * train_config['gradient_accumulation_steps']}"
    )
    logging.info(f"Number of epochs: {train_config['num_epochs']}")
    logging.info(f"Learning rate: {train_config['learning_rate']}")
    logging.info(f"Warmup steps: {train_config['warmup_steps']}")
    logging.info("=" * 80)

    best_val_loss = float("inf")
    patience_counter = 0
    global_step = 0

    for epoch in range(train_config["num_epochs"]):
        model.train()
        epoch_loss = 0.0
        optimizer.zero_grad()

        progress_bar = tqdm(
            train_loader, desc=f"Epoch {epoch+1}/{train_config['num_epochs']}"
        )

        for step, batch in enumerate(progress_bar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )

            loss = outputs.loss / train_config["gradient_accumulation_steps"]
            loss.backward()

            epoch_loss += loss.item() * train_config["gradient_accumulation_steps"]

            # Update weights
            if (step + 1) % train_config["gradient_accumulation_steps"] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    [model.soft_prompts], train_config["max_grad_norm"]
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Update progress bar
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss.item() * train_config['gradient_accumulation_steps']:.4f}",
                        "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                    }
                )

        avg_train_loss = epoch_loss / len(train_loader)
        logging.info(
            f"Epoch {epoch+1}/{train_config['num_epochs']} - Average training loss: {avg_train_loss:.4f}"
        )

        # Validation
        if (epoch + 1) % eval_config["eval_every_n_epochs"] == 0:
            logging.info("Running validation...")
            val_metrics = evaluate_model(model, tokenizer, val_loader, device)
            logging.info(
                f"Validation loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.2f}"
            )

            # Save checkpoint
            checkpoint_data = {
                "soft_prompts": model.soft_prompts.data.cpu(),
                "num_virtual_tokens": prompt_config["num_virtual_tokens"],
                "embedding_dim": embedding_dim,
                "epoch": epoch,
                "global_step": global_step,
                "val_loss": val_metrics["loss"],
                "val_perplexity": val_metrics["perplexity"],
                "config": config,
            }

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0

                checkpoint_path = output_path / output_config["best_checkpoint_name"]
                torch.save(checkpoint_data, checkpoint_path)
                logging.info(f"✓ New best checkpoint saved to {checkpoint_path}")
            else:
                patience_counter += 1

            # Save epoch checkpoint
            if not eval_config["save_best_only"]:
                checkpoint_path = (
                    output_path / f"{output_config['checkpoint_prefix']}_{epoch+1}.pt"
                )
                torch.save(checkpoint_data, checkpoint_path)
                logging.info(f"✓ Checkpoint saved to {checkpoint_path}")

            # Early stopping
            if patience_counter >= eval_config["early_stopping_patience"]:
                logging.info(
                    f"Early stopping triggered after {epoch+1} epochs (patience: {eval_config['early_stopping_patience']})"
                )
                break

    logging.info("=" * 80)
    logging.info("Training complete!")
    logging.info(f"Best validation loss: {best_val_loss:.4f}")
    logging.info(f"Trained prompts saved to: {output_path}")
    logging.info("=" * 80)


def main():
    import sys

    # Support custom config path via command line
    config_path = "prompt_tuning.yml"
    if len(sys.argv) > 1:
        config_path = sys.argv[1]

    train_soft_prompts(config_path)


if __name__ == "__main__":
    main()

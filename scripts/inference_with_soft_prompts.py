#!/usr/bin/env python3
"""
Inference with Trained Soft Prompts for HumanEval.

This script loads trained soft prompt embeddings and uses them for inference
on the HumanEval dataset. The soft prompts are prepended to each problem.

Usage:
    python scripts/inference_with_soft_prompts.py
    python scripts/inference_with_soft_prompts.py --checkpoint soft_prompts/best_soft_prompts.pt
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import torch
import yaml
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import SoftPromptTuning class
from scripts.train_soft_prompts import SoftPromptTuning


def load_trained_prompts(checkpoint_path: str, base_model, tokenizer):
    """
    Load trained soft prompts from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        base_model: Base language model
        tokenizer: Tokenizer

    Returns:
        SoftPromptTuning model with loaded prompts
    """
    logging.info(f"Loading trained prompts from: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract checkpoint info
    soft_prompts = checkpoint["soft_prompts"]
    num_virtual_tokens = checkpoint["num_virtual_tokens"]
    embedding_dim = checkpoint["embedding_dim"]

    logging.info(f"Checkpoint info:")
    logging.info(f"  - Num virtual tokens: {num_virtual_tokens}")
    logging.info(f"  - Embedding dim: {embedding_dim}")
    logging.info(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
    logging.info(f"  - Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    logging.info(
        f"  - Val perplexity: {checkpoint.get('val_perplexity', 'N/A'):.2f}"
    )

    # Create model
    model = SoftPromptTuning(
        base_model=base_model,
        num_virtual_tokens=num_virtual_tokens,
        embedding_dim=embedding_dim,
        init_from_vocab=False,  # We're loading trained prompts
        tokenizer=tokenizer,
    )

    # Load trained prompts
    model.soft_prompts.data = soft_prompts

    return model


def generate_completions(
    model: SoftPromptTuning,
    tokenizer,
    dataset: List[Dict],
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    output_path: str = "results/completions_soft_prompts.jsonl",
):
    """
    Generate code completions using soft prompts.

    Args:
        model: SoftPromptTuning model
        tokenizer: Tokenizer
        dataset: HumanEval dataset
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        output_path: Path to save completions

    Returns:
        List of completions
    """
    model.eval()
    model = model.to(device)

    completions = []

    with torch.no_grad():
        for item in tqdm(dataset, desc="Generating completions"):
            prompt = item["prompt"]

            # Tokenize
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)

            # Generate with soft prompts
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # Decode
            # Remove soft prompt tokens and input prompt tokens
            prompt_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][
                prompt_len + model.num_virtual_tokens :
            ]  # Skip soft prompts
            completion = tokenizer.decode(generated_ids, skip_special_tokens=True)

            completions.append(
                {
                    "task_id": item["task_id"],
                    "prompt": prompt,
                    "completion": completion,
                    "full_code": prompt + completion,
                }
            )

    # Save to file
    output_file = Path(output_path)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(output_file, "w") as f:
        for completion in completions:
            f.write(json.dumps(completion) + "\n")

    logging.info(f"Saved {len(completions)} completions to {output_file}")

    return completions


def main():
    import sys

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Parse arguments
    checkpoint_path = "soft_prompts/best_soft_prompts.pt"
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]

    # Load config (optional)
    config_path = "prompt_tuning.yml"
    if Path(config_path).exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            model_name = config["model"]["name"]
            max_new_tokens = config.get("inference", {}).get("max_new_tokens", 512)
            temperature = config.get("inference", {}).get("temperature", 0.0)
            top_p = config.get("inference", {}).get("top_p", 1.0)
            output_path = config.get("inference", {}).get(
                "output_path", "results/completions_soft_prompts.jsonl"
            )
    else:
        model_name = "Qwen/Qwen2.5-Coder-0.5B"
        max_new_tokens = 512
        temperature = 0.0
        top_p = 1.0
        output_path = "results/completions_soft_prompts.jsonl"

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load tokenizer and base model
    logging.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map=device,
    )

    # Load trained soft prompts
    model = load_trained_prompts(checkpoint_path, base_model, tokenizer)

    # Load HumanEval dataset
    logging.info("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    dataset = list(dataset)

    logging.info(f"Loaded {len(dataset)} problems")

    # Generate completions
    logging.info("=" * 80)
    logging.info("Generating completions with soft prompts")
    logging.info(f"  - Max new tokens: {max_new_tokens}")
    logging.info(f"  - Temperature: {temperature}")
    logging.info(f"  - Top-p: {top_p}")
    logging.info("=" * 80)

    completions = generate_completions(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        device=device,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        output_path=output_path,
    )

    logging.info("=" * 80)
    logging.info("Inference complete!")
    logging.info(
        f"Run evaluation: python scripts/run_evaluation.py --input {output_path}"
    )
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

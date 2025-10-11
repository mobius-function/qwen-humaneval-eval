#!/usr/bin/env python3
"""Inference script to generate code completions for HumanEval dataset."""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict

import requests
from datasets import load_dataset
from tqdm import tqdm

from prompts.code_completion import create_completion_prompt, post_process_completion
from prompts.advanced_prompts import PROMPT_STRATEGIES, POSTPROCESS_STRATEGIES


class VLLMInference:
    """Handle inference with vLLM OpenAI-compatible API."""

    def __init__(self, api_url: str = "http://localhost:8000/v1"):
        self.api_url = api_url
        self.completions_url = f"{api_url}/completions"

    def generate_completion(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.95,
        stop: List[str] = None,
    ) -> str:
        """
        Generate code completion from the model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            Generated completion text
        """
        if stop is None:
            stop = ["\nclass ", "\ndef ", "\n#", "\nif __name__"]

        payload = {
            "model": "Qwen/Qwen2.5-Coder-0.5B",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop,
        }

        try:
            response = requests.post(self.completions_url, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["text"]
        except Exception as e:
            print(f"Error during inference: {e}")
            return ""


def load_humaneval() -> List[Dict]:
    """Load HumanEval dataset from Hugging Face."""
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    return list(dataset)


def run_inference(
    output_path: str = "results/completions.jsonl",
    api_url: str = None,
    temperature: float = 0.2,
    max_samples: int = None,
    prompt_strategy: str = "infilling",
    postprocess_strategy: str = "smart",
):
    """
    Run inference on HumanEval dataset.

    Args:
        output_path: Path to save completions
        api_url: vLLM API URL
        temperature: Sampling temperature
        max_samples: Limit number of samples (for testing)
        prompt_strategy: Prompting strategy (minimal, infilling, instructional, fewshot, cot)
        postprocess_strategy: Post-processing strategy (basic, smart)
    """
    # Initialize inference client
    if api_url is None:
        api_url = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")

    inference = VLLMInference(api_url)

    # Get prompt and postprocess functions
    prompt_fn = PROMPT_STRATEGIES.get(prompt_strategy, PROMPT_STRATEGIES['infilling'])
    postprocess_fn = POSTPROCESS_STRATEGIES.get(postprocess_strategy, POSTPROCESS_STRATEGIES['smart'])

    # Load dataset
    problems = load_humaneval()

    if max_samples:
        problems = problems[:max_samples]
        print(f"Running on first {max_samples} samples only")

    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Generate completions
    results = []
    print(f"\nGenerating completions for {len(problems)} problems...")
    print(f"Prompt strategy: {prompt_strategy}")
    print(f"Postprocess strategy: {postprocess_strategy}")
    print(f"Temperature: {temperature}")

    for problem in tqdm(problems, desc="Inference"):
        task_id = problem["task_id"]
        prompt_text = problem["prompt"]
        entry_point = problem.get("entry_point")

        # Create prompt using selected strategy
        full_prompt = prompt_fn(prompt_text)

        # Generate completion
        completion = inference.generate_completion(
            prompt=full_prompt,
            temperature=temperature,
        )

        # Post-process using selected strategy
        cleaned_completion = postprocess_fn(completion, full_prompt, entry_point)

        # Combine prompt + completion for evaluation
        full_code = prompt_text + cleaned_completion

        result = {
            "task_id": task_id,
            "prompt": prompt_text,
            "completion": cleaned_completion,
            "full_code": full_code,
        }

        results.append(result)

        # Save incrementally
        with open(output_file, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

    print(f"\nSaved {len(results)} completions to {output_path}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run inference on HumanEval")
    parser.add_argument(
        "--output",
        type=str,
        default="results/completions.jsonl",
        help="Output file path",
    )
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help="vLLM API URL (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing)",
    )
    parser.add_argument(
        "--prompt-strategy",
        type=str,
        default="infilling",
        choices=list(PROMPT_STRATEGIES.keys()),
        help="Prompting strategy to use",
    )
    parser.add_argument(
        "--postprocess-strategy",
        type=str,
        default="smart",
        choices=list(POSTPROCESS_STRATEGIES.keys()),
        help="Post-processing strategy to use",
    )

    args = parser.parse_args()

    run_inference(
        output_path=args.output,
        api_url=args.api_url,
        temperature=args.temperature,
        max_samples=args.max_samples,
        prompt_strategy=args.prompt_strategy,
        postprocess_strategy=args.postprocess_strategy,
    )

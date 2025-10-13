#!/usr/bin/env python3
"""Inference script to generate code completions for HumanEval dataset."""

import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Tuple

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


def setup_logging(output_path: str, prompt_strategy: str, postprocess_strategy: str):
    """
    Setup logging for inference run.

    Args:
        output_path: Output file path (used to derive log file names)
        prompt_strategy: Prompt strategy name
        postprocess_strategy: Post-processing strategy name

    Returns:
        Tuple of (main_logger, debug_logger)
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Derive experiment name from output path or use strategy names
    output_stem = Path(output_path).stem
    if output_stem == "completions":
        # Default name - use strategies
        exp_name = f"{prompt_strategy}_{postprocess_strategy}"
    else:
        # Use the output file name
        exp_name = output_stem.replace("completions_", "")

    # Setup main logger
    main_logger = logging.getLogger(f"inference.{exp_name}")
    main_logger.setLevel(logging.INFO)
    main_logger.handlers.clear()

    # Main log file handler
    main_handler = logging.FileHandler(log_dir / f"{exp_name}.log", mode='w')
    main_handler.setLevel(logging.INFO)
    main_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    main_handler.setFormatter(main_formatter)
    main_logger.addHandler(main_handler)

    # Note: We no longer create a debug logger here
    # Failure logging is now handled in run_evaluation.py after evaluation completes

    return main_logger, None


def process_single_problem(
    problem: Dict,
    inference: VLLMInference,
    prompt_fn,
    postprocess_fn,
    temperature: float,
) -> Dict:
    """
    Process a single problem: generate and post-process completion.

    Args:
        problem: HumanEval problem dictionary
        inference: VLLMInference instance
        prompt_fn: Prompt generation function
        postprocess_fn: Post-processing function
        temperature: Sampling temperature

    Returns:
        Result dictionary with completion
    """
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

    return {
        "task_id": task_id,
        "prompt": prompt_text,
        "completion": cleaned_completion,
        "raw_completion": completion,  # Store raw output for failure analysis
        "full_code": full_code,
    }


def run_inference(
    output_path: str = "results/completions.jsonl",
    api_url: str = None,
    temperature: float = 0.2,
    max_samples: int = None,
    prompt_strategy: str = "infilling",
    postprocess_strategy: str = "smart",
    num_workers: int = None,
):
    """
    Run inference on HumanEval dataset with parallel API calls.

    Args:
        output_path: Path to save completions
        api_url: vLLM API URL
        temperature: Sampling temperature
        max_samples: Limit number of samples (for testing)
        prompt_strategy: Prompting strategy (minimal, infilling, instructional, fewshot, cot)
        postprocess_strategy: Post-processing strategy (basic, smart)
        num_workers: Number of parallel workers (default: 16)
    """
    # Setup logging
    main_logger, _ = setup_logging(output_path, prompt_strategy, postprocess_strategy)

    # Initialize inference client
    if api_url is None:
        api_url = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")

    inference = VLLMInference(api_url)
    main_logger.info(f"Initialized vLLM client at {api_url}")

    # Get prompt and postprocess functions
    prompt_fn = PROMPT_STRATEGIES.get(prompt_strategy, PROMPT_STRATEGIES['infilling'])
    postprocess_fn = POSTPROCESS_STRATEGIES.get(postprocess_strategy, POSTPROCESS_STRATEGIES['basic'])

    # Load dataset
    problems = load_humaneval()
    main_logger.info(f"Loaded HumanEval dataset: {len(problems)} problems")

    if max_samples:
        problems = problems[:max_samples]
        print(f"Running on first {max_samples} samples only")
        main_logger.info(f"Limited to first {max_samples} samples")

    # Create output directory
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Determine number of workers
    if num_workers is None:
        num_workers = 16  # Default for I/O-bound API calls

    # Generate completions in parallel
    print(f"\nGenerating completions for {len(problems)} problems...")
    print(f"Prompt strategy: {prompt_strategy}")
    print(f"Postprocess strategy: {postprocess_strategy}")
    print(f"Temperature: {temperature}")
    print(f"Workers: {num_workers}")

    main_logger.info(f"Starting parallel inference on {len(problems)} problems with {num_workers} workers")
    main_logger.info(f"Configuration: strategy={prompt_strategy}, postprocess={postprocess_strategy}, temp={temperature}")

    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_problem = {
            executor.submit(
                process_single_problem,
                problem,
                inference,
                prompt_fn,
                postprocess_fn,
                temperature,
            ): problem
            for problem in problems
        }

        # Collect results as they complete
        for future in tqdm(as_completed(future_to_problem), total=len(problems), desc="Inference"):
            try:
                result = future.result()
                results.append(result)

                # Save incrementally (sorted by task_id for consistency)
                results_sorted = sorted(results, key=lambda x: x["task_id"])
                with open(output_file, "w") as f:
                    for r in results_sorted:
                        f.write(json.dumps(r) + "\n")
            except Exception as e:
                problem = future_to_problem[future]
                main_logger.error(f"Failed to process {problem['task_id']}: {e}")
                print(f"Error processing {problem['task_id']}: {e}")

    print(f"\nSaved {len(results)} completions to {output_path}")
    main_logger.info(f"Completed inference: {len(results)} completions saved to {output_path}")
    main_logger.info(f"Log files: logs/{Path(output_path).stem.replace('completions_', '')}.log")

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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: 16)",
    )

    args = parser.parse_args()

    run_inference(
        output_path=args.output,
        api_url=args.api_url,
        temperature=args.temperature,
        max_samples=args.max_samples,
        prompt_strategy=args.prompt_strategy,
        postprocess_strategy=args.postprocess_strategy,
        num_workers=args.num_workers,
    )

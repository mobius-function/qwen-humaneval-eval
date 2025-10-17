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

from prompts.advanced_prompts import load_prompt, list_available_strategies, get_postprocess_function


class VLLMInference:
    """Handle inference with vLLM OpenAI-compatible API."""

    def __init__(self, api_url: str = "http://localhost:8000/v1", api_mode: str = "completion", model_name: str = None):
        """
        Initialize vLLM inference client.

        Args:
            api_url: Base URL for vLLM API
            api_mode: API mode - "completion" or "chat"
            model_name: Model name to use in API calls (optional)
        """
        self.api_url = api_url
        self.api_mode = api_mode
        self.model_name = model_name or "Qwen/Qwen2.5-Coder-0.5B"
        self.completions_url = f"{api_url}/completions"
        self.chat_url = f"{api_url}/chat/completions"

    def generate_completion(
        self,
        prompt,  # str for completion mode, dict for chat mode
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: List[str] = None,
    ) -> str:
        """
        Generate code completion from the model.

        Args:
            prompt: Input prompt (str for completion mode, dict with 'system'/'user' keys for chat mode)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            Generated completion text
        """
        if stop is None:
            stop = []

        if self.api_mode == "chat":
            # Chat API mode - expect prompt to be a dict with system/user messages
            if isinstance(prompt, dict):
                messages = [
                    {"role": "system", "content": prompt["system"]},
                    {"role": "user", "content": prompt["user"]}
                ]
            else:
                # Fallback if prompt is just a string
                messages = [
                    {"role": "system", "content": "You are an expert Python programmer."},
                    {"role": "user", "content": prompt}
                ]

            payload = {
                "model": self.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop,
            }

            try:
                response = requests.post(self.chat_url, json=payload, timeout=60)
                response.raise_for_status()
                result = response.json()
                return result["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"Error during chat inference: {e}")
                return ""
        else:
            # Completion API mode - expect prompt to be a string
            if isinstance(prompt, dict):
                # If we got a dict but we're in completion mode, just use the user content
                prompt_str = prompt.get("user", str(prompt))
            else:
                prompt_str = prompt

            payload = {
                "model": self.model_name,
                "prompt": prompt_str,
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
                print(f"Error during completion inference: {e}")
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
    log_dir = Path("logs/inference")
    log_dir.mkdir(parents=True, exist_ok=True)

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
    max_tokens: int,
    top_p: float,
    stop: List[str],
) -> Dict:
    """
    Process a single problem: generate and post-process completion.

    Args:
        problem: HumanEval problem dictionary
        inference: VLLMInference instance
        prompt_fn: Prompt generation function
        postprocess_fn: Post-processing function
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        stop: Stop sequences

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
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
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
    api_mode: str = "completion",
    temperature: float = 0.0,
    max_samples: int = None,
    prompt_strategy: str = "infilling",
    postprocess_strategy: str = "none",
    num_workers: int = None,
    max_tokens: int = 512,
    top_p: float = 1.0,
    stop: List[str] = None,
):
    """
    Run inference on HumanEval dataset with parallel API calls.

    Args:
        output_path: Path to save completions
        api_url: vLLM API URL
        api_mode: API mode - "completion" or "chat"
        temperature: Sampling temperature
        max_samples: Limit number of samples (for testing)
        prompt_strategy: Prompting strategy (minimal, infilling, instructional, fewshot, cot, etc.)
        postprocess_strategy: Post-processing strategy (none=raw output, post_v1=fix crashes only)
        num_workers: Number of parallel workers (default: 16)
        max_tokens: Maximum tokens to generate (default: 512)
        top_p: Nucleus sampling parameter (default: 1.0)
        stop: Stop sequences (default: ["\n\n", "\ndef ", "\nclass ", "\nif "])
    """
    # Setup logging
    main_logger, _ = setup_logging(output_path, prompt_strategy, postprocess_strategy)

    # Initialize inference client
    # Load config to get model-specific URLs and stop sequences
    import yaml
    try:
        with open('config.yml', 'r') as f:
            config = yaml.safe_load(f)

        if api_url is None:
            if api_mode == "chat":
                api_url = config['vllm'].get('chat_api_url', 'http://localhost:8001/v1')
                model_name = config['vllm'].get('chat_model', 'Qwen/Qwen2.5-Coder-0.5B-Instruct')
            else:
                api_url = config['vllm'].get('completion_api_url', 'http://localhost:8000/v1')
                model_name = config['vllm'].get('completion_model', 'Qwen/Qwen2.5-Coder-0.5B')
        else:
            model_name = None

        # Load stop sequences from config if not provided
        if stop is None:
            if api_mode == "chat":
                stop = config['vllm'].get('chat_stop_sequences', [])
            else:
                stop = config['vllm'].get('completion_stop_sequences', ['\n\n', '\ndef ', '\nclass '])
    except Exception as e:
        main_logger.warning(f"Could not load config: {e}, using defaults")
        if api_url is None:
            api_url = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")
        model_name = None
        if stop is None:
            stop = []

    inference = VLLMInference(api_url, api_mode=api_mode, model_name=model_name)
    main_logger.info(f"Initialized vLLM client at {api_url} with api_mode={api_mode}, model={model_name}")

    # Validate prompt strategy exists
    available_strategies = list_available_strategies(api_mode=api_mode)
    if prompt_strategy not in available_strategies:
        raise ValueError(f"Unknown prompt strategy: '{prompt_strategy}'. Available: {available_strategies}")

    # Create a prompt function using the loader
    def prompt_fn(problem: str):
        return load_prompt(prompt_strategy, problem, api_mode=api_mode)

    # Get postprocess function (auto-loads from strategy folder if postprocess_strategy is 'auto')
    postprocess_fn = get_postprocess_function(postprocess_strategy, prompt_strategy, api_mode)

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

    # Stop sequences are already loaded from config above

    # Generate completions in parallel
    print(f"\nGenerating completions for {len(problems)} problems...")
    print(f"Prompt strategy: {prompt_strategy}")
    print(f"Postprocess strategy: {postprocess_strategy}")
    print(f"Temperature: {temperature}")
    print(f"Max tokens: {max_tokens}")
    print(f"Top-p: {top_p}")
    print(f"Workers: {num_workers}")

    main_logger.info(f"Starting parallel inference on {len(problems)} problems with {num_workers} workers")
    main_logger.info(f"Configuration: strategy={prompt_strategy}, postprocess={postprocess_strategy}, temp={temperature}, max_tokens={max_tokens}, top_p={top_p}")

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
                max_tokens,
                top_p,
                stop,
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
    main_logger.info(f"Inference log: logs/inference/{Path(output_path).stem.replace('completions_', '')}.log")

    return results


if __name__ == "__main__":
    print("This script should be called via run_experiments.py with config.yml")
    print("Example: python scripts/run_experiments.py")
    print("\nFor direct usage, use run_experiments.py with --experiment flag:")
    print("  python scripts/run_experiments.py --experiment minimal_none")
    sys.exit(1)

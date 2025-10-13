#!/usr/bin/env python3
"""Re-process existing completions with updated post-processing."""

import json
import sys
from pathlib import Path

from datasets import load_dataset
from prompts.post_process_v5 import post_process_code_v5


def reprocess_completions(input_file: str, output_file: str):
    """
    Re-process completions with the updated post-processing function.

    Args:
        input_file: Path to original completions file
        output_file: Path to save reprocessed completions
    """
    print(f"Loading completions from {input_file}...")

    # Load original completions
    completions = []
    with open(input_file, 'r') as f:
        for line in f:
            completions.append(json.loads(line))

    print(f"Loaded {len(completions)} completions")

    # Load HumanEval dataset to get prompts
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")
    prompts_map = {item["task_id"]: item["prompt"] for item in dataset}

    print("Re-processing completions with updated post-processing...")
    reprocessed = []

    for completion in completions:
        task_id = completion["task_id"]
        raw_completion = completion.get("raw_completion", completion.get("completion"))
        prompt = prompts_map.get(task_id, completion.get("prompt", ""))

        # Re-apply post-processing
        cleaned_completion = post_process_code_v5(raw_completion, prompt)

        # Recreate full_code
        full_code = prompt + cleaned_completion

        reprocessed_item = {
            "task_id": task_id,
            "prompt": prompt,
            "completion": cleaned_completion,
            "raw_completion": raw_completion,
            "full_code": full_code,
        }

        reprocessed.append(reprocessed_item)

    # Save reprocessed completions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving reprocessed completions to {output_file}...")
    with open(output_path, 'w') as f:
        for item in reprocessed:
            f.write(json.dumps(item) + "\n")

    print(f"Done! Reprocessed {len(reprocessed)} completions")


if __name__ == "__main__":
    print("This is a utility script - import and call reprocess_completions() directly")
    print("\nExample usage in Python:")
    print('  from scripts.reprocess_completions import reprocess_completions')
    print('  reprocess_completions("results/completions.jsonl", "results/completions_reprocessed.jsonl")')
    sys.exit(1)

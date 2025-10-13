#!/usr/bin/env python3
"""
Compare multiple prompt strategies with post_process_v5.
Tests minimal_v2, minimal_v3, minimal_v4, minimal_v5 with the improved post-processor.
"""

import subprocess
import json
import sys
from pathlib import Path

# Prompt strategies to test
PROMPT_STRATEGIES = [
    "minimal_v2",  # Minimal prompt with '# Your code here\n'
    "minimal_v3",  # Ultra-minimal: problem.rstrip() + "\n"
    "minimal_v4",  # Minimal with indentation: problem.rstrip() + "\n    "
    "minimal_v5",  # Bare minimal: just problem.rstrip()
]

POSTPROCESS = "post_v5"  # Use the improved post-processor

def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*70}")
    print(f"{description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        return False

def main():
    print("="*70)
    print("PROMPT STRATEGY COMPARISON WITH POST_V5")
    print("="*70)
    print(f"\nTesting {len(PROMPT_STRATEGIES)} prompt strategies")
    print(f"Post-processor: {POSTPROCESS}")
    print(f"Strategies: {', '.join(PROMPT_STRATEGIES)}")
    print()

    # Create results directory if needed
    Path("results").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    results = {}

    for prompt_strategy in PROMPT_STRATEGIES:
        print(f"\n{'#'*70}")
        print(f"# Testing: {prompt_strategy} with {POSTPROCESS}")
        print(f"{'#'*70}")

        completions_file = f"results/completions_{prompt_strategy}_{POSTPROCESS}.jsonl"
        eval_file = f"results/eval_{prompt_strategy}_{POSTPROCESS}.json"
        log_file = f"logs/{prompt_strategy}_{POSTPROCESS}_all_cases.log"

        # Step 1: Generate completions
        inference_cmd = [
            "python", "scripts/inference.py",
            "--prompt-strategy", prompt_strategy,
            "--postprocess-strategy", POSTPROCESS,
            "--output", completions_file
        ]

        success = run_command(inference_cmd, f"Step 1: Generating completions for {prompt_strategy}")
        if not success:
            print(f"⚠️  Skipping evaluation for {prompt_strategy} due to inference failure")
            continue

        # Step 2: Run evaluation
        eval_cmd = [
            "python", "scripts/run_evaluation.py",
            "--completions", completions_file,
            "--output", eval_file,
            "--log", log_file
        ]

        success = run_command(eval_cmd, f"Step 2: Evaluating {prompt_strategy}")
        if not success:
            print(f"⚠️  Evaluation failed for {prompt_strategy}")
            continue

        # Load results
        try:
            with open(eval_file, 'r') as f:
                result_data = json.load(f)
                results[prompt_strategy] = result_data
                print(f"\n✅ {prompt_strategy}: {result_data.get('passed', 0)}/{result_data.get('total', 164)} passed ({result_data.get('pass_at_1', 0):.2%})")
        except Exception as e:
            print(f"❌ Error loading results for {prompt_strategy}: {e}")

    # Print summary
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    print(f"\n{'Strategy':<15} {'Passed':<10} {'Total':<10} {'Pass@1':<10}")
    print("-" * 70)

    sorted_results = sorted(results.items(), key=lambda x: x[1].get('passed', 0), reverse=True)

    for strategy, data in sorted_results:
        passed = data.get('passed', 0)
        total = data.get('total', 164)
        pass_at_1 = data.get('pass_at_1', 0)
        print(f"{strategy:<15} {passed:<10} {total:<10} {pass_at_1:<10.2%}")

    if sorted_results:
        best_strategy, best_data = sorted_results[0]
        print(f"\n🏆 Best Strategy: {best_strategy} with {best_data.get('passed', 0)}/{best_data.get('total', 164)} passed ({best_data.get('pass_at_1', 0):.2%})")

    print("\n" + "="*70)
    print("Comparison complete! Check logs/ directory for detailed failure analysis.")
    print("="*70)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Prompt tuning script to find the best strategy for pass@1 > 0.5
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

# Configurations to test
TUNING_CONFIGS = [
    # Test different prompt strategies with optimal temperature
    {"prompt_strategy": "infilling", "postprocess_strategy": "smart", "temperature": 0.2},
    {"prompt_strategy": "minimal", "postprocess_strategy": "smart", "temperature": 0.2},
    {"prompt_strategy": "instructional", "postprocess_strategy": "smart", "temperature": 0.1},

    # Test temperature variations with best prompt
    {"prompt_strategy": "infilling", "postprocess_strategy": "smart", "temperature": 0.1},
    {"prompt_strategy": "infilling", "postprocess_strategy": "smart", "temperature": 0.15},
    {"prompt_strategy": "infilling", "postprocess_strategy": "smart", "temperature": 0.3},

    # Test post-processing strategies
    {"prompt_strategy": "infilling", "postprocess_strategy": "basic", "temperature": 0.2},
]


def run_experiment(config: Dict, test_samples: int = 20) -> Dict:
    """
    Run a single experiment with given configuration.

    Args:
        config: Configuration dictionary
        test_samples: Number of samples to test on

    Returns:
        Results dictionary
    """
    prompt_strategy = config["prompt_strategy"]
    postprocess_strategy = config["postprocess_strategy"]
    temperature = config["temperature"]

    output_file = f"results/tune_{prompt_strategy}_{postprocess_strategy}_t{temperature}.jsonl"
    eval_file = f"results/eval_{prompt_strategy}_{postprocess_strategy}_t{temperature}.json"

    print(f"\n{'='*60}")
    print(f"Testing: {prompt_strategy} + {postprocess_strategy} (T={temperature})")
    print(f"{'='*60}")

    # Run inference
    inference_cmd = [
        "python", "scripts/inference.py",
        "--output", output_file,
        "--prompt-strategy", prompt_strategy,
        "--postprocess-strategy", postprocess_strategy,
        "--temperature", str(temperature),
        "--max-samples", str(test_samples),
    ]

    try:
        subprocess.run(inference_cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"Inference failed: {e}")
        return None

    # Run evaluation
    eval_cmd = [
        "python", "scripts/run_evaluation.py",
        "--completions", output_file,
        "--output", eval_file,
    ]

    try:
        subprocess.run(eval_cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed: {e}")
        return None

    # Load results
    with open(eval_file, 'r') as f:
        results = json.load(f)

    return {
        "config": config,
        "pass@1": results["pass@1"],
        "passed": results["passed"],
        "total": results["total"],
    }


def find_best_config(test_samples: int = 20, full_eval: bool = False):
    """
    Test multiple configurations and find the best one.

    Args:
        test_samples: Number of samples for quick testing
        full_eval: Run full evaluation on best config
    """
    results = []

    print("Starting prompt tuning...")
    print(f"Testing on {test_samples} samples per configuration\n")

    # Test each configuration
    for i, config in enumerate(TUNING_CONFIGS, 1):
        print(f"\nExperiment {i}/{len(TUNING_CONFIGS)}")
        result = run_experiment(config, test_samples)

        if result:
            results.append(result)
            print(f"Result: pass@1 = {result['pass@1']:.3f}")

    # Sort by pass@1
    results.sort(key=lambda x: x["pass@1"], reverse=True)

    # Print summary
    print("\n" + "="*60)
    print("TUNING RESULTS SUMMARY")
    print("="*60)

    for i, result in enumerate(results, 1):
        config = result["config"]
        print(f"{i}. pass@1={result['pass@1']:.3f} - "
              f"{config['prompt_strategy']} + {config['postprocess_strategy']} "
              f"(T={config['temperature']})")

    # Save results
    summary_file = "results/tuning_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSummary saved to: {summary_file}")

    # Get best config
    if results:
        best_result = results[0]
        best_config = best_result["config"]

        print("\n" + "="*60)
        print("BEST CONFIGURATION")
        print("="*60)
        print(f"Prompt strategy:     {best_config['prompt_strategy']}")
        print(f"Postprocess strategy: {best_config['postprocess_strategy']}")
        print(f"Temperature:         {best_config['temperature']}")
        print(f"pass@1 (test):       {best_result['pass@1']:.3f}")
        print("="*60)

        # Run full evaluation if requested
        if full_eval and best_result['pass@1'] >= 0.4:  # Only if promising
            print("\nRunning full evaluation on best config...")
            full_result = run_experiment(best_config, test_samples=164)

            if full_result:
                print("\n" + "="*60)
                print("FULL EVALUATION RESULT")
                print("="*60)
                print(f"pass@1: {full_result['pass@1']:.3f}")
                print(f"Passed: {full_result['passed']}/{full_result['total']}")

                if full_result['pass@1'] > 0.5:
                    print("\nSUCCESS! pass@1 > 0.5 achieved!")
                else:
                    print(f"\nNeed {(0.5 - full_result['pass@1'])*100:.1f}% improvement")

        # Generate recommended command
        print("\nRecommended inference command:")
        print(f"python scripts/inference.py \\")
        print(f"  --prompt-strategy {best_config['prompt_strategy']} \\")
        print(f"  --postprocess-strategy {best_config['postprocess_strategy']} \\")
        print(f"  --temperature {best_config['temperature']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tune prompts for HumanEval")
    parser.add_argument(
        "--test-samples",
        type=int,
        default=20,
        help="Number of samples for quick testing (default: 20)",
    )
    parser.add_argument(
        "--full-eval",
        action="store_true",
        help="Run full evaluation on best config",
    )

    args = parser.parse_args()

    find_best_config(
        test_samples=args.test_samples,
        full_eval=args.full_eval,
    )

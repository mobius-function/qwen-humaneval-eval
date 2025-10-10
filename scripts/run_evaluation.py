#!/usr/bin/env python3
"""Evaluation script for HumanEval completions."""

import json
import sys
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset
from tqdm import tqdm

from sandbox import check_correctness


def load_completions(completions_file: str) -> List[Dict]:
    """Load completions from JSONL file."""
    completions = []
    with open(completions_file, 'r') as f:
        for line in f:
            completions.append(json.loads(line))
    return completions


def load_humaneval_tests() -> Dict[str, Dict]:
    """Load HumanEval dataset with test cases."""
    dataset = load_dataset("openai_humaneval", split="test")
    tests = {}
    for item in dataset:
        tests[item["task_id"]] = {
            "entry_point": item["entry_point"],
            "test": item["test"],
            "prompt": item["prompt"],
        }
    return tests


def evaluate_completions(
    completions_file: str = "results/completions.jsonl",
    output_file: str = "results/evaluation_results.json",
    timeout: int = 3,
) -> Dict:
    """
    Evaluate generated completions against HumanEval test cases.

    Args:
        completions_file: Path to completions JSONL file
        output_file: Path to save evaluation results
        timeout: Timeout for each test execution in seconds

    Returns:
        Evaluation results dictionary
    """
    print("Loading completions...")
    completions = load_completions(completions_file)

    print("Loading HumanEval test cases...")
    tests = load_humaneval_tests()

    results = []
    passed_count = 0
    total_count = len(completions)

    print(f"\nEvaluating {total_count} completions...")

    for completion in tqdm(completions, desc="Evaluating"):
        task_id = completion["task_id"]
        full_code = completion["full_code"]

        if task_id not in tests:
            print(f"Warning: No test found for {task_id}")
            continue

        test_info = tests[task_id]
        test_code = test_info["test"]

        # Check correctness
        result = check_correctness(
            code=full_code,
            test_code=test_code,
            timeout=timeout,
        )

        passed = result["passed"]
        if passed:
            passed_count += 1

        result_entry = {
            "task_id": task_id,
            "passed": passed,
            "error": result.get("error"),
        }

        results.append(result_entry)

    # Calculate pass@1
    pass_at_1 = passed_count / total_count if total_count > 0 else 0

    evaluation_summary = {
        "total": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "pass@1": pass_at_1,
        "results": results,
    }

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Total problems:    {total_count}")
    print(f"Passed:           {passed_count}")
    print(f"Failed:           {total_count - passed_count}")
    print(f"pass@1:           {pass_at_1:.3f} ({pass_at_1*100:.1f}%)")
    print("=" * 50)

    if pass_at_1 > 0.5:
        print("✅ SUCCESS! pass@1 > 0.5 achieved!")
    else:
        print(f"❌ Target not met. Need to improve by {(0.5 - pass_at_1)*100:.1f}%")

    print(f"\nDetailed results saved to: {output_file}")

    return evaluation_summary


def analyze_failures(results_file: str = "results/evaluation_results.json"):
    """Analyze failed test cases to understand common issues."""
    with open(results_file, 'r') as f:
        data = json.load(f)

    failed_results = [r for r in data["results"] if not r["passed"]]

    if not failed_results:
        print("No failures to analyze!")
        return

    print(f"\nAnalyzing {len(failed_results)} failures...")

    # Categorize errors
    error_types = {}
    for result in failed_results:
        error = result.get("error", "Unknown error")
        error_type = error.split(":")[0] if error else "Unknown"

        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(result["task_id"])

    print("\nError Distribution:")
    print("-" * 50)
    for error_type, tasks in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
        print(f"{error_type:30} {len(tasks):3} tasks")

    print("\nSample failures:")
    for i, result in enumerate(failed_results[:5]):
        print(f"\n{i+1}. Task: {result['task_id']}")
        print(f"   Error: {result.get('error', 'Unknown')[:100]}...")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate HumanEval completions")
    parser.add_argument(
        "--completions",
        type=str,
        default="results/completions.jsonl",
        help="Path to completions file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/evaluation_results.json",
        help="Path to save evaluation results",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3,
        help="Timeout for each test in seconds",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze failures after evaluation",
    )

    args = parser.parse_args()

    # Run evaluation
    results = evaluate_completions(
        completions_file=args.completions,
        output_file=args.output,
        timeout=args.timeout,
    )

    # Analyze failures if requested
    if args.analyze:
        analyze_failures(args.output)

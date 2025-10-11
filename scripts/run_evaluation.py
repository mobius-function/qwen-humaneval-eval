#!/usr/bin/env python3
"""Evaluation script for HumanEval completions."""

import json
import sys
from pathlib import Path
from typing import List, Dict

from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from sandbox import check_correctness

console = Console()


def load_completions(completions_file: str) -> List[Dict]:
    """Load completions from JSONL file."""
    with open(completions_file, 'r') as f:
        return [json.loads(line) for line in f]


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
    console.print("Loading completions...", style="cyan")
    completions = load_completions(completions_file)

    console.print("Loading HumanEval test cases...", style="cyan")
    tests = load_humaneval_tests()

    results = []
    passed_count = 0
    total_count = len(completions)

    console.print(f"\nEvaluating {total_count} completions...", style="cyan bold")

    for completion in tqdm(completions, desc="Evaluating"):
        task_id = completion["task_id"]
        full_code = completion["full_code"]

        if task_id not in tests:
            console.print(f"Warning: No test found for {task_id}", style="yellow")
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

    # Print summary with Rich table
    console.print()
    table = Table(title="EVALUATION RESULTS", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", width=20)
    table.add_column("Value", style="green", width=20)

    table.add_row("Total problems", str(total_count))
    table.add_row("Passed", str(passed_count))
    table.add_row("Failed", str(total_count - passed_count))
    table.add_row("pass@1", f"{pass_at_1:.3f} ({pass_at_1*100:.1f}%)")

    console.print(table)

    if pass_at_1 > 0.5:
        console.print("\nSUCCESS! pass@1 > 0.5 achieved!", style="bold green")
    else:
        console.print(f"\nTarget not met. Need to improve by {(0.5 - pass_at_1)*100:.1f}%", style="bold red")

    console.print(f"\nDetailed results saved to: {output_file}", style="dim")

    return evaluation_summary


def analyze_failures(results_file: str = "results/evaluation_results.json"):
    """Analyze failed test cases to understand common issues."""
    with open(results_file, 'r') as f:
        data = json.load(f)

    failed_results = [r for r in data["results"] if not r["passed"]]

    if not failed_results:
        console.print("No failures to analyze!", style="green")
        return

    console.print(f"\n[cyan]Analyzing {len(failed_results)} failures...[/cyan]")

    # Categorize errors
    error_types = {}
    for result in failed_results:
        error = result.get("error", "Unknown error")
        error_type = error.split(":")[0] if error else "Unknown"

        if error_type not in error_types:
            error_types[error_type] = []
        error_types[error_type].append(result["task_id"])

    # Error distribution table
    table = Table(title="Error Distribution", show_header=True, header_style="bold magenta")
    table.add_column("Error Type", style="yellow", width=30)
    table.add_column("Count", style="red", width=10, justify="right")

    for error_type, tasks in sorted(error_types.items(), key=lambda x: len(x[1]), reverse=True):
        table.add_row(error_type, str(len(tasks)))

    console.print(table)

    # Sample failures
    console.print("\n[bold]Sample failures:[/bold]")
    for i, result in enumerate(failed_results[:5]):
        console.print(f"\n[cyan]{i+1}. Task:[/cyan] {result['task_id']}")
        console.print(f"   [red]Error:[/red] {result.get('error', 'Unknown')[:100]}...", style="dim")


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

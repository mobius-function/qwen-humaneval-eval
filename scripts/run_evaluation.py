#!/usr/bin/env python3
"""Evaluation script for HumanEval completions."""

import json
import multiprocessing
import os
import signal
import sys
from pathlib import Path
from typing import List, Dict, Tuple

import yaml
from datasets import load_dataset
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

console = Console()


class TimeoutException(Exception):
    """Raised when code execution times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Execution timed out")


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


def evaluate_single_completion(args: Tuple[Dict, Dict[str, Dict], int]) -> Dict:
    """
    Evaluate a single completion (worker function for multiprocessing).

    Args:
        args: Tuple of (completion, tests, timeout)

    Returns:
        Result dictionary with task_id, passed status, and error if any
    """
    completion, tests, timeout = args
    task_id = completion["task_id"]
    full_code = completion["full_code"]

    if task_id not in tests:
        return {
            "task_id": task_id,
            "passed": False,
            "error": "No test found for task",
            "completion": completion.get("completion"),
            "raw_completion": completion.get("raw_completion", completion.get("completion")),
        }

    test_info = tests[task_id]
    test_code = test_info["test"]
    entry_point = test_info["entry_point"]

    # Execute test directly in this process with timeout protection
    # (we're already in an isolated Pool worker, so this is safe)
    try:
        # Set up timeout using signal (Unix/Linux only)
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)

        try:
            namespace = {}
            # Execute the generated code
            exec(full_code, namespace)
            # Execute the test code (defines check function)
            exec(test_code, namespace)
            # Actually run the tests by calling check with the entry point
            namespace['check'](namespace[entry_point])

            # Cancel the alarm if test completes successfully
            signal.alarm(0)

            return {
                "task_id": task_id,
                "passed": True,
                "error": None,
                "completion": completion.get("completion"),
                "raw_completion": completion.get("raw_completion", completion.get("completion")),
            }
        finally:
            # Always cancel the alarm
            signal.alarm(0)

    except TimeoutException:
        return {
            "task_id": task_id,
            "passed": False,
            "error": f"Timeout after {timeout} seconds",
            "raw_completion": completion.get("raw_completion", completion.get("completion")),
            "completion": completion.get("completion"),
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "passed": False,
            "error": str(e),
            "raw_completion": completion.get("raw_completion", completion.get("completion")),
            "completion": completion.get("completion"),
        }


def evaluate_completions(
    completions_file: str = "results/completions.jsonl",
    output_file: str = "results/evaluation_results.json",
    timeout: int = 3,
    num_workers: int = None,
) -> Dict:
    """
    Evaluate generated completions against HumanEval test cases.

    Args:
        completions_file: Path to completions JSONL file
        output_file: Path to save evaluation results
        timeout: Timeout for each test execution in seconds
        num_workers: Number of parallel workers (default: CPU count)

    Returns:
        Evaluation results dictionary
    """
    console.print("Loading completions...", style="cyan")
    completions = load_completions(completions_file)

    console.print("Loading HumanEval test cases...", style="cyan")
    tests = load_humaneval_tests()

    total_count = len(completions)

    # Determine number of workers
    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, total_count)

    console.print(f"\nEvaluating {total_count} completions using {num_workers} workers...", style="cyan bold")

    # Prepare arguments for parallel processing
    eval_args = [(completion, tests, timeout) for completion in completions]

    # Run evaluations in parallel
    # Note: Using a custom context to allow nested multiprocessing (sandbox.py spawns subprocesses)
    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(evaluate_single_completion, eval_args),
            total=total_count,
            desc="Evaluating"
        ))

    # Count passed tests
    passed_count = sum(1 for r in results if r["passed"])

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

    # Log failed cases with before/after post-processing
    failed_results = [r for r in results if not r["passed"]]
    if failed_results:
        # Create failure log
        log_dir = Path("logs/evaluation")
        log_dir.mkdir(parents=True, exist_ok=True)
        exp_name = output_path.stem.replace("evaluation_", "")

        # Save comprehensive log with ALL test cases
        all_cases_log = log_dir / f"{exp_name}_all_cases.log"

        with open(all_cases_log, 'w') as f:
            f.write(f"ALL TEST CASES - {passed_count} passed, {len(failed_results)} failed out of {total_count}\n")
            f.write("="*80 + "\n\n")

            for idx, result in enumerate(results, 1):
                task_id = result['task_id']
                passed = result['passed']
                status = "✓ PASSED" if passed else "✗ FAILED"

                f.write(f"[{idx}/{total_count}] {task_id} - {status}\n")
                f.write("="*80 + "\n")

                # Get the original problem
                if task_id in tests:
                    test_info = tests[task_id]

                    f.write("INPUT DOCSTRING:\n")
                    f.write("-"*80 + "\n")
                    f.write(test_info['prompt'] + "\n")
                    f.write("\n")

                    f.write("GENERATED CODE (RAW MODEL OUTPUT):\n")
                    f.write("-"*80 + "\n")
                    raw_completion = result.get('raw_completion', 'N/A')
                    f.write(raw_completion + "\n")
                    f.write("\n")

                    f.write("POST-PROCESSED CODE:\n")
                    f.write("-"*80 + "\n")
                    processed_completion = result.get('completion', 'N/A')
                    f.write(processed_completion + "\n")
                    f.write("\n")

                    if not passed:
                        f.write("ERROR:\n")
                        f.write("-"*80 + "\n")
                        f.write(result.get('error', 'No error message') + "\n")
                        f.write("\n")
                else:
                    f.write(f"WARNING: Could not find test info for {task_id}\n")

                f.write("\n" + "="*80 + "\n\n")

        console.print(f"[dim]All cases log saved to: {all_cases_log}[/dim]")

        # Also save failures-only log for quick reference
        failures_log = log_dir / f"{exp_name}_failures.log"

        with open(failures_log, 'w') as f:
            f.write(f"FAILED CASES - {len(failed_results)} out of {total_count} failures\n")
            f.write("="*80 + "\n\n")

            for idx, result in enumerate(failed_results, 1):
                task_id = result['task_id']
                f.write(f"[{idx}/{len(failed_results)}] Task: {task_id}\n")
                f.write("="*80 + "\n")
                f.write(f"Error: {result.get('error', 'No error message')}\n")
                f.write("\n")

                if task_id in tests:
                    test_info = tests[task_id]
                    f.write("INPUT DOCSTRING:\n")
                    f.write("-"*80 + "\n")
                    f.write(test_info['prompt'] + "\n")
                    f.write("\n")

                    f.write("GENERATED CODE (RAW MODEL OUTPUT):\n")
                    f.write("-"*80 + "\n")
                    f.write(result.get('raw_completion', 'N/A') + "\n")
                    f.write("\n")

                    f.write("POST-PROCESSED CODE:\n")
                    f.write("-"*80 + "\n")
                    f.write(result.get('completion', 'N/A') + "\n")
                    f.write("\n")
                else:
                    f.write(f"WARNING: Could not find test info for {task_id}\n")

                f.write("\n" + "="*80 + "\n\n")

        console.print(f"[dim]Failures log saved to: {failures_log}[/dim]")

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
    # Load config from YAML file
    config_path = "prompt_tuning.yml"

    if not Path(config_path).exists():
        console.print(f"[red]Error: Config file not found: {config_path}[/red]")
        sys.exit(1)

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Get evaluation settings from config
    eval_config = config.get("evaluation", {})

    # Get completions path from inference config
    completions_file = config.get("inference", {}).get(
        "output_path", "results/completions.jsonl"
    )

    # Derive evaluation results path from completions path
    completions_stem = Path(completions_file).stem
    output_file = f"results/evaluation_{completions_stem.replace('completions_', '')}.json"

    # Get other evaluation parameters
    timeout = eval_config.get("timeout", 3)
    num_workers = eval_config.get("num_workers", None)

    # Run evaluation
    results = evaluate_completions(
        completions_file=completions_file,
        output_file=output_file,
        timeout=timeout,
        num_workers=num_workers,
    )

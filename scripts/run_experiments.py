#!/usr/bin/env python3
"""Run multiple experiments based on config.yml configuration."""

import json
import logging
import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from inference import run_inference
from run_evaluation import evaluate_completions

console = Console()


def load_config(config_path: str = "config.yml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_experiment_name(exp_config: Dict) -> str:
    """
    Auto-generate experiment name from configuration.

    Format: {prompt_strategy}_{api_mode}_{postprocess_strategy}

    Args:
        exp_config: Experiment configuration dictionary

    Returns:
        Generated experiment name
    """
    return f"{exp_config['prompt_strategy']}_{exp_config['api_mode']}_{exp_config['postprocess_strategy']}"


def setup_experiment_logging(exp_name: str) -> logging.Logger:
    """
    Setup logging for experiment runner.

    Args:
        exp_name: Name of the experiment or 'all' for combined run

    Returns:
        Logger instance
    """
    log_dir = Path("logs/orchestration")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"experiment.{exp_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler - always use timestamp for orchestration logs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}.log" if exp_name == "all" else log_dir / f"{exp_name}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def run_single_experiment(exp_config: Dict, global_config: Dict, exp_logger: logging.Logger) -> Dict:
    """
    Run a single experiment: inference + evaluation.

    Args:
        exp_config: Experiment-specific configuration
        global_config: Global configuration (vllm, evaluation, dataset)
        exp_logger: Logger for experiment tracking

    Returns:
        Dictionary with experiment results
    """
    # Auto-generate experiment name
    exp_name = generate_experiment_name(exp_config)
    results_dir = Path(global_config['evaluation']['results_dir'])
    results_dir.mkdir(exist_ok=True)

    console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
    console.print(f"[bold cyan]Experiment: {exp_name}[/bold cyan]")
    console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")

    exp_logger.info(f"Starting experiment: {exp_name}")
    exp_logger.info(f"Configuration: prompt={exp_config['prompt_strategy']}, api_mode={exp_config['api_mode']}, postprocess={exp_config['postprocess_strategy']}, temp={exp_config['temperature']}")

    # Auto-generate file paths
    completions_file = str(results_dir / f"completions_{exp_name}.jsonl")
    results_file = str(results_dir / f"evaluation_{exp_name}.json")

    # Step 1: Inference
    console.print("[bold yellow]Step 1: Running inference...[/bold yellow]")
    exp_logger.info("Starting inference step")
    try:
        run_inference(
            output_path=completions_file,
            api_url=None,  # Auto-select based on api_mode from config
            api_mode=exp_config['api_mode'],
            temperature=exp_config['temperature'],
            max_samples=global_config['dataset']['max_samples'],
            prompt_strategy=exp_config['prompt_strategy'],
            postprocess_strategy=exp_config['postprocess_strategy'],
            num_workers=global_config['inference']['num_workers'],
            max_tokens=global_config['vllm']['max_tokens'],
            top_p=global_config['vllm']['top_p'],
            # stop sequences are auto-loaded from config based on api_mode
        )
        exp_logger.info(f"Inference completed successfully: {completions_file}")
    except Exception as e:
        console.print(f"[bold red]Inference failed: {e}[/bold red]")
        exp_logger.error(f"Inference failed: {e}")
        return {
            'name': exp_name,
            'status': 'failed',
            'error': str(e),
            'prompt_strategy': exp_config['prompt_strategy'],
            'api_mode': exp_config['api_mode'],
            'postprocess_strategy': exp_config['postprocess_strategy'],
            'temperature': exp_config['temperature'],
        }

    # Step 2: Evaluation
    console.print(f"\n[bold yellow]Step 2: Running evaluation...[/bold yellow]")
    exp_logger.info("Starting evaluation step")
    try:
        eval_results = evaluate_completions(
            completions_file=completions_file,
            output_file=results_file,
            timeout=global_config['evaluation']['timeout'],
            num_workers=global_config['evaluation']['num_workers'],
        )

        exp_logger.info(f"Evaluation completed: pass@1={eval_results['pass@1']:.3f}, passed={eval_results['passed']}/{eval_results['total']}")
        exp_logger.info(f"Results saved to: {results_file}")

        return {
            'name': exp_name,
            'status': 'success',
            'prompt_strategy': exp_config['prompt_strategy'],
            'api_mode': exp_config['api_mode'],
            'postprocess_strategy': exp_config['postprocess_strategy'],
            'temperature': exp_config['temperature'],
            'pass@1': eval_results['pass@1'],
            'passed': eval_results['passed'],
            'failed': eval_results['failed'],
            'total': eval_results['total'],
            'results_file': results_file,
        }
    except Exception as e:
        console.print(f"[bold red]Evaluation failed: {e}[/bold red]")
        exp_logger.error(f"Evaluation failed: {e}")
        return {
            'name': exp_name,
            'status': 'failed',
            'error': str(e),
            'prompt_strategy': exp_config['prompt_strategy'],
            'api_mode': exp_config['api_mode'],
            'postprocess_strategy': exp_config['postprocess_strategy'],
            'temperature': exp_config['temperature'],
        }


def print_summary(results: List[Dict]):
    """Print summary table of all experiment results."""
    console.print("\n" + "="*80)
    console.print("[bold cyan]EXPERIMENT SUMMARY[/bold cyan]")
    console.print("="*80 + "\n")

    # Create summary table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Experiment", style="cyan", width=25)
    table.add_column("Prompt", style="yellow", width=15)
    table.add_column("Postprocess", style="yellow", width=12)
    table.add_column("Temp", justify="right", width=6)
    table.add_column("pass@1", justify="right", width=10)
    table.add_column("Passed/Total", justify="right", width=12)
    table.add_column("Status", width=10)

    # Sort by pass@1 score (descending)
    sorted_results = sorted(
        results,
        key=lambda x: x.get('pass@1', 0) if x['status'] == 'success' else -1,
        reverse=True
    )

    for result in sorted_results:
        if result['status'] == 'success':
            pass_rate = f"{result['pass@1']:.3f}"
            passed_total = f"{result['passed']}/{result['total']}"
            status_style = "green" if result['pass@1'] > 0.5 else "yellow"
            table.add_row(
                result['name'],
                result['prompt_strategy'],
                result['postprocess_strategy'],
                f"{result['temperature']:.1f}",
                f"[{status_style}]{pass_rate}[/{status_style}]",
                passed_total,
                f"[green]Success[/green]"
            )
        else:
            # Failed experiments still show their configuration
            table.add_row(
                result['name'],
                result.get('prompt_strategy', '-'),
                result.get('postprocess_strategy', '-'),
                f"{result.get('temperature', 0):.1f}" if result.get('temperature') else "-",
                "-",
                "-",
                f"[red]Failed[/red]"
            )

    console.print(table)

    # Print best result
    successful = [r for r in sorted_results if r['status'] == 'success']
    if successful:
        best = successful[0]
        console.print(f"\n[bold green]Best Result:[/bold green] {best['name']}")
        console.print(f"  Strategy: {best['prompt_strategy']} + {best['postprocess_strategy']}")
        console.print(f"  pass@1: {best['pass@1']:.3f} ({best['pass@1']*100:.1f}%)")
        console.print(f"  Passed: {best['passed']}/{best['total']}")


def main():
    """Main entry point for running experiments."""
    import argparse

    parser = argparse.ArgumentParser(description="Run HumanEval experiments from config")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yml",
        help="Path to config file (default: config.yml)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Run specific experiment by name (default: run all enabled)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all experiments and exit",
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        console.print(f"[bold red]Error: Config file not found: {args.config}[/bold red]")
        sys.exit(1)
    except yaml.YAMLError as e:
        console.print(f"[bold red]Error parsing config file: {e}[/bold red]")
        sys.exit(1)

    # List experiments if requested
    if args.list:
        console.print("\n[bold]Available Experiments:[/bold]\n")
        for exp in config['experiments']:
            exp_name = generate_experiment_name(exp)
            status = "✓ enabled" if exp['enabled'] else "✗ disabled"
            console.print(f"  {exp_name:<40} [{status}]")
            console.print(f"    Prompt: {exp['prompt_strategy']}, API: {exp['api_mode']}, Post: {exp['postprocess_strategy']}, Temp: {exp['temperature']}")
        return

    # Filter experiments
    experiments = config['experiments']
    if args.experiment:
        experiments = [e for e in experiments if generate_experiment_name(e) == args.experiment]
        if not experiments:
            console.print(f"[bold red]Error: Experiment '{args.experiment}' not found[/bold red]")
            console.print(f"[yellow]Use --list to see available experiments[/yellow]")
            sys.exit(1)
    else:
        experiments = [e for e in experiments if e['enabled']]

    if not experiments:
        console.print("[bold yellow]No experiments enabled. Edit config.yml to enable experiments.[/bold yellow]")
        return

    # Setup experiment logging
    exp_logger = setup_experiment_logging("all" if len(experiments) > 1 else generate_experiment_name(experiments[0]))
    exp_logger.info(f"Starting experiment run with {len(experiments)} experiments")

    # Run experiments
    console.print(f"\n[bold]Running {len(experiments)} experiment(s)...[/bold]\n")
    results = []

    for exp in experiments:
        result = run_single_experiment(exp, config, exp_logger)
        results.append(result)

    # Print summary
    print_summary(results)

    # Save combined results
    results_dir = Path(config['evaluation']['results_dir'])
    summary_file = results_dir / "experiments_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)

    exp_logger.info(f"All experiments completed. Summary saved to: {summary_file}")
    console.print(f"\n[dim]Summary saved to: {summary_file}[/dim]")
    console.print(f"[dim]Logs saved to: logs/orchestration/, logs/inference/, logs/evaluation/[/dim]")


if __name__ == "__main__":
    main()

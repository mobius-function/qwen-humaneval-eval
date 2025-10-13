#!/usr/bin/env python3
"""
Iterative Prompt Optimizer - Simplified version using existing infrastructure.

Reuses existing run_inference() and evaluate_completions() functions.
Only manages prompt evolution and comparison logic.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List
from datasets import load_dataset

# Use existing inference and evaluation functions
from scripts.inference import run_inference
from scripts.run_evaluation import evaluate_completions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/iterative_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_dynamic_prompt_strategy(instructions: str):
    """
    Create a prompt function with given instructions.

    Args:
        instructions: Additional prompting instructions to append

    Returns:
        Prompt function compatible with PROMPT_STRATEGIES
    """
    def dynamic_prompt(problem: str) -> str:
        if not instructions:
            return problem  # Minimal prompt

        return f"""{problem}

# Important Guidelines:
{instructions}
"""

    return dynamic_prompt


def analyze_failures(results: List[Dict]) -> Dict:
    """
    Analyze evaluation results to identify failure patterns.

    Args:
        results: List of evaluation results

    Returns:
        Analysis dictionary with error patterns
    """
    failures = [r for r in results if not r['passed']]
    successes = [r for r in results if r['passed']]

    error_categories = {
        'syntax_errors': 0,
        'assertion_errors': 0,
        'type_errors': 0,
        'index_errors': 0,
        'timeout_errors': 0,
    }

    for failure in failures:
        error_msg = failure.get('error', '').lower()
        if 'syntaxerror' in error_msg or 'indentation' in error_msg:
            error_categories['syntax_errors'] += 1
        elif 'assertionerror' in error_msg:
            error_categories['assertion_errors'] += 1
        elif 'typeerror' in error_msg:
            error_categories['type_errors'] += 1
        elif 'indexerror' in error_msg or 'keyerror' in error_msg:
            error_categories['index_errors'] += 1
        elif 'timeout' in error_msg:
            error_categories['timeout_errors'] += 1

    return {
        'total': len(results),
        'failures': len(failures),
        'successes': len(successes),
        'error_categories': error_categories,
    }


def generate_refined_instructions(current_instructions: str, analysis: Dict) -> str:
    """
    Generate refined prompt instructions based on failure analysis.

    Args:
        current_instructions: Current prompt instructions
        analysis: Failure analysis dictionary

    Returns:
        Updated instructions
    """
    error_cats = analysis['error_categories']
    failures = analysis['failures']
    total = analysis['total']

    if failures == 0:
        return current_instructions

    refinements = []

    # Add refinements based on error patterns
    if error_cats['syntax_errors'] > 0:
        refinements.append("- Write complete, syntactically correct Python code")

    if error_cats['assertion_errors'] > failures * 0.3:
        refinements.append("- CRITICAL: Examples in docstring are test cases - your code MUST pass them exactly")
        refinements.append("- Trace through each example to verify your logic")

    if error_cats['type_errors'] > 0:
        refinements.append("- Pay attention to return types (list vs tuple, int vs float)")

    if error_cats['index_errors'] > 0:
        refinements.append("- Handle edge cases: empty inputs, boundary indices")

    if failures > total * 0.6:
        refinements.append("- Read docstring carefully - requirements are precisely specified")
        refinements.append("- Think step-by-step about the algorithm")

    # Remove duplicates
    if current_instructions:
        existing = set(current_instructions.split('\n'))
        refinements = [r for r in refinements if r not in existing]

    if not refinements:
        return current_instructions

    # Combine
    if current_instructions:
        return current_instructions + "\n" + "\n".join(refinements)
    else:
        return "\n".join(refinements)


def run_with_prompt(prompt_instructions: str, iteration: int) -> tuple:
    """
    Run inference and evaluation with given prompt instructions.

    Args:
        prompt_instructions: Prompt instructions to use
        iteration: Iteration number

    Returns:
        (completions_file, results, accuracy)
    """
    # Register dynamic prompt strategy
    from prompts.advanced_prompts import PROMPT_STRATEGIES
    PROMPT_STRATEGIES['dynamic'] = create_dynamic_prompt_strategy(prompt_instructions)

    # Output files
    completions_file = f"results/iter_{iteration}_completions.jsonl"
    results_file = f"results/iter_{iteration}_results.json"
    failures_log = f"logs/iter_{iteration}_failures.log"

    # Run inference using existing function
    logger.info(f"Running inference (iteration {iteration})...")
    run_inference(
        output_path=completions_file,
        temperature=0.2,
        prompt_strategy='dynamic',
        postprocess_strategy='none',
        num_workers=16,
    )

    # Evaluate using existing function (pass file path, not list)
    logger.info(f"Evaluating (iteration {iteration})...")
    eval_summary = evaluate_completions(
        completions_file=completions_file,
        output_file=results_file
    )

    # Extract results from summary
    results = eval_summary['results']
    accuracy = eval_summary['pass@1']

    # Save failure log
    failures = [r for r in results if not r['passed']]
    if failures:
        with open(failures_log, 'w') as f:
            f.write(f"ITERATION {iteration} - FAILURES ({len(failures)}/{len(results)})\n")
            f.write("="*80 + "\n\n")
            for fail in failures:
                f.write(f"Task: {fail['task_id']}\n")
                f.write(f"Error: {fail.get('error', 'Unknown')}\n")
                f.write("-"*80 + "\n\n")

    return completions_file, results, accuracy


def optimize():
    """Run iterative prompt optimization."""
    logger.info("="*80)
    logger.info("Starting Iterative Prompt Optimization")
    logger.info("="*80)

    max_iterations = 10

    # Initialize with minimal prompt
    current_instructions = ""
    current_results = None
    current_accuracy = 0.0

    best_instructions = ""
    best_accuracy = 0.0

    # Evaluate initial minimal prompt
    logger.info("\n[Iteration 0] Evaluating minimal prompt...")
    _, current_results, current_accuracy = run_with_prompt(current_instructions, 0)

    best_accuracy = current_accuracy
    best_instructions = current_instructions

    logger.info(f"Initial accuracy: {current_accuracy:.3f} ({int(current_accuracy*164)}/164)")

    iteration_history = [{
        'iteration': 0,
        'instructions': current_instructions,
        'accuracy': current_accuracy,
        'passed': int(current_accuracy * 164),
        'improved': True
    }]

    # Iterate 10 times
    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[Iteration {iteration}]")
        logger.info(f"{'='*80}")

        # ALWAYS analyze current_results (the current baseline)
        analysis = analyze_failures(current_results)
        logger.info(f"Current failures: {analysis['failures']}/{analysis['total']}")
        logger.info(f"Error categories: {analysis['error_categories']}")

        if analysis['failures'] == 0:
            logger.info("Perfect score! No refinement needed.")
            break

        # Generate refined instructions based on current_results
        new_instructions = generate_refined_instructions(current_instructions, analysis)

        if new_instructions == current_instructions:
            logger.info("No new refinements generated. Stopping.")
            break

        logger.info(f"\nRefined instructions:\n{new_instructions}\n")

        # Test refined prompt on full dataset
        logger.info("Testing refined prompt on full dataset...")
        _, new_results, new_accuracy = run_with_prompt(new_instructions, iteration)

        logger.info(f"New accuracy: {new_accuracy:.3f} ({int(new_accuracy*164)}/164)")
        logger.info(f"Current baseline: {current_accuracy:.3f} ({int(current_accuracy*164)}/164)")

        # Compare and update
        improved = False
        if new_accuracy > current_accuracy:
            improvement = new_accuracy - current_accuracy
            logger.info(f"✓ IMPROVEMENT: +{improvement:.3f}")
            logger.info(f"Updating current baseline to new prompt")

            # Update current baseline
            current_instructions = new_instructions
            current_results = new_results
            current_accuracy = new_accuracy
            improved = True

            # Update best if this is the best so far
            if new_accuracy > best_accuracy:
                best_instructions = new_instructions
                best_accuracy = new_accuracy
        else:
            logger.info(f"✗ No improvement. Keeping current baseline.")
            logger.info(f"Will re-analyze current baseline in next iteration")

        iteration_history.append({
            'iteration': iteration,
            'instructions': new_instructions,
            'accuracy': new_accuracy,
            'passed': int(new_accuracy * 164),
            'improved': improved
        })

    # Final results
    logger.info("\n" + "="*80)
    logger.info("OPTIMIZATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Final baseline accuracy: {current_accuracy:.3f} ({int(current_accuracy*164)}/164)")
    logger.info(f"Best accuracy achieved: {best_accuracy:.3f} ({int(best_accuracy*164)}/164)")
    logger.info(f"\nBest prompt instructions:\n{best_instructions}")

    # Save results
    output = {
        'final_baseline_accuracy': current_accuracy,
        'best_accuracy': best_accuracy,
        'best_passed': int(best_accuracy * 164),
        'best_instructions': best_instructions,
        'total_iterations': len(iteration_history) - 1,
        'improvements': sum(1 for h in iteration_history if h.get('improved', False)),
        'history': iteration_history
    }

    Path('results').mkdir(exist_ok=True)
    with open('results/iterative_optimization_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    with open('results/best_prompt_instructions.txt', 'w') as f:
        f.write(best_instructions)

    logger.info(f"\nTotal iterations: {len(iteration_history) - 1}")
    logger.info(f"Successful improvements: {output['improvements']}")
    logger.info("\nResults saved to results/")


if __name__ == '__main__':
    Path('logs').mkdir(exist_ok=True)
    optimize()

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
        # Base prompt with role assignment
        base_prompt = f"""You are an expert Python programmer. Your task is to implement the function below based on its docstring.

{problem}

IMPORTANT:
- The examples in the docstring are test cases - verify your implementation matches them
- Write general code that works on ANY valid input, not just the examples
- Do not hardcode outputs - implement the actual logic described in the docstring"""

        if not instructions:
            return base_prompt  # Base prompt with role

        return f"""{base_prompt}

# Additional Guidelines:
{instructions}
"""

    return dynamic_prompt


def analyze_failures(results: List[Dict]) -> Dict:
    """
    Analyze evaluation results to identify failure patterns.

    Args:
        results: List of evaluation results

    Returns:
        Analysis dictionary with error patterns and detailed results
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
        'detailed_results': results,  # Include full results for detailed analysis
    }


def generate_refined_instructions(
    current_instructions: str,
    analysis: Dict,
    iteration_history: List[Dict],
    client
) -> str:
    """
    Generate refined prompt instructions using Qwen as the evaluator.

    Uses the LLM to analyze:
    1. Previous prompt attempts and their accuracy
    2. Current failure patterns

    Args:
        current_instructions: Current prompt instructions
        analysis: Failure analysis dictionary
        iteration_history: List of previous iterations with prompts and accuracy
        client: VLLMInference client to call Qwen

    Returns:
        Updated instructions
    """
    error_cats = analysis['error_categories']
    failures = analysis['failures']
    total = analysis['total']

    if failures == 0:
        return current_instructions

    # Build context for Qwen
    history_summary = "Previous prompt attempts:\n"
    for hist in iteration_history:
        iter_num = hist['iteration']
        acc = hist['accuracy']
        passed = hist['passed']
        instructions = hist['instructions'] if hist['instructions'] else "[Minimal - no instructions]"
        history_summary += f"\nIteration {iter_num}: {acc:.3f} ({passed}/164 passed)\n"
        history_summary += f"Instructions:\n{instructions}\n"

    # Get actual failure details from current_results (passed via analysis)
    failures_list = [r for r in analysis.get('detailed_results', []) if not r['passed']]

    # Load the HumanEval dataset to get original prompts
    from datasets import load_dataset
    humaneval = load_dataset("openai_humaneval", split="test")
    humaneval_dict = {item['task_id']: item for item in humaneval}

    # Sample some failures for context (show first 5 to avoid huge prompts, with full details)
    failure_examples = ""
    for i, fail in enumerate(failures_list[:5]):
        task_id = fail['task_id']
        error = fail.get('error', 'Unknown error')

        # Get the original problem prompt
        original_problem = humaneval_dict.get(task_id, {}).get('prompt', 'N/A')
        if len(original_problem) > 400:
            original_problem = original_problem[:400] + "\n... (truncated)"

        # Get the generated code if available
        generated_code = fail.get('completion', 'N/A')
        if len(generated_code) > 300:
            generated_code = generated_code[:300] + "\n... (truncated)"

        failure_examples += f"""
Failure {i+1}: {task_id}

Original Problem:
{original_problem}

Generated Code:
{generated_code}

Error:
{error[:400]}

---"""

    if len(failures_list) > 5:
        failure_examples += f"\n\n... and {len(failures_list) - 5} more failures"

    # Current failure analysis
    failure_summary = f"""
Current baseline failures: {failures}/{total}

Sample of actual failures (first 10):
{failure_examples}
"""

    # Ask Qwen to generate refined instructions with a clearer, simpler prompt
    meta_prompt = f"""Analyze these code generation failures and suggest prompt improvements.

{history_summary}

{failure_summary}

Task: Generate NEW prompt instructions to reduce these failures. Look at what errors are happening and suggest specific guidance.

Example good instructions:
- Study the examples in docstrings carefully - they are test cases
- Handle empty lists and edge cases explicitly
- Return the exact type specified (list not tuple, int not float)
- Verify your logic against each docstring example

Now write 3-5 NEW instructions based on the failures above. Start each with "-":

-"""

    # Call Qwen to generate refinement
    try:
        response = client.generate_completion(
            prompt=meta_prompt,
            max_tokens=300,
            temperature=0.8,  # Higher temp for creativity
            stop=["\n\n", "Task:", "Example:"]
        )

        # Qwen will continue from the "-" we started
        new_instructions = "-" + response.strip()

        # Clean up any repeated prompts or garbage
        lines = new_instructions.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            if line.startswith('-') and len(line) > 5:  # Valid instruction
                clean_lines.append(line)
            elif not line.startswith('-') and clean_lines:  # Continuation of previous
                clean_lines[-1] += " " + line

        new_instructions = '\n'.join(clean_lines)

        # If Qwen didn't generate anything useful, fall back to rule-based
        if not new_instructions or len(new_instructions) < 20 or len(clean_lines) == 0:
            logger.warning(f"Qwen output not useful: '{new_instructions[:100]}', falling back")
            return fallback_refinement(current_instructions, analysis)

        logger.info(f"Qwen generated {len(clean_lines)} instruction lines")
        logger.info(f"Generated instructions:\n{new_instructions}")
        return new_instructions

    except Exception as e:
        logger.error(f"Error calling Qwen for refinement: {e}")
        return fallback_refinement(current_instructions, analysis)


def fallback_refinement(current_instructions: str, analysis: Dict) -> str:
    """Fallback rule-based refinement if Qwen fails."""
    error_cats = analysis['error_categories']
    failures = analysis['failures']
    total = analysis['total']

    refinements = []

    if error_cats['syntax_errors'] > 0:
        refinements.append("- Write complete, syntactically correct Python code")

    if error_cats['assertion_errors'] > failures * 0.3:
        refinements.append("- CRITICAL: Examples in docstring are test cases - your code MUST pass them exactly")

    if error_cats['type_errors'] > 0:
        refinements.append("- Pay attention to return types (list vs tuple, int vs float)")

    if error_cats['index_errors'] > 0:
        refinements.append("- Handle edge cases: empty inputs, boundary indices")

    if failures > total * 0.6:
        refinements.append("- Read docstring carefully - requirements are precisely specified")

    if current_instructions:
        existing = set(current_instructions.split('\n'))
        refinements = [r for r in refinements if r not in existing]

    if not refinements:
        return current_instructions

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

    # Initialize vLLM client for meta-prompting
    from scripts.inference import VLLMInference
    client = VLLMInference(api_url="http://localhost:8000/v1")
    logger.info("Initialized Qwen client for prompt optimization")

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

        # Generate refined instructions using Qwen as evaluator
        logger.info("Calling Qwen to generate refined prompt based on history and failures...")
        new_instructions = generate_refined_instructions(
            current_instructions,
            analysis,
            iteration_history,
            client
        )

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

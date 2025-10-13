#!/usr/bin/env python3
"""
Iterative Post-Processing Optimizer

Instead of optimizing prompts, this optimizes the post-processing logic.
Uses Qwen to generate Python code for cleaning/fixing model outputs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

from scripts.inference import run_inference
from scripts.run_evaluation import evaluate_completions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/iterative_postprocess_optimizer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def analyze_failures(results: List[Dict]) -> Dict:
    """
    Analyze evaluation results to identify failure patterns.

    Returns:
        Analysis dictionary with error patterns and detailed results
    """
    failures = [r for r in results if not r['passed']]
    successes = [r for r in results if r['passed']]

    return {
        'total': len(results),
        'failures': len(failures),
        'successes': len(successes),
        'detailed_results': results,
    }


def generate_postprocess_code(
    current_postprocess_code: str,
    analysis: Dict,
    iteration_history: List[Dict],
    client
) -> str:
    """
    Generate refined post-processing code using Qwen as the evaluator.

    Args:
        current_postprocess_code: Current post-processing Python code
        analysis: Failure analysis dictionary
        iteration_history: List of previous iterations with code and accuracy
        client: VLLMInference client to call Qwen

    Returns:
        Updated post-processing Python code
    """
    from datasets import load_dataset

    failures = analysis['failures']
    total = analysis['total']

    if failures == 0:
        return current_postprocess_code

    # Build iteration history
    history_summary = "Previous post-processing attempts:\n"
    for hist in iteration_history:
        iter_num = hist['iteration']
        acc = hist['accuracy']
        passed = hist['passed']
        history_summary += f"\nIteration {iter_num}: {acc:.3f} ({passed}/164 passed)\n"

    # Get actual failure details
    failures_list = [r for r in analysis.get('detailed_results', []) if not r['passed']]

    # Load HumanEval to get original problems
    humaneval = load_dataset("openai_humaneval", split="test")
    humaneval_dict = {item['task_id']: item for item in humaneval}

    # Show 5 detailed failure examples
    failure_examples = ""
    for i, fail in enumerate(failures_list[:5]):
        task_id = fail['task_id']
        error = fail.get('error', 'Unknown error')

        # Get original problem
        original_problem = humaneval_dict.get(task_id, {}).get('prompt', 'N/A')
        if len(original_problem) > 300:
            original_problem = original_problem[:300] + "\n... (truncated)"

        # Get raw completion (before post-processing)
        raw_completion = fail.get('raw_completion', 'N/A')
        if len(raw_completion) > 300:
            raw_completion = raw_completion[:300] + "\n... (truncated)"

        # Get the final completion (after post-processing)
        final_code = fail.get('completion', 'N/A')
        if len(final_code) > 300:
            final_code = final_code[:300] + "\n... (truncated)"

        failure_examples += f"""
Failure {i+1}: {task_id}

Original Problem:
{original_problem}

Raw Model Output (before post-processing):
{raw_completion}

After Current Post-Processing:
{final_code}

Error When Running:
{error[:400]}

---"""

    if len(failures_list) > 5:
        failure_examples += f"\n\n... and {len(failures_list) - 5} more failures"

    failure_summary = f"""
Current baseline failures: {failures}/{total}

Sample of failures (showing raw output vs post-processed output):
{failure_examples}
"""

    # Meta-prompt for Qwen to generate post-processing code
    meta_prompt = f"""You are a Python code expert. Your task is to write a post-processing function that cleans up raw model outputs.

{history_summary}

{failure_summary}

Current post-processing code:
```python
{current_postprocess_code if current_postprocess_code else "# No post-processing (returns raw output)"}
```

Task: Write an IMPROVED post_process function that fixes the issues above. The function signature must be:

def post_process(raw_completion: str, prompt: str) -> str:
    \"\"\"Clean up raw model output.\"\"\"
    # Your code here
    return cleaned_code

Common fixes needed:
- Remove echoed prompts
- Strip markdown code blocks (```python)
- Remove trailing comments
- Fix indentation issues
- Remove incomplete lines
- Stop at next function/class definitions

Write the complete function below:

```python
def post_process(raw_completion: str, prompt: str) -> str:
"""

    # Call Qwen to generate post-processing code
    try:
        response = client.generate_completion(
            prompt=meta_prompt,
            max_tokens=800,
            temperature=0.7,
            stop=["```\n\n", "# Test", "# Example"]
        )

        # Extract the function code
        new_code = "def post_process(raw_completion: str, prompt: str) -> str:\n" + response.strip()

        # Basic validation - check if it at least has a return statement
        if 'return' not in new_code:
            logger.warning("Generated code doesn't have return statement, falling back")
            return fallback_postprocess_code()

        logger.info(f"Qwen generated new post-processing code ({len(new_code)} chars)")
        return new_code

    except Exception as e:
        logger.error(f"Error calling Qwen for post-processing code: {e}")
        return fallback_postprocess_code()


def fallback_postprocess_code() -> str:
    """Fallback basic post-processing if Qwen fails."""
    return """def post_process(raw_completion: str, prompt: str) -> str:
    # Remove echoed prompt
    if raw_completion.startswith(prompt):
        raw_completion = raw_completion[len(prompt):]

    # Strip whitespace
    return raw_completion.strip()
"""


def create_postprocess_function(code: str):
    """
    Execute the post-processing code and return the function.

    Args:
        code: Python code containing post_process function

    Returns:
        The post_process function
    """
    import re

    try:
        # Create a namespace and execute the code
        namespace = {}
        exec(code, namespace)

        # Get the post_process function
        if 'post_process' not in namespace:
            logger.error("Generated code doesn't define post_process function")
            # Return identity function
            return lambda c, p: c

        return namespace['post_process']

    except Exception as e:
        logger.error(f"Error executing post-processing code: {e}")
        # Return identity function
        return lambda c, p: c


def run_with_postprocess(postprocess_code: str, iteration: int) -> tuple:
    """
    Run inference and evaluation with custom post-processing code.

    Args:
        postprocess_code: Python code for post-processing
        iteration: Iteration number

    Returns:
        (completions_file, results, accuracy)
    """
    # Create the post-processing function from code
    postprocess_fn = create_postprocess_function(postprocess_code)

    # Register as a custom strategy
    from prompts.advanced_prompts import POSTPROCESS_STRATEGIES
    POSTPROCESS_STRATEGIES['dynamic'] = lambda c, p, e=None: postprocess_fn(c, p)

    # Output files
    completions_file = f"results/postprocess_iter_{iteration}_completions.jsonl"
    results_file = f"results/postprocess_iter_{iteration}_results.json"
    failures_log = f"logs/postprocess_iter_{iteration}_failures.log"

    # Run inference with minimal prompt but custom post-processing
    logger.info(f"Running inference (iteration {iteration})...")
    run_inference(
        output_path=completions_file,
        temperature=0.2,
        prompt_strategy='minimal',  # Always use minimal prompt
        postprocess_strategy='dynamic',  # Use our custom post-processing
        num_workers=16,
    )

    # Evaluate
    logger.info(f"Evaluating (iteration {iteration})...")
    eval_summary = evaluate_completions(
        completions_file=completions_file,
        output_file=results_file
    )

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
    """Run iterative post-processing optimization."""
    logger.info("="*80)
    logger.info("Starting Iterative Post-Processing Optimization")
    logger.info("="*80)

    max_iterations = 10

    # Initialize vLLM client for meta-prompting
    from scripts.inference import VLLMInference
    client = VLLMInference(api_url="http://localhost:8000/v1")
    logger.info("Initialized Qwen client for post-processing optimization")

    # Initialize with no post-processing (identity function)
    current_postprocess_code = """def post_process(raw_completion: str, prompt: str) -> str:
    return raw_completion  # No post-processing
"""
    current_results = None
    current_accuracy = 0.0

    best_postprocess_code = ""
    best_accuracy = 0.0

    # Evaluate initial (no post-processing)
    logger.info("\n[Iteration 0] Evaluating with no post-processing...")
    _, current_results, current_accuracy = run_with_postprocess(current_postprocess_code, 0)

    best_accuracy = current_accuracy
    best_postprocess_code = current_postprocess_code

    logger.info(f"Initial accuracy: {current_accuracy:.3f} ({int(current_accuracy*164)}/164)")

    iteration_history = [{
        'iteration': 0,
        'postprocess_code': current_postprocess_code,
        'accuracy': current_accuracy,
        'passed': int(current_accuracy * 164),
        'improved': True
    }]

    # Iterate up to 10 times
    for iteration in range(1, max_iterations + 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"[Iteration {iteration}]")
        logger.info(f"{'='*80}")

        # Analyze current results
        analysis = analyze_failures(current_results)
        logger.info(f"Current failures: {analysis['failures']}/{analysis['total']}")

        if analysis['failures'] == 0:
            logger.info("Perfect score! No refinement needed.")
            break

        # Generate refined post-processing code using Qwen
        logger.info("Calling Qwen to generate improved post-processing code...")
        new_postprocess_code = generate_postprocess_code(
            current_postprocess_code,
            analysis,
            iteration_history,
            client
        )

        if new_postprocess_code == current_postprocess_code:
            logger.info("No new post-processing code generated. Stopping.")
            break

        logger.info(f"\nGenerated new post-processing code:\n{new_postprocess_code}\n")

        # Test new post-processing on full dataset
        logger.info("Testing new post-processing on full dataset...")
        _, new_results, new_accuracy = run_with_postprocess(new_postprocess_code, iteration)

        logger.info(f"New accuracy: {new_accuracy:.3f} ({int(new_accuracy*164)}/164)")
        logger.info(f"Current baseline: {current_accuracy:.3f} ({int(current_accuracy*164)}/164)")

        # Compare and update
        improved = False
        if new_accuracy > current_accuracy:
            improvement = new_accuracy - current_accuracy
            logger.info(f"✓ IMPROVEMENT: +{improvement:.3f}")
            logger.info(f"Updating current baseline to new post-processing")

            current_postprocess_code = new_postprocess_code
            current_results = new_results
            current_accuracy = new_accuracy
            improved = True

            if new_accuracy > best_accuracy:
                best_postprocess_code = new_postprocess_code
                best_accuracy = new_accuracy
        else:
            logger.info(f"✗ No improvement. Keeping current baseline.")

        iteration_history.append({
            'iteration': iteration,
            'postprocess_code': new_postprocess_code,
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
    logger.info(f"\nBest post-processing code:\n{best_postprocess_code}")

    # Save results
    output = {
        'final_baseline_accuracy': current_accuracy,
        'best_accuracy': best_accuracy,
        'best_passed': int(best_accuracy * 164),
        'best_postprocess_code': best_postprocess_code,
        'total_iterations': len(iteration_history) - 1,
        'improvements': sum(1 for h in iteration_history if h.get('improved', False)),
        'history': iteration_history
    }

    Path('results').mkdir(exist_ok=True)
    with open('results/iterative_postprocess_results.json', 'w') as f:
        json.dump(output, f, indent=2)

    with open('results/best_postprocess_code.py', 'w') as f:
        f.write(best_postprocess_code)

    logger.info(f"\nTotal iterations: {len(iteration_history) - 1}")
    logger.info(f"Successful improvements: {output['improvements']}")
    logger.info("\nResults saved to results/")


if __name__ == '__main__':
    Path('logs').mkdir(exist_ok=True)
    optimize()

#!/usr/bin/env python3
"""
Iterative Prompt Optimizer for HumanEval Code Generation

This script automatically refines prompts by:
1. Testing on batches of 8 problems sequentially
2. Analyzing failure patterns
3. Generating refined prompts
4. Validating on all 164 problems
5. Keeping the better performing prompt
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple
from datasets import load_dataset
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.inference import generate_completion, get_vllm_client
from scripts.run_evaluation import evaluate_completions
from prompts.advanced_prompts import create_minimal_prompt, smart_post_process


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


class PromptOptimizer:
    """Automated iterative prompt optimizer."""

    def __init__(
        self,
        batch_size: int = 41,
        temperature: float = 0.2,
        max_tokens: int = 768,
    ):
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Track best prompt and its performance
        self.best_prompt_template = None
        self.best_accuracy = 0.0
        self.best_passed = 0

        # Load dataset
        logger.info("Loading HumanEval dataset...")
        self.dataset = load_dataset("openai_humaneval", split="test")
        logger.info(f"Loaded {len(self.dataset)} problems")

        # Initialize vLLM client
        self.client = get_vllm_client()

    def analyze_batch_results(
        self,
        batch_problems: List[Dict],
        batch_results: List[Dict]
    ) -> Dict[str, any]:
        """
        Deeply analyze BOTH failures and successes to understand patterns.

        Returns:
            Dictionary with comprehensive analysis
        """
        failures = []
        successes = []

        for problem, result in zip(batch_problems, batch_results):
            if result['passed']:
                successes.append({
                    'task_id': problem['task_id'],
                    'prompt': problem['prompt'],
                    'completion': result['completion'],
                    'entry_point': problem.get('entry_point', '')
                })
            else:
                failures.append({
                    'task_id': problem['task_id'],
                    'prompt': problem['prompt'],
                    'completion': result['completion'],
                    'error': result.get('error', 'Unknown error'),
                    'entry_point': problem.get('entry_point', '')
                })

        # Categorize errors in failures
        error_categories = {
            'syntax_errors': 0,
            'assertion_errors': 0,
            'type_errors': 0,
            'index_errors': 0,
            'key_errors': 0,
            'attribute_errors': 0,
            'timeout_errors': 0,
            'name_errors': 0,
            'other_errors': 0
        }

        for failure in failures:
            error_msg = failure['error'].lower()
            if 'syntaxerror' in error_msg or 'indentation' in error_msg:
                error_categories['syntax_errors'] += 1
            elif 'assertionerror' in error_msg or 'assert' in error_msg:
                error_categories['assertion_errors'] += 1
            elif 'typeerror' in error_msg:
                error_categories['type_errors'] += 1
            elif 'indexerror' in error_msg:
                error_categories['index_errors'] += 1
            elif 'keyerror' in error_msg:
                error_categories['key_errors'] += 1
            elif 'attributeerror' in error_msg:
                error_categories['attribute_errors'] += 1
            elif 'timeout' in error_msg:
                error_categories['timeout_errors'] += 1
            elif 'nameerror' in error_msg:
                error_categories['name_errors'] += 1
            else:
                error_categories['other_errors'] += 1

        # Analyze problem characteristics for failures
        failure_characteristics = self._analyze_problem_characteristics(failures)
        success_characteristics = self._analyze_problem_characteristics(successes)

        return {
            'total': len(batch_results),
            'failures': len(failures),
            'successes': len(successes),
            'failure_details': failures,
            'success_details': successes,
            'error_categories': error_categories,
            'failure_characteristics': failure_characteristics,
            'success_characteristics': success_characteristics
        }

    def _analyze_problem_characteristics(self, problems: List[Dict]) -> Dict:
        """
        Analyze what types of problems succeeded or failed.

        Returns insights about:
        - Problem complexity (docstring length, examples count)
        - Common keywords (list, dict, string, math operations)
        - Edge cases mentioned
        """
        if not problems:
            return {}

        characteristics = {
            'avg_docstring_length': 0,
            'has_examples': 0,
            'involves_lists': 0,
            'involves_strings': 0,
            'involves_math': 0,
            'involves_dicts': 0,
            'mentions_edge_cases': 0,
            'mentions_empty': 0,
            'has_multiple_conditions': 0,
            'avg_code_length': 0
        }

        for prob in problems:
            prompt = prob['prompt'].lower()
            completion = prob['completion']

            # Docstring analysis
            docstring_start = prompt.find('"""')
            if docstring_start != -1:
                docstring_end = prompt.find('"""', docstring_start + 3)
                if docstring_end != -1:
                    docstring = prompt[docstring_start:docstring_end]
                    characteristics['avg_docstring_length'] += len(docstring)

                    # Check for examples (>>>, Example:, etc.)
                    if '>>>' in docstring or 'example' in docstring:
                        characteristics['has_examples'] += 1

                    # Edge case mentions
                    if 'edge' in docstring or 'corner' in docstring:
                        characteristics['mentions_edge_cases'] += 1
                    if 'empty' in docstring or 'none' in docstring:
                        characteristics['mentions_empty'] += 1

            # Data structure analysis
            if 'list' in prompt or '[' in prompt or 'array' in prompt:
                characteristics['involves_lists'] += 1
            if 'string' in prompt or 'str' in prompt or "'" in prompt:
                characteristics['involves_strings'] += 1
            if 'dict' in prompt or '{' in prompt or 'key' in prompt:
                characteristics['involves_dicts'] += 1

            # Math operations
            if any(word in prompt for word in ['sum', 'product', 'multiply', 'add', 'subtract', 'divide', 'math', 'number']):
                characteristics['involves_math'] += 1

            # Code complexity
            if completion:
                characteristics['avg_code_length'] += len(completion)
                # Multiple conditions (if, elif, for, while)
                condition_count = completion.count('if ') + completion.count('elif ') + \
                                completion.count('for ') + completion.count('while ')
                if condition_count >= 2:
                    characteristics['has_multiple_conditions'] += 1

        # Calculate averages
        n = len(problems)
        characteristics['avg_docstring_length'] = characteristics['avg_docstring_length'] // n if n > 0 else 0
        characteristics['avg_code_length'] = characteristics['avg_code_length'] // n if n > 0 else 0

        return characteristics

    def generate_refined_prompt(
        self,
        current_prompt_instructions: str,
        batch_analysis: Dict
    ) -> str:
        """
        Generate a refined prompt based on BOTH failure and success analysis.

        Analyzes WHY things failed and WHY things succeeded to create
        better prompting instructions.

        Args:
            current_prompt_instructions: Current additional instructions (empty for minimal)
            batch_analysis: Analysis of recent batch (failures + successes)

        Returns:
            New prompt instructions
        """
        error_cats = batch_analysis['error_categories']
        failures = batch_analysis['failures']
        successes = batch_analysis['successes']
        total = batch_analysis['total']

        fail_chars = batch_analysis['failure_characteristics']
        success_chars = batch_analysis['success_characteristics']

        if failures == 0:
            return current_prompt_instructions  # No changes needed

        # Build refinement instructions based on comprehensive analysis
        refinements = []

        # === ANALYZE WHAT'S FAILING ===

        # Syntax errors - add clarity about code structure
        if error_cats['syntax_errors'] > 0:
            refinements.append(
                "- Write complete, syntactically correct Python code with proper indentation"
            )

        # Assertion errors - emphasize docstring examples
        if error_cats['assertion_errors'] > failures * 0.3:  # >30% of failures
            refinements.append(
                "- CRITICAL: The examples in the docstring are test cases - your code MUST pass them exactly"
            )
            refinements.append(
                "- Trace through each example step-by-step to verify your logic"
            )

        # Type errors - add type awareness
        if error_cats['type_errors'] > 0:
            refinements.append(
                "- Pay careful attention to return types (list vs tuple, int vs float, string vs number)"
            )

        # Index/Key errors - edge cases
        if error_cats['index_errors'] > 0 or error_cats['key_errors'] > 0:
            refinements.append(
                "- Always handle edge cases: empty inputs, missing keys, boundary indices"
            )

        # Name errors - undefined variables
        if error_cats['name_errors'] > 0:
            refinements.append(
                "- Make sure all variables are defined before use - no undefined references"
            )

        # Timeout errors - infinite loops
        if error_cats['timeout_errors'] > 0:
            refinements.append(
                "- Ensure loops terminate: verify loop conditions and avoid infinite recursion"
            )

        # === COMPARE FAILURE vs SUCCESS CHARACTERISTICS ===

        # If failures involve lists but successes don't, add list-specific guidance
        if fail_chars.get('involves_lists', 0) > success_chars.get('involves_lists', 0):
            refinements.append(
                "- For list operations: check boundaries, handle empty lists, verify indices"
            )

        # If failures involve edge case mentions more than successes
        if fail_chars.get('mentions_edge_cases', 0) > 0:
            refinements.append(
                "- When docstring mentions edge cases explicitly, test them thoroughly"
            )

        # If failures mention 'empty' more than successes
        if fail_chars.get('mentions_empty', 0) > 0:
            refinements.append(
                "- Handle empty inputs explicitly (empty lists, empty strings, zero values)"
            )

        # If failures have more complex conditions than successes
        if fail_chars.get('has_multiple_conditions', 0) > success_chars.get('has_multiple_conditions', 0):
            refinements.append(
                "- For complex logic with multiple conditions, verify each branch carefully"
            )

        # === ANALYZE WHAT'S WORKING ===

        # If successes have examples and failures don't leverage them
        if success_chars.get('has_examples', 0) > 0 and error_cats['assertion_errors'] > 0:
            refinements.append(
                "- Use the provided examples as your ground truth - implement to match them exactly"
            )

        # High failure rate - need fundamental improvements
        if failures > total * 0.6:  # >60% failure rate
            refinements.append(
                "- Read the entire docstring carefully before coding - requirements are precisely specified"
            )
            refinements.append(
                "- Think step-by-step about the algorithm before implementing"
            )
            refinements.append(
                "- Consider what the function should return for typical cases AND edge cases"
            )

        # Medium failure rate - need targeted improvements
        elif failures > total * 0.3:  # 30-60% failure rate
            refinements.append(
                "- Double-check your understanding of what the function should return"
            )
            refinements.append(
                "- Verify edge cases are handled correctly"
            )

        # Remove duplicates from current instructions
        if current_prompt_instructions:
            existing_lines = set(current_prompt_instructions.split('\n'))
            refinements = [r for r in refinements if r not in existing_lines]

        if not refinements:
            return current_prompt_instructions

        # Combine old and new instructions
        if current_prompt_instructions:
            new_instructions = current_prompt_instructions + "\n" + "\n".join(refinements)
        else:
            new_instructions = "\n".join(refinements)

        return new_instructions

    def create_prompt_with_instructions(self, problem: str, instructions: str) -> str:
        """Create a prompt with additional instructions."""
        if not instructions:
            return problem  # Minimal prompt

        return f"""{problem}

# Important Guidelines:
{instructions}
"""

    def generate_batch_completions(
        self,
        batch_problems: List[Dict],
        prompt_instructions: str
    ) -> List[Dict]:
        """Generate completions for a batch of problems."""
        results = []

        for problem_data in tqdm(batch_problems, desc="Generating batch", leave=False):
            problem = problem_data['prompt']
            prompt = self.create_prompt_with_instructions(problem, prompt_instructions)

            # Generate completion
            raw_completion = generate_completion(
                self.client,
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Post-process
            completion = smart_post_process(raw_completion, prompt)

            # Combine prompt + completion
            full_code = problem + "\n" + completion

            results.append({
                'task_id': problem_data['task_id'],
                'completion': full_code
            })

        return results

    def generate_all_completions(
        self,
        prompt_instructions: str
    ) -> List[Dict]:
        """Generate completions for ALL 164 problems."""
        logger.info(f"Generating completions for all {len(self.dataset)} problems...")

        all_completions = []

        for problem_data in tqdm(self.dataset, desc="Full evaluation"):
            problem = problem_data['prompt']
            prompt = self.create_prompt_with_instructions(problem, prompt_instructions)

            # Generate completion
            raw_completion = generate_completion(
                self.client,
                prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Post-process
            completion = smart_post_process(raw_completion, prompt)

            # Combine prompt + completion
            full_code = problem + "\n" + completion

            all_completions.append({
                'task_id': problem_data['task_id'],
                'completion': full_code
            })

        return all_completions

    def evaluate_completions_full(self, completions: List[Dict]) -> Tuple[float, int, int]:
        """
        Evaluate completions and return accuracy metrics.

        Returns:
            (accuracy, passed_count, total_count)
        """
        results = evaluate_completions(completions)

        passed = sum(1 for r in results if r['passed'])
        total = len(results)
        accuracy = passed / total if total > 0 else 0.0

        return accuracy, passed, total

    def optimize(self) -> Dict:
        """
        Run the iterative optimization process.

        Returns:
            Dictionary with optimization results
        """
        logger.info("="*80)
        logger.info("Starting Iterative Prompt Optimization")
        logger.info("="*80)

        # Initialize with minimal prompt
        current_prompt_instructions = ""
        self.best_prompt_template = current_prompt_instructions

        # Evaluate initial minimal prompt on full dataset
        logger.info("\nEvaluating initial minimal prompt on full dataset...")
        initial_completions = self.generate_all_completions(current_prompt_instructions)
        initial_accuracy, initial_passed, total = self.evaluate_completions_full(initial_completions)

        self.best_accuracy = initial_accuracy
        self.best_passed = initial_passed

        logger.info(f"Initial Accuracy: {initial_accuracy:.3f} ({initial_passed}/{total})")
        logger.info(f"Target: >0.5 (>82/164)")

        # Track iterations
        iteration_history = [{
            'iteration': 0,
            'batch_range': 'N/A',
            'prompt_instructions': current_prompt_instructions,
            'accuracy': initial_accuracy,
            'passed': initial_passed,
            'total': total,
            'is_best': True
        }]

        # Iterate through dataset in batches
        num_batches = (len(self.dataset) + self.batch_size - 1) // self.batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, len(self.dataset))
            batch_problems = [self.dataset[i] for i in range(start_idx, end_idx)]

            logger.info(f"\n{'='*80}")
            logger.info(f"Iteration {batch_idx + 1}/{num_batches}: Problems {start_idx}-{end_idx-1}")
            logger.info(f"{'='*80}")

            # Generate completions for this batch
            batch_completions = self.generate_batch_completions(
                batch_problems,
                current_prompt_instructions
            )

            # Evaluate batch
            batch_results = evaluate_completions(batch_completions)

            # Analyze BOTH failures and successes
            batch_analysis = self.analyze_batch_results(batch_problems, batch_results)

            logger.info(f"Batch results: {batch_analysis['successes']}/{batch_analysis['total']} passed")

            if batch_analysis['failures'] == 0:
                logger.info("All problems in batch passed! Skipping refinement.")
                iteration_history.append({
                    'iteration': batch_idx + 1,
                    'batch_range': f"{start_idx}-{end_idx-1}",
                    'prompt_instructions': current_prompt_instructions,
                    'accuracy': self.best_accuracy,
                    'passed': self.best_passed,
                    'total': total,
                    'is_best': False,
                    'note': 'Skipped - all passed'
                })
                continue

            # Generate refined prompt based on failures AND successes
            logger.info("Analyzing batch (failures + successes) and refining prompt...")
            logger.info(f"Error categories: {batch_analysis['error_categories']}")
            logger.info(f"Failure characteristics: {batch_analysis['failure_characteristics']}")
            logger.info(f"Success characteristics: {batch_analysis['success_characteristics']}")

            new_prompt_instructions = self.generate_refined_prompt(
                current_prompt_instructions,
                batch_analysis
            )

            if new_prompt_instructions == current_prompt_instructions:
                logger.info("No new refinements generated. Continuing...")
                continue

            logger.info(f"\nRefined prompt instructions:\n{new_prompt_instructions}\n")

            # Evaluate refined prompt on FULL dataset
            logger.info("Evaluating refined prompt on full dataset...")
            new_completions = self.generate_all_completions(new_prompt_instructions)
            new_accuracy, new_passed, total = self.evaluate_completions_full(new_completions)

            logger.info(f"New Accuracy: {new_accuracy:.3f} ({new_passed}/{total})")
            logger.info(f"Previous Best: {self.best_accuracy:.3f} ({self.best_passed}/{total})")

            # Compare and update if better
            is_best = False
            if new_accuracy > self.best_accuracy:
                improvement = new_accuracy - self.best_accuracy
                logger.info(f"✓ IMPROVEMENT: +{improvement:.3f} ({new_passed - self.best_passed} more problems)")
                current_prompt_instructions = new_prompt_instructions
                self.best_accuracy = new_accuracy
                self.best_passed = new_passed
                self.best_prompt_template = new_prompt_instructions
                is_best = True
            else:
                logger.info(f"✗ No improvement. Keeping previous prompt.")

            iteration_history.append({
                'iteration': batch_idx + 1,
                'batch_range': f"{start_idx}-{end_idx-1}",
                'prompt_instructions': new_prompt_instructions,
                'accuracy': new_accuracy,
                'passed': new_passed,
                'total': total,
                'is_best': is_best
            })

        # Final results
        logger.info("\n" + "="*80)
        logger.info("OPTIMIZATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Initial Accuracy: {initial_accuracy:.3f} ({initial_passed}/{total})")
        logger.info(f"Final Best Accuracy: {self.best_accuracy:.3f} ({self.best_passed}/{total})")
        logger.info(f"Improvement: +{self.best_accuracy - initial_accuracy:.3f} ({self.best_passed - initial_passed} more problems)")
        logger.info(f"\nBest Prompt Instructions:\n{self.best_prompt_template}")

        return {
            'initial_accuracy': initial_accuracy,
            'initial_passed': initial_passed,
            'final_accuracy': self.best_accuracy,
            'final_passed': self.best_passed,
            'total_problems': total,
            'improvement': self.best_accuracy - initial_accuracy,
            'best_prompt_template': self.best_prompt_template,
            'iteration_history': iteration_history
        }


def main():
    """Main entry point."""
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Initialize optimizer with batch_size=41 (4 iterations total)
    optimizer = PromptOptimizer(batch_size=41, temperature=0.2)

    # Run optimization
    results = optimizer.optimize()

    # Save results
    output_file = Path('results/iterative_optimization_results.json')
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")

    # Save best prompt template
    prompt_file = Path('results/best_prompt_template.txt')
    with open(prompt_file, 'w') as f:
        f.write(results['best_prompt_template'])

    logger.info(f"Best prompt template saved to: {prompt_file}")


if __name__ == '__main__':
    main()

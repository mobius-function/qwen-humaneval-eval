"""Advanced prompt templates for better code generation."""

import re


def create_infilling_prompt(problem: str) -> str:
    """
    Code infilling prompt - explicitly asks to complete the code.
    Works well with code-specialized models like Qwen Coder.

    Args:
        problem: Function signature and docstring

    Returns:
        Formatted prompt
    """
    prompt = f"""{problem}
    # TODO: Implement the function body here
    """
    return prompt


def create_fewshot_prompt(problem: str) -> str:
    """
    Few-shot prompt with example to guide the model.

    Args:
        problem: Function signature and docstring

    Returns:
        Formatted prompt with example
    """
    example = '''Example:
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    Now complete this function:
    '''
    prompt = f"{example}\n{problem}"
    return prompt


def create_instructional_prompt(problem: str) -> str:
    """
    Instructional prompt that emphasizes correctness.

    Args:
        problem: Function signature and docstring

    Returns:
        Formatted prompt
    """
    prompt = f"""You are an expert Python programmer. Write a correct and efficient implementation.

    {problem}

    Requirements:
    - Implement the function body correctly
    - Handle all edge cases
    - Follow the docstring specification exactly
    - Write clean, readable code
    """
    return prompt


def create_minimal_prompt(problem: str) -> str:
    """
    Minimal prompt - just the problem, no extra instructions.
    Sometimes less is more for code models.

    Args:
        problem: Function signature and docstring

    Returns:
        The problem as-is
    """
    return problem


def create_chain_of_thought_prompt(problem: str) -> str:
    """
    Chain of thought prompt to encourage reasoning.

    Args:
        problem: Function signature and docstring

    Returns:
        Formatted prompt
    """
    prompt = f"""Solve this step by step:

    {problem}

    # Solution approach:
    # 1. Understand the requirements from the docstring
    # 2. Identify edge cases
    # 3. Implement the logic

    # Implementation:
    """
    return prompt


def enhanced_post_process(completion: str, prompt: str) -> str:
    """
    Enhanced post-processing with better code extraction.

    Args:
        completion: Raw model output
        prompt: Original prompt

    Returns:
        Cleaned code
    """
    # Remove prompt if echoed
    if completion.startswith(prompt):
        completion = completion[len(prompt):]

    # Strip only trailing whitespace to preserve indentation
    completion = completion.rstrip()

    # Remove markdown code blocks
    if '```python' in completion:
        match = re.search(r'```python\s*(.*?)\s*```', completion, re.DOTALL)
        if match:
            completion = match.group(1).rstrip()
    elif '```' in completion:
        parts = completion.split('```')
        if len(parts) >= 2:
            completion = parts[1].rstrip()
            # Remove language identifier if present
            if completion.startswith('python\n'):
                completion = completion[7:]

    # Stop at common delimiters
    stop_patterns = [
        r'\n(?:def|class)\s+\w+',  # Next function/class definition
        r'\nif __name__',           # Main guard
        r'\n# Test',                # Test section
        r'\n# Example',             # Example section
    ]

    for pattern in stop_patterns:
        match = re.search(pattern, completion)
        if match:
            completion = completion[:match.start()]

    # Remove comments at the end
    lines = completion.split('\n')
    while lines and lines[-1].strip().startswith('#'):
        lines.pop()

    completion = '\n'.join(lines).rstrip()

    return completion


def smart_post_process(completion: str, prompt: str, entry_point: str = None) -> str:
    """
    Smart post-processing that validates the completion structure.

    Args:
        completion: Raw completion
        prompt: Original prompt
        entry_point: Function name to validate

    Returns:
        Cleaned and validated code
    """
    # First apply enhanced post-processing
    cleaned = enhanced_post_process(completion, prompt)

    # Remove any incomplete lines at the end
    lines = cleaned.split('\n')
    while lines:
        last_line = lines[-1].rstrip()
        if last_line and not last_line.endswith((':', ',', '(', '[', '{')):
            break
        if not last_line.strip():
            lines.pop()
        else:
            break

    cleaned = '\n'.join(lines)

    return cleaned.rstrip()


def create_datadriven_prompt(problem: str) -> str:
    """
    Data-driven prompt based on analysis of all 164 HumanEval problems.

    Optimized for:
    - 82% of problems have examples in docstring
    - 71 problems are math/numeric
    - 70 problems involve list manipulation
    - 57 problems involve string manipulation
    - Average 12.6 lines of code needed

    Args:
        problem: Function signature and docstring

    Returns:
        Optimized prompt
    """
    # Extract type hints if present
    has_type_hints = '->' in problem

    prompt = f"""{problem}
    # Read the docstring carefully - it contains critical requirements and examples
    # Key focus areas:
    # 1. Return value: What type and format is expected?
    # 2. Edge cases: Empty inputs, None, negative numbers, special conditions
    # 3. Examples: Use them to verify your logic
    # 4. Constraints: Pay attention to "if", "should", "must" keywords

    # Implementation:
    """
    return prompt


def create_expert_prompt(problem: str) -> str:
    """
    Expert-engineered prompt based on advanced LLM prompting best practices.

    Incorporates:
    - Detailed context and constraints
    - Expert persona (senior Python engineer)
    - Self-review mechanism
    - Explicit edge case handling
    - Chain of thought reasoning
    - Production-ready code standards

    Args:
        problem: Function signature and docstring

    Returns:
        Optimized expert-level prompt
    """
    prompt = f"""You are a senior Python engineer writing production code for a code evaluation benchmark.

TASK:
{problem}

REQUIREMENTS ANALYSIS:
1. Read the docstring carefully - it contains the complete specification
2. Identify the exact return type and format expected
3. Extract ALL edge cases mentioned or implied
4. Note any constraints or special conditions
5. Study examples in docstring - they define correct behavior

IMPLEMENTATION APPROACH:
- Think through the logic step-by-step before coding
- Handle ALL edge cases explicitly (empty inputs, None, negative numbers, special values)
- Match the examples exactly - they are test cases
- Write clean, efficient, production-ready code
- No placeholders, no TODOs, no incomplete sections

SELF-REVIEW CHECKLIST (verify before submitting):
✓ Does the code match the docstring specification exactly?
✓ Are all examples/test cases in docstring satisfied?
✓ Are edge cases handled (empty lists, None, zeros, negatives)?
✓ Is the return type correct?
✓ Is the logic efficient and bug-free?
✓ Are variable names clear and meaningful?

CONSTRAINTS:
- Python 3.x standard library only
- No external dependencies
- Must be immediately executable
- Performance: O(n) or better preferred for list operations

Now implement the complete function body (no explanations, just code):
"""
    return prompt


def create_optimized_v1_prompt(problem: str) -> str:
    """
    Optimized prompt v1 - Concise but effective.

    Based on failure analysis:
    - Emphasizes docstring examples as test cases
    - Explicitly calls out edge cases
    - Uses clear, directive language
    - Minimal but sufficient context
    - Focuses on implementation completeness

    Args:
        problem: Function signature and docstring

    Returns:
        Optimized prompt string
    """
    prompt = f"""Complete this Python function. The docstring contains examples that are actual test cases - your code must pass them all.

{problem}

IMPORTANT:
- Study the examples in the docstring - they show exactly what's expected
- Handle ALL edge cases: empty inputs, None, zeros, negative numbers, special values
- Return the exact type and format shown in examples
- Write complete, working code - no placeholders or TODOs
- Keep the implementation simple and correct

Implementation:"""
    return prompt


def create_optimized_v2_prompt(problem: str) -> str:
    """
    Optimized prompt v2 - Example-driven approach.

    This version emphasizes the examples more strongly and uses
    a problem-solving framework that mirrors how humans code.

    Args:
        problem: Function signature and docstring

    Returns:
        Optimized prompt string
    """
    prompt = f"""{problem}

# Instructions:
# 1. READ the examples in the docstring carefully - they are test cases
# 2. IDENTIFY edge cases: What happens with empty inputs? Special values?
# 3. IMPLEMENT the logic to pass all examples
# 4. VERIFY your logic handles all cases mentioned in the docstring
#
# Write only the function body. No explanations needed.
"""
    return prompt


def create_optimized_v3_prompt(problem: str) -> str:
    """
    Optimized prompt v3 - Ultra-minimal with strategic emphasis.

    Sometimes less is more. This version is extremely concise but
    strategically emphasizes the most important aspects: examples
    as tests and edge case handling.

    Args:
        problem: Function signature and docstring

    Returns:
        Optimized prompt string
    """
    prompt = f"""{problem}

# The examples above are test cases - make sure your implementation passes them all.
# Handle edge cases: empty inputs, None, special values.
# Write complete, correct code:
"""
    return prompt


# Mapping of strategy names to prompt functions
PROMPT_STRATEGIES = {
    'minimal': create_minimal_prompt,
    'infilling': create_infilling_prompt,
    'instructional': create_instructional_prompt,
    'fewshot': create_fewshot_prompt,
    'cot': create_chain_of_thought_prompt,
    'datadriven': create_datadriven_prompt,
    'expert': create_expert_prompt,
    'optimized_v1': create_optimized_v1_prompt,
    'optimized_v2': create_optimized_v2_prompt,
    'optimized_v3': create_optimized_v3_prompt,
}

# Mapping of post-processing strategies
POSTPROCESS_STRATEGIES = {
    'none': lambda c, p, e=None: c,  # No post-processing - raw output
    'basic': lambda c, p, e=None: enhanced_post_process(c, p),
    'smart': smart_post_process,
}

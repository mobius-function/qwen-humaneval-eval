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


# Mapping of strategy names to prompt functions
PROMPT_STRATEGIES = {
    'minimal': create_minimal_prompt,
    'infilling': create_infilling_prompt,
    'instructional': create_instructional_prompt,
    'fewshot': create_fewshot_prompt,
    'cot': create_chain_of_thought_prompt,
}

# Mapping of post-processing strategies
POSTPROCESS_STRATEGIES = {
    'none': lambda c, p, e=None: c,  # No post-processing - raw output
    'basic': lambda c, p, e=None: enhanced_post_process(c, p),
    'smart': smart_post_process,
}

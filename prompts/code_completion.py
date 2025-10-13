"""Prompt templates for HumanEval code completion."""

def create_completion_prompt(problem: str) -> str:
    """
    Create a prompt for code completion task.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Formatted prompt string
    """
    prompt = f"""Complete the following Python function. Only provide the function body implementation.

{problem}"""
    return prompt


def create_instruction_prompt(problem: str) -> str:
    """
    Alternative instruction-based prompt format.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Formatted prompt string
    """
    prompt = f"""You are an expert Python programmer. Complete the following function:

{problem}

Provide only the complete function implementation without any explanation."""
    return prompt


def create_minimal_prompt_v2(problem: str) -> str:
    """
    Minimal prompt with single critical instruction to prevent stub code.
    Also known as 'try1' in experiments.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Formatted prompt string
    """
    return f"""{problem}
# Implement completely. No pass statements. No undefined functions."""


def create_minimal_v2(problem: str) -> str:
    """
    Minimal prompt that primes with 'return' keyword.
    Includes the function signature, docstring, and a 'return' hint to
    encourage the model to generate complete implementations.

    This strategy primes the model to start with a return statement,
    which often leads to more complete and working code.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Formatted prompt string with 'return' starter
    """
    return problem + "\n    # Your code here\n"


def create_minimal_v3(problem: str) -> str:
    """
    Ultra-minimal prompt with just the problem and a newline.
    Strips trailing whitespace and adds a single newline to prime completion.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Formatted prompt string
    """
    return problem.rstrip() + "\n"


def create_minimal_v4(problem: str) -> str:
    """
    Minimal prompt with indentation hint.
    Strips trailing whitespace and adds a newline with 4-space indentation
    to prime the model for function body completion.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Formatted prompt string with indentation
    """
    return problem.rstrip() + "\n    "


def create_minimal_v5(problem: str) -> str:
    """
    Bare minimal prompt - just the problem with trailing whitespace removed.
    No additional characters added. Removes trailing whitespace/newlines,
    but keeps the internal structure intact.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Problem string with trailing whitespace stripped
    """
    return problem.rstrip()  # remove trailing whitespace/newlines, but keep structure


def create_minimal_v6(problem: str) -> str:
    """
    Provides the model with clear, direct instructions to improve code generation.

    Args:
        problem: The string containing the function signature and docstring.

    Returns:
        A detailed and structured prompt for the language model.
    """
    # This f-string is the core of the new prompt.
    return f"""You are an expert Python programmer. Your task is to write a correct, efficient, and complete Python function.
Pay close attention to all details, constraints, and edge cases described in the docstring and examples.

**Problem:**
{problem}

**Instructions:**
1.  Carefully read the function signature and the entire docstring.
2.  Your solution must correctly handle all examples provided in the docstring.
3.  Address all edge cases mentioned or implied, such as empty inputs, negative numbers, or specific formatting rules.
4.  If you need a helper function (e.g., `is_prime`) or a library (e.g., `math`, `hashlib`), you **must** implement the helper or include the necessary `import` statement. Do not assume they are already defined.
5.  Provide only the code that belongs inside the function. Do not repeat the `def` line or the docstring. Your response should be ready to be indented and placed into the function body.
"""


def create_robust_prompt(problem: str) -> str:
    """
    A balanced prompt that gives clear instructions without confusing the model.
    It focuses on the most critical failure points: missing imports and incorrect logic.
    """
    return f"""You are a Python programming expert.
Your task is to write the complete and correct Python code for the following problem.
Your solution must be efficient and handle all edge cases.
If helper functions or libraries are needed, you MUST include their implementation or the necessary import statements.

**Problem:**
{problem}

**Function Code:**
"""


def post_process_completion(completion: str, prompt: str) -> str:
    """
    Post-process the model completion to extract clean code.

    Args:
        completion: Raw completion from the model
        prompt: Original prompt sent to model

    Returns:
        Cleaned code string
    """
    # Remove the prompt if it was echoed back
    if completion.startswith(prompt):
        completion = completion[len(prompt):]

    # Strip whitespace
    completion = completion.strip()

    # Stop at common code delimiters
    stop_tokens = ['\nclass ', '\ndef ', '\n#', '\nif __name__']
    for stop in stop_tokens:
        if stop in completion:
            completion = completion[:completion.index(stop)]

    # Remove markdown code blocks if present
    if '```python' in completion:
        start = completion.index('```python') + len('```python')
        end = completion.index('```', start)
        completion = completion[start:end].strip()
    elif '```' in completion:
        parts = completion.split('```')
        if len(parts) >= 2:
            completion = parts[1].strip()

    return completion.strip()

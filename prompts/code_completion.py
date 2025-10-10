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

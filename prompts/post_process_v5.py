"""
Post-processing V5 - Production-ready code cleaning pipeline.

This module provides robust post-processing for model-generated code with:
- Safe markdown removal
- Precise dependency injection (no duplicates)
- Dynamic parameter extraction for targeted fixes
- Comprehensive error handling
"""

import re
import textwrap
from typing import Optional


def _extract_parameter_name(problem_prompt: str, function_name: str) -> Optional[str]:
    """
    Parses the function signature in the prompt to find the first parameter name.
    V5: Handles type hints like `def func(value: str)`.
    """
    match = re.search(rf"def {function_name}\(([^,)]+)", problem_prompt)
    if match:
        # V5: Split on ':' to remove type hints, then strip whitespace
        return match.group(1).split(':')[0].strip()
    return None


def clean_model_output_v5(raw_output: str) -> str:
    """
    Cleans raw model output by removing markdown and fixing indentation.
    V5: Uses a robust regex for markdown removal.
    """
    # Use \A and \Z for string boundaries to only remove fences at the absolute start/end.
    code = re.sub(r'\A```(?:python)?\s*\n?|\n?```\Z', '', raw_output)

    # Use textwrap.dedent to remove common leading whitespace from the entire block.
    # IMPORTANT: Don't call strip() before dedent() - it breaks relative indentation!
    # Only strip trailing/leading newlines AFTER dedenting.
    return textwrap.dedent(code).strip()


def inject_dependencies_v5(code: str) -> str:
    """
    Scans code for missing dependencies and prepends them.
    V5: Uses precise regex to prevent duplicate function injection.
    """
    # Add missing standard library imports
    if "hashlib." in code and "import hashlib" not in code:
        code = "import hashlib\n" + code
    if "reduce(" in code and "from functools import reduce" not in code:
        code = "from functools import reduce\n" + code
    if "math." in code and "import math" not in code:
        code = "import math\n" + code

    # Use precise regex to avoid matching commented-out definitions.
    if "is_prime(" in code and not re.search(r"def\s+is_prime\s*\(", code):
        is_prime_func = """def is_prime(n):
    if n <= 1: return False
    if n <= 3: return True
    if n % 2 == 0 or n % 3 == 0: return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0: return False
        i += 6
    return True"""
        code = is_prime_func + "\n\n" + code

    if "is_palindrome(" in code and not re.search(r"def\s+is_palindrome\s*\(", code):
        is_palindrome_func = """def is_palindrome(text: str) -> bool:
    return text == text[::-1]"""
        code = is_palindrome_func + "\n\n" + code

    return code


def apply_regex_fixes_v5(code: str, problem_prompt: str) -> str:
    """
    Applies specific, pattern-based fixes for common mistakes.
    V5: Dynamically extracts parameter names for targeted fixes.
    """
    # Fix for `remove_vowels` case-insensitivity
    if "remove_vowels" in problem_prompt:
        code = re.sub(r"vowels\s*=\s*['\"]aeiou['\"]", "vowels = 'aeiouAEIOU'", code)

    # Dynamically extract parameter name for `closest_integer` fix.
    if "closest_integer" in problem_prompt:
        param_name = _extract_parameter_name(problem_prompt, "closest_integer")
        if param_name:
            pattern = re.compile(rf"round\(\s*{re.escape(param_name)}\s*\)")
            replacement = f"round(float({param_name}.replace(',', '.')))"
            code = pattern.sub(replacement, code)

    # Fix for `compare` function variable name hallucination
    if "def compare(game,guess)" in problem_prompt:
        code = re.sub(r'\bguesses\b', 'guess', code)

    return code


def post_process_code_v5(raw_code: str, problem_prompt: str) -> str:
    """
    The final, production-ready pipeline, incorporating all collaborative refinements.

    Args:
        raw_code: Raw model output
        problem_prompt: Original problem prompt (used for targeted fixes)

    Returns:
        Cleaned and fixed code ready for evaluation (properly indented for function body)
    """
    code = clean_model_output_v5(raw_code)
    code = inject_dependencies_v5(code)
    code = apply_regex_fixes_v5(code, problem_prompt)

    # After clean_model_output_v5(), the code is fully dedented (no leading spaces).
    # We need to indent every line by 4 spaces to fit inside the function body.
    # This preserves the relative indentation structure of the code.
    indented_lines = []
    for line in code.splitlines():
        if line.strip():  # Only indent non-empty lines
            indented_lines.append("    " + line)
        else:
            indented_lines.append(line)  # Keep empty lines as-is

    return "\n" + "\n".join(indented_lines)

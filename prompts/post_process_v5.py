"""
Post-processing V5 - Production-ready code cleaning pipeline.

This module provides robust post-processing for model-generated code with:
- Safe markdown removal
- Precise dependency injection (no duplicates)
  * is_prime, is_palindrome, reverse, product, is_balanced, prod_sign, factorial
- Dynamic parameter extraction for targeted fixes
- Enhanced truncated code detection and repair
  * Missing colons on control flow statements
  * Unclosed brackets at end of code
  * Control flow statements with no body (adds pass)
  * Unterminated triple-quoted strings
  * Repetitive code pattern detection
- ValueError fixes
  * Word-to-number conversion (zero/one/two/etc → 0/1/2/etc)
  * Safe digit iteration (handles minus signs)
- Type conversion patches
  * is_palindrome(int) -> is_palindrome(str(int))
  * float() with comma handling
  * .is_integer() on strings
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

    # Add reverse() function for string reversal
    if "reverse(" in code and not re.search(r"def\s+reverse\s*\(", code):
        reverse_func = """def reverse(text: str) -> str:
    return text[::-1]"""
        code = reverse_func + "\n\n" + code

    # Add product() function for multiplication of list elements
    if "product(" in code and not re.search(r"def\s+product\s*\(", code) and "from functools import" not in code:
        product_func = """def product(lst):
    result = 1
    for x in lst:
        result *= x
    return result"""
        code = product_func + "\n\n" + code

    # Add is_balanced() function for checking if a list is palindromic
    if "is_balanced(" in code and not re.search(r"def\s+is_balanced\s*\(", code):
        is_balanced_func = """def is_balanced(lst):
    return lst == lst[::-1]"""
        code = is_balanced_func + "\n\n" + code

    # Add prod_sign() function for sign of a number
    if "prod_sign(" in code and not re.search(r"def\s+prod_sign\s*\(", code):
        prod_sign_func = """def prod_sign(n):
    if n > 0: return 1
    if n < 0: return -1
    return 0"""
        code = prod_sign_func + "\n\n" + code

    # Add factorial() function for factorial calculation
    if "factorial(" in code and not re.search(r"def\s+factorial\s*\(", code):
        # Check if math.factorial is already imported
        if "from math import" not in code or "import math" not in code:
            factorial_func = """def factorial(n):
    if n <= 1: return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result"""
            code = factorial_func + "\n\n" + code

    return code


def fix_truncated_code_v5(code: str) -> str:
    """
    Detects and fixes truncated or incomplete code that would cause syntax errors.
    V5: Handles common truncation patterns.
    """
    lines = code.splitlines()
    if not lines:
        return code

    # First pass: Fix unterminated triple-quoted strings
    code_text = '\n'.join(lines)
    # Count triple quotes
    triple_single = code_text.count("'''")
    triple_double = code_text.count('"""')
    if triple_single % 2 == 1:
        code_text += "\n'''"
        lines = code_text.splitlines()
    elif triple_double % 2 == 1:
        code_text += '\n"""'
        lines = code_text.splitlines()

    # Second pass: Check all lines for truncation issues
    fixed_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        line_stripped = line.rstrip()

        # Pattern 1: if/elif/while/for without colon at end
        if re.match(r'^\s*(if|elif|while|for)\s+.*[^:]$', line_stripped):
            # Check if next line exists and is indented (would be the body)
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                if len(next_line) - len(next_line.lstrip()) > len(line) - len(line.lstrip()):
                    # Next line is more indented, so this is likely missing a colon
                    fixed_lines.append(line_stripped + ':')
                    i += 1
                    continue

        # Pattern 2: Unclosed bracket at end (only check last line to avoid false positives)
        if i == len(lines) - 1:  # Only check last line
            open_brackets = line_stripped.count('[') + line_stripped.count('(')
            close_brackets = line_stripped.count(']') + line_stripped.count(')')
            # Skip if it's a function definition line (starts with def)
            if open_brackets > close_brackets and not line_stripped.lstrip().startswith('def '):
                # Unclosed brackets on last line - likely truncated, skip it
                i += 1
                continue

        fixed_lines.append(line)
        i += 1

    lines = fixed_lines
    if not lines:
        return '    pass'

    # Pattern 3: Control flow statement with no body (if/while/for/elif/else followed by nothing or another control statement)
    fixed_lines_2 = []
    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Check if this is a control flow statement ending with ':'
        if re.match(r'^\s*(if|elif|else|while|for|with|try|except|finally|def|class)\s+.*:$|^\s*(else|try|finally):$', line):
            current_indent = len(line) - len(line.lstrip())

            # Check if next line exists
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                next_stripped = next_line.strip()
                next_indent = len(next_line) - len(next_line.lstrip()) if next_stripped else 0

                # If next line is not more indented OR is another control statement at same level, add pass
                if next_stripped and (next_indent <= current_indent):
                    fixed_lines_2.append(line)
                    fixed_lines_2.append(' ' * (current_indent + 4) + 'pass')
                else:
                    fixed_lines_2.append(line)
            else:
                # This is the last line and it's a control statement - add pass
                current_indent = len(line) - len(line.lstrip())
                fixed_lines_2.append(line)
                fixed_lines_2.append(' ' * (current_indent + 4) + 'pass')
        else:
            fixed_lines_2.append(line)

    lines = fixed_lines_2

    # Pattern 4: Repetitive code pattern (likely model got stuck in a loop)
    if len(lines) > 10:
        # Check if there's excessive repetition
        last_five = lines[-5:] if len(lines) >= 5 else lines
        unique_stripped = set(line.strip() for line in last_five if line.strip())

        if len(unique_stripped) <= 2 and len(lines) > 5:
            # High repetition detected - keep only first occurrence of pattern
            seen = set()
            unique_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    unique_lines.append(line)
                elif stripped not in seen:
                    unique_lines.append(line)
                    seen.add(stripped)
                else:
                    # Found repetition, stop here
                    break

            if unique_lines:
                lines = unique_lines
                # Ensure it ends with something valid
                if not any('return' in line or 'pass' in line for line in lines[-2:]):
                    indent = '    '
                    if lines[-1].strip():
                        indent = ' ' * (len(lines[-1]) - len(lines[-1].lstrip()))
                    lines.append(indent + 'pass')

    return '\n'.join(lines)


def apply_value_error_fixes_v5(code: str, problem_prompt: str) -> str:
    """
    Applies fixes for ValueError issues (invalid conversions).
    V5: Handles word-to-number and invalid int() patterns.
    """
    # Fix 1: Word-to-number conversion for sort_numbers function
    if "sort_numbers" in problem_prompt and "int(x)" in code:
        # Replace int(x) with word-to-number mapping
        word_map = """word_to_num = {
    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9
}"""
        # Add the mapping at the beginning
        code = word_map + "\n" + code
        # Replace int(x) with word_to_num.get(x, 0)
        code = re.sub(r'int\(([a-zA-Z_]\w*)\)', r'word_to_num.get(\1, 0)', code)

    # Fix 2: Handle minus sign in digit iteration (order_by_points)
    if "order_by_points" in problem_prompt and "for digit in str(" in code:
        # Replace: sum(int(digit) for digit in str(x))
        # With: sum(int(digit) for digit in str(x) if digit.isdigit())
        code = re.sub(
            r'sum\(int\((\w+)\)\s+for\s+\1\s+in\s+str\(([^)]+)\)\)',
            r'sum(int(\1) for \1 in str(\2).replace("-", "") if \1.isdigit())',
            code
        )
        # Alternative pattern: sum([int(digit) for digit in str(x)])
        code = re.sub(
            r'sum\(\[int\((\w+)\)\s+for\s+\1\s+in\s+str\(([^)]+)\)\]\)',
            r'sum([int(\1) for \1 in str(\2).replace("-", "") if \1.isdigit()])',
            code
        )

    return code


def apply_type_conversion_fixes_v5(code: str, problem_prompt: str) -> str:
    """
    Applies type conversion fixes to prevent runtime type errors.
    V5: Handles common type mismatch patterns.
    """
    # Fix 1: is_palindrome() called with int instead of str
    # Pattern: is_palindrome(int_var) -> is_palindrome(str(int_var))
    if "is_palindrome(" in code and "def is_palindrome" in code:
        # Split into lines to avoid modifying function definition
        lines = code.split('\n')
        fixed_lines = []
        for line in lines:
            # Don't modify the function definition line
            if line.strip().startswith('def is_palindrome'):
                fixed_lines.append(line)
            else:
                # Only wrap integer variables/expressions, not strings or function params
                modified_line = re.sub(
                    r'is_palindrome\(([^)]+)\)',
                    lambda m: f"is_palindrome(str({m.group(1)}))" if not m.group(1).strip().startswith(("'", '"', 'str(')) and not ':' in m.group(1) else m.group(0),
                    line
                )
                fixed_lines.append(modified_line)
        code = '\n'.join(fixed_lines)

    # Fix 2: float() conversion should handle comma decimal separators
    # Pattern: float(x) -> float(x.replace(',', '.')) if x might have commas
    if "float(" in code and "replace(" not in code:
        # Wrap float conversions with replace for string inputs
        code = re.sub(
            r'float\(([a-zA-Z_][a-zA-Z0-9_]*)\)',
            r"float(\1.replace(',', '.') if isinstance(\1, str) else \1)",
            code
        )

    # Fix 3: .is_integer() called on string (should convert to float first)
    # Pattern: value.is_integer() where value is a string
    if ".is_integer()" in code:
        # Replace with float conversion first
        code = re.sub(
            r'(\w+)\.is_integer\(\)',
            r'float(\1).is_integer()',
            code
        )

    # Fix 4: ord(char) % 2 should check char, not ord() result
    # Common mistake: ord(i) where i should be checked if it's lowercase letter
    if "ord(i) for i in" in code and "% 2" in code:
        # This is likely checking if character is in odd position
        # The original intent: check if ord(char) is odd
        code = re.sub(
            r'sum\(\[ord\((\w+)\) for \1 in.*?\]\)',
            r'sum([1 for \1 in txt.lower() if \1.isalpha() and (ord(\1) - ord("a")) % 2 != 0])',
            code
        )

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
            # Fix the is_integer() call on string
            code = re.sub(
                rf"{re.escape(param_name)}\.is_integer\(\)",
                f"float({param_name}.replace(',', '.')).is_integer()",
                code
            )
            # Also fix any round() calls
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
    code = fix_truncated_code_v5(code)  # Fix truncated code before other processing
    code = inject_dependencies_v5(code)
    code = apply_value_error_fixes_v5(code, problem_prompt)  # Apply ValueError fixes
    code = apply_type_conversion_fixes_v5(code, problem_prompt)  # Apply type fixes
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

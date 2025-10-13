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


def create_minimal_v0(problem: str) -> str:
    """
    Expert-framed prompt targeting string manipulation weaknesses.
    Based on failure pattern analysis showing model struggles with:
    - String manipulation (36% of failures - largest category)

    Focused on single expertise area to avoid confusing small models.

    This prompt primes the model with domain expertise to improve
    algorithmic choices for string problems.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Expertise-framed prompt string focused on strings
    """
    return f"""You are an expert Python programmer who specializes in string manipulation. Solve the following problem:

{problem}"""


def create_expert_v00(problem: str) -> str:
    """
    Smart prompt that detects string manipulation problems and applies
    expert framing with DIRECT phrasing: "You are an expert in Python string manipulation"
    (vs expert_v0 which uses: "You are an expert Python programmer who specializes in...")

    String detection heuristics:
    - Function parameters include 'str' type hints
    - Docstring mentions: string, character, text, substring, prefix, suffix,
      case, upper, lower, split, join, reverse, palindrome, etc.

    This targeted approach avoids confusing the model on non-string problems
    while providing expertise framing where it might help.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Direct expert string prompt if string-related, otherwise minimal prompt
    """
    # Convert to lowercase for case-insensitive matching
    problem_lower = problem.lower()

    # String-related keywords to detect string manipulation problems
    string_keywords = [
        'string', 'str', 'text', 'char', 'character',
        'substring', 'prefix', 'suffix',
        'upper', 'lower', 'case', 'capitalize',
        'split', 'join', 'reverse', 'palindrome',
        'vowel', 'consonant', 'digit', 'letter',
        'trim', 'strip', 'replace', 'match',
        'parse', 'format', 'concatenate'
    ]

    # Check if problem involves string manipulation
    is_string_problem = any(keyword in problem_lower for keyword in string_keywords)

    if is_string_problem:
        # Use direct expert string manipulation framing
        return f"""You are an expert in Python string manipulation. Solve the following problem:

{problem}"""
    else:
        # Use minimal prompt (no expert framing)
        return problem.rstrip()


def create_expert_v0(problem: str) -> str:
    """
    Smart prompt that detects string manipulation problems and applies
    expert framing only when relevant. Falls back to minimal prompt otherwise.

    String detection heuristics:
    - Function parameters include 'str' type hints
    - Docstring mentions: string, character, text, substring, prefix, suffix,
      case, upper, lower, split, join, reverse, palindrome, etc.

    This targeted approach avoids confusing the model on non-string problems
    while providing expertise framing where it might help.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Expert string prompt if string-related, otherwise minimal prompt
    """
    # Convert to lowercase for case-insensitive matching
    problem_lower = problem.lower()

    # String-related keywords to detect string manipulation problems
    string_keywords = [
        'string', 'str', 'text', 'char', 'character',
        'substring', 'prefix', 'suffix',
        'upper', 'lower', 'case', 'capitalize',
        'split', 'join', 'reverse', 'palindrome',
        'vowel', 'consonant', 'digit', 'letter',
        'trim', 'strip', 'replace', 'match',
        'parse', 'format', 'concatenate'
    ]

    # Check if problem involves string manipulation
    is_string_problem = any(keyword in problem_lower for keyword in string_keywords)

    if is_string_problem:
        # Use expert string manipulation framing
        return f"""You are an expert Python programmer who specializes in string manipulation. Solve the following problem:

{problem}"""
    else:
        # Use minimal prompt (no expert framing)
        return problem.rstrip()


def create_expert_v1(problem: str) -> str:
    """
    Smart prompt with TWO expert personas:
    - String manipulation expert for PURE string problems
    - List/array operations expert for PURE list/array problems
    - Minimal prompt for everything else (math, mixed, etc.)

    Strict detection heuristics with mutual exclusion:
    - Must have strong signals for the category
    - Must NOT have strong signals for other categories
    - Type hints in function signature are primary determinant

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Expert prompt if clearly string/list problem, otherwise minimal prompt
    """
    # Convert to lowercase for case-insensitive matching
    problem_lower = problem.lower()

    # Get function signature (first line) for type hint analysis
    first_line = problem.split('\n')[0].lower() if '\n' in problem else problem.lower()

    # STRONG string indicators
    strong_string_keywords = [
        'string', 'text', 'char', 'character',
        'substring', 'prefix', 'suffix',
        'upper', 'lower', 'case', 'capitalize',
        'palindrome', 'vowel', 'consonant',
        'concatenate'
    ]

    # STRONG list/array indicators
    strong_list_keywords = [
        'list', 'array', 'sequence',
        'sort', 'sorted', 'filter',
        'subarray', 'sublist'
    ]

    # Check type hints first (most reliable)
    has_str_return = '-> str' in first_line
    has_list_return = '-> list' in first_line or 'list[' in first_line
    has_str_param = 'str:' in first_line or 'str,' in first_line or 'str)' in first_line
    has_list_param = 'list[' in first_line or 'list:' in first_line or 'list,' in first_line

    # Count strong keyword matches
    string_score = sum(1 for kw in strong_string_keywords if kw in problem_lower)
    list_score = sum(1 for kw in strong_list_keywords if kw in problem_lower)

    # Add type hint bonuses
    if has_str_return or has_str_param:
        string_score += 3
    if has_list_return or has_list_param:
        list_score += 3

    # Decision logic: must have clear winner with sufficient score
    # Require score >= 2 to trigger expert, and must be clearly dominant
    if string_score >= 2 and string_score > list_score:
        # String manipulation expert (direct phrasing)
        return f"""You are an expert in Python string manipulation. Solve the following problem:

{problem}"""
    elif list_score >= 2 and list_score > string_score:
        # List/array operations expert (direct phrasing)
        return f"""You are an expert in Python list and array operations. Solve the following problem:

{problem}"""
    else:
        # Use minimal prompt (no expert framing)
        # This includes: math problems, mixed problems, unclear problems
        return problem.rstrip()


def create_expert_v2(problem: str) -> str:
    """
    Smart prompt with THREE expert personas:
    - String manipulation expert for PURE string problems
    - List/array operations expert for PURE list/array problems
    - Sorting expert for PURE sorting problems
    - Minimal prompt for everything else (math, mixed, etc.)

    Targets the top 3 failure categories:
    - String manipulation: 36% of failures
    - List/array operations: 27% of failures
    - Sorting/ordering: 21% of failures

    Strict detection with mutual exclusion to minimize false positives.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Expert prompt if clearly matches a category, otherwise minimal prompt
    """
    # Convert to lowercase for case-insensitive matching
    problem_lower = problem.lower()

    # Get function signature (first line) for type hint analysis
    first_line = problem.split('\n')[0].lower() if '\n' in problem else problem.lower()

    # STRONG string indicators
    strong_string_keywords = [
        'string', 'text', 'char', 'character',
        'substring', 'prefix', 'suffix',
        'upper', 'lower', 'case', 'capitalize',
        'palindrome', 'vowel', 'consonant',
        'concatenate'
    ]

    # STRONG list/array indicators
    strong_list_keywords = [
        'list', 'array', 'sequence',
        'filter', 'map',
        'subarray', 'sublist',
        'append', 'remove', 'insert'
    ]

    # STRONG sorting indicators
    strong_sorting_keywords = [
        'sort', 'sorted', 'order', 'ordering',
        'ascending', 'descending',
        'arrange', 'rank'
    ]

    # Check type hints first (most reliable)
    has_str_return = '-> str' in first_line
    has_list_return = '-> list' in first_line or 'list[' in first_line
    has_str_param = 'str:' in first_line or 'str,' in first_line or 'str)' in first_line
    has_list_param = 'list[' in first_line or 'list:' in first_line or 'list,' in first_line

    # Count strong keyword matches
    string_score = sum(1 for kw in strong_string_keywords if kw in problem_lower)
    list_score = sum(1 for kw in strong_list_keywords if kw in problem_lower)
    sorting_score = sum(1 for kw in strong_sorting_keywords if kw in problem_lower)

    # Add type hint bonuses
    if has_str_return or has_str_param:
        string_score += 3
    if has_list_return or has_list_param:
        list_score += 2  # Reduced because sorting also works with lists

    # Sorting problems often have list return types
    if sorting_score > 0 and has_list_return:
        sorting_score += 3

    # Decision logic: must have clear winner with sufficient score
    # Require score >= 2 to trigger expert, and must be clearly dominant
    max_score = max(string_score, list_score, sorting_score)

    # Only use expert if there's a clear winner (no ties or close scores)
    if max_score >= 2:
        scores = [string_score, list_score, sorting_score]
        scores.sort(reverse=True)

        # Require clear dominance (winner must be > second place)
        if scores[0] > scores[1]:
            if string_score == max_score:
                # String manipulation expert
                return f"""You are an expert Python programmer who specializes in string manipulation. Solve the following problem:

{problem}"""
            elif sorting_score == max_score:
                # Sorting expert
                return f"""You are an expert Python programmer who specializes in sorting and ordering algorithms. Solve the following problem:

{problem}"""
            elif list_score == max_score:
                # List/array operations expert
                return f"""You are an expert Python programmer who specializes in list and array operations. Solve the following problem:

{problem}"""

    # Use minimal prompt (no expert framing)
    # This includes: math problems, mixed problems, unclear problems, ties
    return problem.rstrip()


def create_example_v0(problem: str) -> str:
    """
    Smart prompt that adds ONE relevant example based on problem type:
    - String problems → string manipulation example
    - List/array problems → list/array example
    - Sorting problems → sorting example
    - Otherwise → minimal (no example)

    Uses same detection logic as expert_v2 for consistency.

    Args:
        problem: The function signature and docstring from HumanEval

    Returns:
        Minimal prompt + relevant example if detected, otherwise just minimal
    """
    # Convert to lowercase for case-insensitive matching
    problem_lower = problem.lower()

    # Get function signature (first line) for type hint analysis
    first_line = problem.split('\n')[0].lower() if '\n' in problem else problem.lower()

    # STRONG string indicators
    strong_string_keywords = [
        'string', 'text', 'char', 'character',
        'substring', 'prefix', 'suffix',
        'upper', 'lower', 'case', 'capitalize',
        'palindrome', 'vowel', 'consonant',
        'concatenate'
    ]

    # STRONG list/array indicators
    strong_list_keywords = [
        'list', 'array', 'sequence',
        'filter', 'map',
        'subarray', 'sublist',
        'append', 'remove', 'insert'
    ]

    # STRONG sorting indicators
    strong_sorting_keywords = [
        'sort', 'sorted', 'order', 'ordering',
        'ascending', 'descending',
        'arrange', 'rank'
    ]

    # Check type hints first (most reliable)
    has_str_return = '-> str' in first_line
    has_list_return = '-> list' in first_line or 'list[' in first_line
    has_str_param = 'str:' in first_line or 'str,' in first_line or 'str)' in first_line
    has_list_param = 'list[' in first_line or 'list:' in first_line or 'list,' in first_line

    # Count strong keyword matches
    string_score = sum(1 for kw in strong_string_keywords if kw in problem_lower)
    list_score = sum(1 for kw in strong_list_keywords if kw in problem_lower)
    sorting_score = sum(1 for kw in strong_sorting_keywords if kw in problem_lower)

    # Add type hint bonuses
    if has_str_return or has_str_param:
        string_score += 3
    if has_list_return or has_list_param:
        list_score += 2

    # Sorting problems often have list return types
    if sorting_score > 0 and has_list_return:
        sorting_score += 3

    # Decision logic: must have clear winner with sufficient score
    max_score = max(string_score, list_score, sorting_score)

    # Only add example if there's a clear winner (no ties or close scores)
    if max_score >= 2:
        scores = [string_score, list_score, sorting_score]
        scores.sort(reverse=True)

        # Require clear dominance (winner must be > second place)
        if scores[0] > scores[1]:
            if string_score == max_score:
                # String manipulation example
                example = """Example - String manipulation:
def reverse_text(s: str) -> str:
    return s[::-1]

"""
                return example + problem
            elif sorting_score == max_score:
                # Sorting example
                example = """Example - Sorting:
def sort_numbers(nums: List[int]) -> List[int]:
    return sorted(nums)

"""
                return example + problem
            elif list_score == max_score:
                # List/array example
                example = """Example - List operations:
def filter_even(nums: List[int]) -> List[int]:
    return [n for n in nums if n % 2 == 0]

"""
                return example + problem

    # Use minimal prompt (no example)
    return problem.rstrip()


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

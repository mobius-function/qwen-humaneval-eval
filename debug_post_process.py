#!/usr/bin/env python3
"""Debug the post-processing with actual model output."""

from prompts.post_process_v5 import post_process_code_v5

# The RAW model output from HumanEval/0
raw_code = """    for i in range(len(numbers) - 1):
        if abs(numbers[i] - numbers[i + 1]) < threshold:
            return True
    return False"""

prompt = """from typing import List


def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    \"\"\"
"""

print("RAW MODEL OUTPUT:")
print(repr(raw_code))
print()

result = post_process_code_v5(raw_code, prompt)

print("POST-PROCESSED OUTPUT:")
print(repr(result))
print()

print("FULL CODE:")
full_code = prompt + result
print(full_code)
print()

# Try to exec it
try:
    exec(full_code)
    print("✓ Code is syntactically valid!")
except SyntaxError as e:
    print(f"✗ Syntax error: {e}")

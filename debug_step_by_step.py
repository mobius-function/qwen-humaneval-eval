#!/usr/bin/env python3
"""Step by step debugging of the post-processing."""

import textwrap
import re

# The RAW model output
raw_code = """    for i in range(len(numbers) - 1):
        if abs(numbers[i] - numbers[i + 1]) < threshold:
            return True
    return False"""

print("=" * 80)
print("STEP 0: RAW MODEL OUTPUT")
print("=" * 80)
for i, line in enumerate(raw_code.split('\n'), 1):
    spaces = len(line) - len(line.lstrip())
    print(f"Line {i} ({spaces:2d} spaces): {repr(line)}")
print()

# Step 1: clean_model_output_v5
print("=" * 80)
print("STEP 1: clean_model_output_v5 - Remove markdown and dedent")
print("=" * 80)
code = re.sub(r'\A```(?:python)?\s*\n?|\n?```\Z', '', raw_code.strip())
print("After markdown removal (no change, no markdown present):")
for i, line in enumerate(code.split('\n'), 1):
    spaces = len(line) - len(line.lstrip())
    print(f"Line {i} ({spaces:2d} spaces): {repr(line)}")
print()

code = textwrap.dedent(code).strip()
print("After textwrap.dedent() and strip():")
for i, line in enumerate(code.split('\n'), 1):
    spaces = len(line) - len(line.lstrip())
    print(f"Line {i} ({spaces:2d} spaces): {repr(line)}")
print()

# The dedented code should look like this:
expected = """for i in range(len(numbers) - 1):
    if abs(numbers[i] - numbers[i + 1]) < threshold:
        return True
return False"""

print("This is CORRECT Python code with proper relative indentation!")
print("Now the question is: how do we indent this to be inside a function?")
print()

# The function signature ends with "):
#     """
# So after the docstring there's already a newline. We need to add 4 spaces to each line.

print("=" * 80)
print("SOLUTION: Add 4 spaces to EVERY line (preserving relative indentation)")
print("=" * 80)

indented = '\n'.join('    ' + line if line.strip() else line for line in code.split('\n'))
print(indented)
print()

# But wait, the function prompt ends with """ and TWO newlines
# So we should have:\n\n{indented_code}

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

print("=" * 80)
print("TESTING: prompt + '\\n' + indented_code")
print("=" * 80)
full_code = prompt + '\n' + indented
print(full_code)

try:
    exec(full_code)
    print("\n✓✓✓ SUCCESS! Code is syntactically valid!")
except SyntaxError as e:
    print(f"\n✗✗✗ FAILED: {e}")

#!/usr/bin/env python3
"""Test with the ACTUAL broken output from the log."""

import textwrap

# The ACTUAL output from the log (lines 30-33)
# Notice the inconsistent indentation!
raw_code = """    for i in range(len(numbers) - 1):
            if abs(numbers[i] - numbers[i + 1]) < threshold:
                return True
        return False"""

print("Original code (notice inconsistent indentation!):")
for i, line in enumerate(raw_code.split('\n'), 1):
    print(f"Line {i}: {len(line) - len(line.lstrip())} spaces: {repr(line)}")
print()

# Try dedent
code = textwrap.dedent(raw_code).strip()
print("After dedent:")
for i, line in enumerate(code.split('\n'), 1):
    print(f"Line {i}: {len(line) - len(line.lstrip())} spaces: {repr(line)}")

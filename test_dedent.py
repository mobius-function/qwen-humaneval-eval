#!/usr/bin/env python3
"""Test the dedenting behavior."""

import textwrap

# Simulate the raw model output (already has 4 spaces of indentation)
raw_code = """    for i in range(len(numbers) - 1):
        if abs(numbers[i] - numbers[i + 1]) < threshold:
            return True
    return False"""

print("Original code:")
print(repr(raw_code))
print(raw_code)
print()

# First dedent (in clean_model_output_v5)
code = textwrap.dedent(raw_code).strip()
print("After first dedent:")
print(repr(code))
print(code)
print()

# Second dedent (in post_process_code_v5)
code = textwrap.dedent(code).strip()
print("After second dedent:")
print(repr(code))
print(code)
print()

# Re-indent
indented_lines = []
for line in code.splitlines():
    if line.strip():
        indented_lines.append("    " + line)
    else:
        indented_lines.append(line)

final_code = "\n".join(indented_lines)
print("After re-indenting:")
print(repr(final_code))
print(final_code)

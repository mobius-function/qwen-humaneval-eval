"""Post-processing for fewshot_v1_cot chat strategy.

This CoT strategy may generate reasoning text + code, so we:
1. Extract the last code block (final implementation)
2. Handle mixed reasoning/code text
3. Extract only the function body
"""

import ast
import re


def extract_function_body(code: str, entry_point: str = None) -> str:
    """
    Extract only the function body from generated code.

    Args:
        code: Complete generated code
        entry_point: Function name to extract (if None, extracts first function)

    Returns:
        Only the function body (indented code inside the function)
    """
    try:
        tree = ast.parse(code)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        if not functions:
            return code.strip()

        target_func = None
        if entry_point:
            for func in functions:
                if func.name == entry_point:
                    target_func = func
                    break

        if target_func is None:
            target_func = functions[0]

        lines = code.split('\n')
        body_start_idx = None

        for i, node in enumerate(target_func.body):
            if i == 0 and isinstance(node, ast.Expr):
                if isinstance(node.value, (ast.Str, ast.Constant)):
                    continue
            body_start_idx = i
            body_start_line = node.lineno - 1
            break

        if body_start_idx is None or body_start_line is None:
            return "pass"

        last_node = target_func.body[-1]
        body_end_line = last_node.end_lineno if hasattr(last_node, 'end_lineno') else last_node.lineno

        body_lines = lines[body_start_line:body_end_line]

        if not body_lines:
            return "pass"

        min_indent = float('inf')
        for line in body_lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        if min_indent > 0 and min_indent != float('inf'):
            dedented_lines = []
            for line in body_lines:
                if line.strip():
                    dedented_lines.append(' ' * 4 + line[min_indent:])
                else:
                    dedented_lines.append(line)
            return '\n'.join(dedented_lines)
        else:
            return '\n'.join(' ' * 4 + line if line.strip() else line for line in body_lines)

    except SyntaxError:
        return code.strip()


def post_process(raw_code: str, problem_prompt: str = None, entry_point: str = None) -> str:
    """
    Post-process code from fewshot_v1_cot chat strategy.

    CoT may produce:
    - Reasoning text followed by code
    - Multiple code blocks (drafts + final)
    - Complete function definitions

    Handles:
    1. Extracting the LAST code block (final implementation after reasoning)
    2. Removing markdown code fences
    3. Extracting only the function body from complete function definitions

    Args:
        raw_code: Raw model output (may include reasoning + code)
        problem_prompt: Original problem prompt (unused, for compatibility)
        entry_point: Function name to extract

    Returns:
        Cleaned function body only
    """
    code = raw_code

    # Step 1: Handle multiple code blocks (take the LAST one - final implementation)
    if '```python' in code:
        matches = re.findall(r'```python\s*(.*?)\s*```', code, re.DOTALL)
        if matches:
            code = matches[-1]  # Last code block is likely the refined solution
    elif '```' in code:
        matches = re.findall(r'```\s*(.*?)\s*```', code, re.DOTALL)
        if matches:
            code = matches[-1]
        else:
            code = code.replace('```', '')

    code = code.strip()

    # Step 2: If no code blocks found, try to extract code from mixed text
    if not code or len(code) < 10:
        lines = raw_code.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            # Detect start of code
            if line.strip().startswith(('def ', 'class ', 'for ', 'if ', 'while ', 'return ', 'import ')):
                in_code = True

            if in_code:
                code_lines.append(line)

                # Stop at reasoning text indicators
                if any(indicator in line.lower() for indicator in ['step ', 'algorithm:', 'approach:', 'explanation:']):
                    break

        if code_lines:
            code = '\n'.join(code_lines).strip()

    # Step 3: Extract function body
    body = extract_function_body(code, entry_point)

    return body

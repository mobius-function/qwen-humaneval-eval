"""Post-processing for chat mode - extracts function body from complete function definitions."""

import ast
import re


def extract_function_body(code: str, entry_point: str = None) -> str:
    """
    Extract only the function body from generated code that includes the full function.

    This handles the case where chat/instruct models generate complete functions
    including imports, signature, and docstring, but we only need the body.

    Args:
        code: Complete generated code
        entry_point: Function name to extract (if None, extracts first function)

    Returns:
        Only the function body (indented code inside the function)
    """
    try:
        # Parse the code as AST
        tree = ast.parse(code)

        # Find all function definitions
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        if not functions:
            # No function found, return cleaned code as-is
            return code.strip()

        # If entry_point specified, find that function; otherwise use first function
        target_func = None
        if entry_point:
            for func in functions:
                if func.name == entry_point:
                    target_func = func
                    break

        if target_func is None:
            target_func = functions[0]

        # Extract the function body
        # Get the source lines
        lines = code.split('\n')

        # Find the function definition line
        func_start_line = target_func.lineno - 1

        # Find where the function body starts (after signature and docstring)
        body_start_idx = None
        for i, node in enumerate(target_func.body):
            # Skip docstring (first node if it's an Expr with a Str/Constant)
            if i == 0 and isinstance(node, ast.Expr):
                if isinstance(node.value, (ast.Str, ast.Constant)):
                    continue  # Skip docstring
            # This is the first actual statement
            body_start_idx = i
            body_start_line = node.lineno - 1
            break

        if body_start_idx is None or body_start_line is None:
            # Function has only docstring, no body
            return "pass"

        # Find where the function ends
        last_node = target_func.body[-1]
        body_end_line = last_node.end_lineno if hasattr(last_node, 'end_lineno') else last_node.lineno

        # Extract the body lines
        body_lines = lines[body_start_line:body_end_line]

        if not body_lines:
            return "pass"

        # Find the minimum indentation
        min_indent = float('inf')
        for line in body_lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                min_indent = min(min_indent, indent)

        # Remove the base indentation to get relative indentation
        if min_indent > 0 and min_indent != float('inf'):
            dedented_lines = []
            for line in body_lines:
                if line.strip():
                    dedented_lines.append(' ' * 4 + line[min_indent:])  # Re-indent with 4 spaces
                else:
                    dedented_lines.append(line)
            return '\n'.join(dedented_lines)
        else:
            # Add 4-space indentation
            return '\n'.join(' ' * 4 + line if line.strip() else line for line in body_lines)

    except SyntaxError:
        # If parsing fails, return cleaned code
        return code.strip()


def post_process_chat(raw_code: str, problem_prompt: str = None, entry_point: str = None) -> str:
    """
    Post-process code from chat/instruct models.

    Handles:
    1. Removing reasoning text (for CoT prompts)
    2. Removing markdown code fences
    3. Extracting only the function body from complete function definitions
    4. Cleaning up the code

    Args:
        raw_code: Raw model output (may include reasoning + code)
        problem_prompt: Original problem prompt (unused, for compatibility)
        entry_point: Function name to extract

    Returns:
        Cleaned function body only
    """
    # Step 1: Handle multiple code blocks (CoT might produce reasoning + code)
    # If there are multiple code blocks, take the LAST one (final implementation)
    code = raw_code

    if '```python' in code:
        # Find all Python code blocks
        matches = re.findall(r'```python\s*(.*?)\s*```', code, re.DOTALL)
        if matches:
            # Take the last code block (most likely the final implementation)
            code = matches[-1]
    elif '```' in code:
        # Find all code blocks (without language specifier)
        matches = re.findall(r'```\s*(.*?)\s*```', code, re.DOTALL)
        if matches:
            # Take the last code block
            code = matches[-1]
        else:
            # Simple case: just remove backticks
            code = code.replace('```', '')

    code = code.strip()

    # Step 2: If no code blocks found, try to extract code from mixed text
    # Look for lines that look like Python code (indented, have Python keywords)
    if not code or len(code) < 10:
        # Try to find Python code in the raw output
        lines = raw_code.split('\n')
        code_lines = []
        in_code = False

        for line in lines:
            # Detect start of code (indentation or Python keywords)
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

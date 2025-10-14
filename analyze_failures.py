#!/usr/bin/env python3
"""
Analyze failure patterns in the HumanEval failures log.
"""

import re
from collections import Counter, defaultdict

def parse_failures(log_file):
    """Parse the failure log and extract structured information."""
    with open(log_file, 'r') as f:
        content = f.read()

    # Split by task separators
    tasks = re.split(r'\[(\d+)/\d+\] Task: (HumanEval/\d+)', content)

    failures = []
    for i in range(1, len(tasks), 3):
        if i+2 >= len(tasks):
            break
        task_num = tasks[i]
        task_id = tasks[i+1]
        task_content = tasks[i+2]

        # Extract error type
        error_match = re.search(r'^Error: (.*)$', task_content, re.MULTILINE)
        error = error_match.group(1).strip() if error_match else ""

        # Extract input docstring
        docstring_match = re.search(
            r'INPUT DOCSTRING:.*?^-+\s*$(.*?)^GENERATED CODE',
            task_content,
            re.MULTILINE | re.DOTALL
        )
        docstring = docstring_match.group(1).strip() if docstring_match else ""

        # Extract function name
        func_match = re.search(r'def (\w+)\(', docstring)
        func_name = func_match.group(1) if func_match else ""

        # Extract generated code
        code_match = re.search(
            r'POST-PROCESSED CODE:.*?^-+\s*$(.*?)^={70,}',
            task_content,
            re.MULTILINE | re.DOTALL
        )
        generated_code = code_match.group(1).strip() if code_match else ""

        failures.append({
            'task_num': task_num,
            'task_id': task_id,
            'function_name': func_name,
            'error': error,
            'docstring': docstring,
            'generated_code': generated_code
        })

    return failures

def categorize_failure(failure):
    """Categorize the type of failure."""
    code = failure['generated_code']
    error = failure['error']

    categories = []

    # Model completely failed to generate code
    if 'pass' in code and code.count('\n') < 5:
        categories.append('NO_IMPLEMENTATION')

    # Model generated only comments (repetitive pattern)
    if '# Your code here' in code or '# Write your code here' in code:
        comment_lines = [line for line in code.split('\n') if line.strip().startswith('#')]
        code_lines = [line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')]
        if len(comment_lines) > 10 and len(code_lines) < 5:
            categories.append('COMMENT_REPETITION')

    # Model generated repetitive if statements (degeneration)
    if code.count('if ') > 10:
        lines = code.split('\n')
        if_lines = [l for l in lines if 'if ' in l]
        if len(set(if_lines)) < len(if_lines) / 2:
            categories.append('IF_REPETITION')

    # Syntax errors
    if 'expected' in error.lower() or 'invalid' in error.lower():
        categories.append('SYNTAX_ERROR')

    # Index errors
    if 'index out of range' in error.lower():
        categories.append('INDEX_ERROR')

    # Undefined name errors
    if 'is not defined' in error:
        categories.append('UNDEFINED_NAME')

    # Test failures (logic errors)
    if 'test' in error.lower() or 'assert' in error.lower():
        categories.append('LOGIC_ERROR')

    # Check for specific logic errors
    if code and not categories:
        # Wrong algorithm - checking requirements vs implementation
        if 'sum' in failure['docstring'].lower() and 'sum' not in code:
            categories.append('MISSING_OPERATION')

        # Simplistic implementation (just checks one condition)
        if len(code.split('\n')) < 10 and 'return' in code:
            categories.append('OVERSIMPLIFIED')

    # No error message but failed (silent failure)
    if not error and not categories:
        categories.append('SILENT_FAILURE')

    return categories if categories else ['OTHER']

def analyze_logic_errors(failures):
    """Deep dive into logic errors to find patterns."""
    logic_errors = [f for f in failures if 'LOGIC_ERROR' in categorize_failure(f)]

    patterns = {
        'wrong_algorithm': [],
        'off_by_one': [],
        'missing_edge_case': [],
        'misunderstood_requirements': [],
        'incorrect_loop': [],
        'wrong_return_type': [],
    }

    for f in logic_errors:
        code = f['generated_code']
        doc = f['docstring'].lower()

        # Check for misunderstood requirements
        if 'average' in doc and '(n + m) // 2' in code and 'sum' in doc:
            patterns['wrong_algorithm'].append(f)

        # Check for missing filtering/edge cases
        if 'filter' in doc or 'only' in doc or 'between' in doc:
            patterns['missing_edge_case'].append(f)

        # Check for wrong loop bounds
        if 'range(len(' in code and 'arr[i+1]' in code:
            patterns['off_by_one'].append(f)

        # Check for string formatting issues
        if 'string' in doc and 'format' in doc.lower():
            patterns['misunderstood_requirements'].append(f)

    return patterns

def main():
    log_file = '/teamspace/studios/this_studio/qwen-humaneval-eval/logs/fewshot_none_failures.log'

    print("Parsing failure log...")
    failures = parse_failures(log_file)
    print(f"Found {len(failures)} failed tasks\n")

    # Categorize all failures
    print("="*80)
    print("FAILURE CATEGORIES")
    print("="*80)
    category_counts = Counter()
    failure_by_category = defaultdict(list)

    for f in failures:
        cats = categorize_failure(f)
        for cat in cats:
            category_counts[cat] += 1
            failure_by_category[cat].append(f)

    for cat, count in category_counts.most_common():
        print(f"{cat:30s}: {count:3d} ({count/len(failures)*100:.1f}%)")

    print("\n" + "="*80)
    print("TOP ERROR PATTERNS WITH EXAMPLES")
    print("="*80)

    # Show examples for each major category
    top_categories = [cat for cat, _ in category_counts.most_common(5)]

    for cat in top_categories:
        examples = failure_by_category[cat][:3]  # Show up to 3 examples
        print(f"\n{'='*80}")
        print(f"Category: {cat}")
        print(f"Count: {category_counts[cat]}")
        print(f"{'='*80}")

        for i, ex in enumerate(examples, 1):
            print(f"\nExample {i}: {ex['task_id']} - {ex['function_name']}")
            print(f"Error: {ex['error'][:100]}")
            print(f"\nDocstring (first 300 chars):")
            print(ex['docstring'][:300].replace('\n', '\n  '))
            print(f"\nGenerated Code (first 400 chars):")
            print(ex['generated_code'][:400].replace('\n', '\n  '))
            print("-"*80)

    # Analyze logic errors specifically
    print("\n" + "="*80)
    print("LOGIC ERROR PATTERNS")
    print("="*80)

    logic_patterns = analyze_logic_errors(failures)
    for pattern_name, pattern_failures in logic_patterns.items():
        if pattern_failures:
            print(f"\n{pattern_name}: {len(pattern_failures)} occurrences")
            for f in pattern_failures[:2]:
                print(f"  - {f['task_id']}: {f['function_name']}")

if __name__ == '__main__':
    main()

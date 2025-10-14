#!/usr/bin/env python3
"""
Comprehensive analysis of fewshot_v2_post_v5 results to identify failure patterns
and suggest prompt improvements.
"""

import re
from collections import Counter, defaultdict
import json

def parse_log_file(log_file):
    """Parse the log file and extract all test cases."""
    with open(log_file, 'r') as f:
        content = f.read()

    # Extract overall stats
    stats_match = re.search(r'ALL TEST CASES - (\d+) passed, (\d+) failed out of (\d+)', content)
    if stats_match:
        passed = int(stats_match.group(1))
        failed = int(stats_match.group(2))
        total = int(stats_match.group(3))
    else:
        passed = failed = total = 0

    # Split into individual test cases
    cases = re.split(r'\[(\d+)/\d+\] HumanEval/(\d+) - ([✓✗]) (PASSED|FAILED)', content)

    results = []
    for i in range(1, len(cases), 5):
        if i+4 >= len(cases):
            break

        test_num = cases[i]
        humaneval_num = cases[i+1]
        status_symbol = cases[i+2]
        status = cases[i+3]
        test_content = cases[i+4]

        # Extract docstring
        docstring_match = re.search(
            r'INPUT DOCSTRING:.*?^-+\s*$(.*?)^GENERATED CODE',
            test_content,
            re.MULTILINE | re.DOTALL
        )
        docstring = docstring_match.group(1).strip() if docstring_match else ""

        # Extract function name
        func_match = re.search(r'def (\w+)\(', docstring)
        func_name = func_match.group(1) if func_match else ""

        # Extract raw generated code
        raw_code_match = re.search(
            r'GENERATED CODE \(RAW MODEL OUTPUT\):.*?^-+\s*$(.*?)^POST-PROCESSED CODE',
            test_content,
            re.MULTILINE | re.DOTALL
        )
        raw_code = raw_code_match.group(1).strip() if raw_code_match else ""

        # Extract post-processed code
        processed_code_match = re.search(
            r'POST-PROCESSED CODE:.*?^-+\s*$(.*?)^(?:ERROR:|={70,})',
            test_content,
            re.MULTILINE | re.DOTALL
        )
        processed_code = processed_code_match.group(1).strip() if processed_code_match else ""

        # Extract error if present
        error_match = re.search(r'ERROR:.*?^-+\s*$(.*?)^={70,}', test_content, re.MULTILINE | re.DOTALL)
        error = error_match.group(1).strip() if error_match else ""

        results.append({
            'test_num': int(test_num),
            'humaneval_num': int(humaneval_num),
            'status': status,
            'function_name': func_name,
            'docstring': docstring,
            'raw_code': raw_code,
            'processed_code': processed_code,
            'error': error
        })

    return {
        'stats': {'passed': passed, 'failed': failed, 'total': total},
        'results': results
    }

def categorize_problem(docstring, func_name):
    """Categorize the problem type based on docstring."""
    doc_lower = docstring.lower()

    categories = []

    # String manipulation
    if any(word in doc_lower for word in ['string', 'character', 'substring', 'prefix', 'suffix', 'palindrome', 'vowel', 'consonant']):
        categories.append('string_manipulation')

    # List operations
    if any(word in doc_lower for word in ['list', 'array', 'sort', 'filter', 'remove', 'insert', 'append']):
        categories.append('list_operations')

    # Math operations
    if any(word in doc_lower for word in ['sum', 'product', 'multiply', 'divide', 'factorial', 'prime', 'gcd', 'lcm', 'fibonacci']):
        categories.append('math_operations')

    # Data structures
    if any(word in doc_lower for word in ['dictionary', 'dict', 'hash', 'map', 'set', 'tuple']):
        categories.append('data_structures')

    # Parsing/Tokenization
    if any(word in doc_lower for word in ['parse', 'token', 'bracket', 'paren', 'expression']):
        categories.append('parsing')

    # Algorithm/Logic
    if any(word in doc_lower for word in ['find', 'search', 'match', 'compare', 'check', 'validate']):
        categories.append('logic_algorithm')

    # Recursion
    if 'recurs' in doc_lower:
        categories.append('recursion')

    # Edge cases
    if any(word in doc_lower for word in ['empty', 'none', 'null', 'edge']):
        categories.append('edge_cases')

    return categories if categories else ['uncategorized']

def categorize_failure(result):
    """Categorize the type of failure with detailed analysis."""
    code = result['processed_code']
    raw_code = result['raw_code']
    error = result['error']
    docstring = result['docstring']

    categories = []

    # 1. Placeholder/No implementation
    if not code or code.strip() == '':
        categories.append('EMPTY_CODE')
    elif 'pass' in code and len([l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]) <= 2:
        categories.append('PLACEHOLDER_PASS')
    elif '# Your code here' in code or '# Write your code here' in code or 'TODO' in code:
        categories.append('PLACEHOLDER_COMMENT')

    # 2. Repetition issues (model degeneration)
    lines = code.split('\n')
    if len(lines) > 15:
        # Comment repetition
        comment_lines = [l.strip() for l in lines if l.strip().startswith('#')]
        if len(comment_lines) > 10:
            unique_comments = len(set(comment_lines))
            if unique_comments < len(comment_lines) * 0.3:  # Less than 30% unique
                categories.append('COMMENT_REPETITION')

        # If statement repetition
        if_lines = [l for l in lines if 'if ' in l]
        if len(if_lines) > 8:
            unique_ifs = len(set(if_lines))
            if unique_ifs < len(if_lines) * 0.5:
                categories.append('IF_REPETITION')

        # General line repetition
        code_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]
        if len(code_lines) > 10:
            unique_code = len(set(code_lines))
            if unique_code < len(code_lines) * 0.4:
                categories.append('CODE_REPETITION')

    # 3. Syntax errors
    if error:
        if 'SyntaxError' in error or 'invalid syntax' in error:
            categories.append('SYNTAX_ERROR')
        elif 'IndentationError' in error:
            categories.append('INDENTATION_ERROR')
        elif 'NameError' in error or 'is not defined' in error:
            categories.append('UNDEFINED_NAME')
        elif 'IndexError' in error or 'index out of range' in error:
            categories.append('INDEX_ERROR')
        elif 'KeyError' in error:
            categories.append('KEY_ERROR')
        elif 'AttributeError' in error:
            categories.append('ATTRIBUTE_ERROR')
        elif 'TypeError' in error:
            categories.append('TYPE_ERROR')
        elif 'ValueError' in error:
            categories.append('VALUE_ERROR')
        elif 'RecursionError' in error or 'maximum recursion' in error:
            categories.append('RECURSION_ERROR')
        elif 'AssertionError' in error or 'assert' in error.lower():
            categories.append('ASSERTION_FAILED')

    # 4. Logic errors (if test failed but no exception)
    if result['status'] == 'FAILED' and not categories:
        categories.append('LOGIC_ERROR')

    # 5. Analyze specific logic patterns
    if code and 'LOGIC_ERROR' in categories or 'ASSERTION_FAILED' in categories:
        doc_lower = docstring.lower()

        # Wrong algorithm
        if 'sort' in doc_lower and 'sort' not in code.lower() and 'sorted' not in code.lower():
            categories.append('MISSING_SORT')

        # Off-by-one errors
        if 'range(len(' in code:
            if 'arr[i+1]' in code or 'lst[i+1]' in code or '[i + 1]' in code:
                # Check if there's proper bound checking
                if 'len(' not in code[code.index('range(len('):]:
                    categories.append('POTENTIAL_OFF_BY_ONE')

        # Misunderstood requirements
        if 'return list' in doc_lower or 'return a list' in doc_lower:
            if 'return ' in code:
                return_lines = [l for l in lines if 'return' in l and not l.strip().startswith('#')]
                if return_lines and not any('[' in l or 'list(' in l for l in return_lines):
                    categories.append('WRONG_RETURN_TYPE')

        # Missing edge case handling
        if 'empty' in doc_lower or 'none' in doc_lower:
            if 'if not' not in code and 'if len(' not in code and 'if ' not in code[:50]:
                categories.append('MISSING_EDGE_CASE')

        # Incomplete loop
        if 'for ' in code or 'while ' in code:
            # Check if loop body is trivial
            for_matches = re.finditer(r'for .+?:', code)
            for match in for_matches:
                start = match.end()
                # Get indented content after for
                following = code[start:start+100]
                if 'pass' in following or not following.strip():
                    categories.append('INCOMPLETE_LOOP')
                    break

    return categories if categories else ['UNKNOWN_FAILURE']

def analyze_specific_failures(results):
    """Deep dive into specific failure patterns."""
    failed_results = [r for r in results if r['status'] == 'FAILED']

    patterns = {
        'placeholder_variations': [],
        'off_by_one_examples': [],
        'wrong_algorithm': [],
        'misunderstood_requirements': [],
        'edge_case_failures': [],
        'loop_issues': [],
        'repetition_issues': [],
    }

    for result in failed_results:
        code = result['processed_code']
        doc = result['docstring']
        categories = categorize_failure(result)

        # Collect examples for each pattern
        if any(cat in categories for cat in ['PLACEHOLDER_PASS', 'PLACEHOLDER_COMMENT', 'EMPTY_CODE']):
            patterns['placeholder_variations'].append(result)

        if 'POTENTIAL_OFF_BY_ONE' in categories or 'INDEX_ERROR' in categories:
            patterns['off_by_one_examples'].append(result)

        if 'MISSING_SORT' in categories or 'WRONG_RETURN_TYPE' in categories:
            patterns['wrong_algorithm'].append(result)

        if 'WRONG_RETURN_TYPE' in categories:
            patterns['misunderstood_requirements'].append(result)

        if 'MISSING_EDGE_CASE' in categories:
            patterns['edge_case_failures'].append(result)

        if 'INCOMPLETE_LOOP' in categories:
            patterns['loop_issues'].append(result)

        if any(cat in categories for cat in ['COMMENT_REPETITION', 'IF_REPETITION', 'CODE_REPETITION']):
            patterns['repetition_issues'].append(result)

    return patterns

def main():
    log_file = '/teamspace/studios/this_studio/qwen-humaneval-eval/logs/fewshot_v2_post_v5_all_cases.log'

    print("=" * 80)
    print("FEWSHOT_V2_POST_V5 FAILURE ANALYSIS")
    print("=" * 80)
    print()

    # Parse log file
    print("Parsing log file...")
    data = parse_log_file(log_file)
    stats = data['stats']
    results = data['results']

    print(f"Total test cases: {stats['total']}")
    print(f"Passed: {stats['passed']} ({stats['passed']/stats['total']*100:.1f}%)")
    print(f"Failed: {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    print()

    # Categorize by problem type
    print("=" * 80)
    print("1. SUCCESS/FAILURE RATES BY PROBLEM CATEGORY")
    print("=" * 80)
    print()

    category_stats = defaultdict(lambda: {'passed': 0, 'failed': 0})
    for result in results:
        categories = categorize_problem(result['docstring'], result['function_name'])
        for cat in categories:
            if result['status'] == 'PASSED':
                category_stats[cat]['passed'] += 1
            else:
                category_stats[cat]['failed'] += 1

    # Sort by total count
    sorted_categories = sorted(category_stats.items(),
                              key=lambda x: x[1]['passed'] + x[1]['failed'],
                              reverse=True)

    for cat, counts in sorted_categories:
        total = counts['passed'] + counts['failed']
        success_rate = counts['passed'] / total * 100 if total > 0 else 0
        print(f"{cat:25s}: {counts['passed']:3d}/{total:3d} passed ({success_rate:5.1f}%) - {counts['failed']:3d} failed")

    print()

    # Failure pattern breakdown
    print("=" * 80)
    print("2. FAILURE PATTERN BREAKDOWN")
    print("=" * 80)
    print()

    failed_results = [r for r in results if r['status'] == 'FAILED']
    all_failure_categories = Counter()
    failure_by_category = defaultdict(list)

    for result in failed_results:
        categories = categorize_failure(result)
        for cat in categories:
            all_failure_categories[cat] += 1
            failure_by_category[cat].append(result)

    print(f"Total failures analyzed: {len(failed_results)}")
    print()
    print("Failure Type                    Count    Percentage")
    print("-" * 80)
    for cat, count in all_failure_categories.most_common():
        pct = count / len(failed_results) * 100
        print(f"{cat:30s}: {count:4d}    ({pct:5.1f}%)")

    print()

    # Top failure patterns with examples
    print("=" * 80)
    print("3. TOP FAILURE PATTERNS WITH EXAMPLES")
    print("=" * 80)
    print()

    top_n = 10
    for i, (cat, count) in enumerate(all_failure_categories.most_common(top_n), 1):
        examples = failure_by_category[cat][:2]  # Show 2 examples

        print(f"\n{i}. {cat} ({count} occurrences, {count/len(failed_results)*100:.1f}%)")
        print("-" * 80)

        for j, ex in enumerate(examples, 1):
            print(f"\n  Example {j}: HumanEval/{ex['humaneval_num']} - {ex['function_name']}()")
            print(f"  Docstring (excerpt):")
            doc_lines = ex['docstring'].split('\n')[:5]
            for line in doc_lines:
                print(f"    {line[:76]}")

            print(f"\n  Generated Code (excerpt):")
            code_lines = ex['processed_code'].split('\n')[:10]
            for line in code_lines:
                print(f"    {line[:76]}")

            if ex['error']:
                print(f"\n  Error (excerpt):")
                error_lines = ex['error'].split('\n')[:3]
                for line in error_lines:
                    print(f"    {line[:76]}")
            print()

    # Specific pattern analysis
    print("\n" + "=" * 80)
    print("4. DETAILED PATTERN ANALYSIS")
    print("=" * 80)
    print()

    patterns = analyze_specific_failures(results)

    for pattern_name, pattern_results in patterns.items():
        if pattern_results:
            print(f"\n{pattern_name.upper().replace('_', ' ')} ({len(pattern_results)} cases)")
            print("-" * 80)
            for r in pattern_results[:3]:
                print(f"  - HumanEval/{r['humaneval_num']}: {r['function_name']}()")

    print()

    # Recommendations
    print("\n" + "=" * 80)
    print("5. PROMPT IMPROVEMENT RECOMMENDATIONS")
    print("=" * 80)
    print()

    # Count major issues
    placeholder_count = sum(1 for r in failed_results if any(cat in categorize_failure(r)
                           for cat in ['PLACEHOLDER_PASS', 'PLACEHOLDER_COMMENT', 'EMPTY_CODE']))
    repetition_count = sum(1 for r in failed_results if any('REPETITION' in cat for cat in categorize_failure(r)))
    logic_count = sum(1 for r in failed_results if 'LOGIC_ERROR' in categorize_failure(r) or 'ASSERTION_FAILED' in categorize_failure(r))
    edge_case_count = sum(1 for r in failed_results if 'MISSING_EDGE_CASE' in categorize_failure(r))

    recommendations = []

    if placeholder_count > stats['failed'] * 0.15:
        recommendations.append({
            'issue': f'High placeholder rate ({placeholder_count} cases, {placeholder_count/stats["failed"]*100:.1f}%)',
            'suggestion': 'Add explicit instruction: "You MUST provide a complete working implementation. Do NOT use placeholder comments like \'# Your code here\' or just \'pass\'. Write actual code that solves the problem."',
            'priority': 'HIGH'
        })

    if repetition_count > 5:
        recommendations.append({
            'issue': f'Code repetition/degeneration ({repetition_count} cases)',
            'suggestion': 'Add stop sequences and temperature adjustment. Include instruction: "Write concise, non-repetitive code. Each line should serve a unique purpose."',
            'priority': 'HIGH'
        })

    if logic_count > stats['failed'] * 0.3:
        recommendations.append({
            'issue': f'Logic errors ({logic_count} cases, {logic_count/stats["failed"]*100:.1f}%)',
            'suggestion': 'Add more diverse few-shot examples covering edge cases. Include instruction: "Carefully read all requirements and handle all edge cases mentioned in the docstring, including empty inputs."',
            'priority': 'HIGH'
        })

    if edge_case_count > 10:
        recommendations.append({
            'issue': f'Missing edge case handling ({edge_case_count} cases)',
            'suggestion': 'Add explicit instruction: "Always handle edge cases first: check for empty/None inputs, single-element cases, and boundary conditions before implementing the main logic."',
            'priority': 'MEDIUM'
        })

    # Check for index errors
    index_error_count = sum(1 for r in failed_results if 'INDEX_ERROR' in categorize_failure(r))
    if index_error_count > 5:
        recommendations.append({
            'issue': f'Index errors ({index_error_count} cases)',
            'suggestion': 'Add instruction: "When iterating with indices, always verify array bounds. Use range(len(arr)-1) when accessing arr[i+1]."',
            'priority': 'MEDIUM'
        })

    # Check for wrong return types
    wrong_return_count = sum(1 for r in failed_results if 'WRONG_RETURN_TYPE' in categorize_failure(r))
    if wrong_return_count > 3:
        recommendations.append({
            'issue': f'Wrong return types ({wrong_return_count} cases)',
            'suggestion': 'Emphasize return type in prompt: "Pay close attention to the expected return type in the function signature and docstring. Return exactly the type specified."',
            'priority': 'MEDIUM'
        })

    # Sort by priority
    priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
    recommendations.sort(key=lambda x: priority_order[x['priority']])

    for i, rec in enumerate(recommendations, 1):
        print(f"\n{i}. [{rec['priority']}] {rec['issue']}")
        print(f"   Recommendation: {rec['suggestion']}")

    # Post-processing improvements
    print("\n" + "=" * 80)
    print("6. POST-PROCESSING IMPROVEMENTS")
    print("=" * 80)
    print()

    print("1. Placeholder Detection Enhancement:")
    print("   - Detect and reject code containing only 'pass' statements")
    print("   - Detect '# Your code here' or 'TODO' patterns and trigger retry")
    print("   - Check for minimum code complexity (e.g., at least 3 non-trivial lines)")

    print("\n2. Repetition Detection:")
    print("   - Add check for repeated lines (>50% duplicate lines = reject)")
    print("   - Detect comment loops and if-statement degeneration")
    print("   - Implement early stopping if repetition detected during generation")

    print("\n3. Syntax Validation:")
    print("   - Run AST parse before test execution to catch syntax errors early")
    print("   - Add retry logic for syntax errors with modified prompt")

    print("\n4. Edge Case Injection:")
    print("   - Automatically inject edge case checks for common patterns")
    print("   - Add empty/None checks if mentioned in docstring")

    print()

    # Summary statistics
    print("\n" + "=" * 80)
    print("7. SUMMARY STATISTICS")
    print("=" * 80)
    print()

    print(f"Success Rate: {stats['passed']}/{stats['total']} = {stats['passed']/stats['total']*100:.1f}%")
    print(f"\nTop 3 Failure Categories:")
    for i, (cat, count) in enumerate(all_failure_categories.most_common(3), 1):
        print(f"  {i}. {cat}: {count} cases ({count/len(failed_results)*100:.1f}%)")

    print(f"\nMost Problematic Problem Categories:")
    worst_categories = sorted(sorted_categories, key=lambda x: x[1]['passed']/(x[1]['passed']+x[1]['failed']) if x[1]['passed']+x[1]['failed'] > 5 else 1)[:3]
    for i, (cat, counts) in enumerate(worst_categories, 1):
        total = counts['passed'] + counts['failed']
        if total > 5:
            print(f"  {i}. {cat}: {counts['passed']}/{total} = {counts['passed']/total*100:.1f}% success")

    print()

if __name__ == '__main__':
    main()

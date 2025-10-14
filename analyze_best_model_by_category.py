#!/usr/bin/env python3
"""
Analyze fewshot_v2_devise_post_v7 results by problem category.
Categorizes all 164 HumanEval problems into max 16 categories.
"""

import re
from collections import defaultdict
from datasets import load_dataset

def parse_all_cases_log(log_path):
    """Parse all_cases log file to extract pass/fail results."""
    results = {}
    with open(log_path, 'r') as f:
        content = f.read()

    # Split by test case headers
    test_cases = re.split(r'\n\[(\d+)/164\] (HumanEval/\d+) - ([✓✗]) (PASSED|FAILED)', content)

    # Process matches (format: text, index, test_id, symbol, status, ...)
    for i in range(1, len(test_cases), 5):
        if i+3 < len(test_cases):
            test_id = test_cases[i+1]
            status = test_cases[i+3]
            results[test_id] = (status == "PASSED")

    return results

def categorize_problem(task_id, prompt, entry_point):
    """
    Categorize a problem into ONE primary category (max 16 categories).
    Uses hierarchical rules to assign single best-fit category.
    """
    docstring = ""
    docstring_match = re.search(r'"""(.*?)"""', prompt, re.DOTALL)
    if docstring_match:
        docstring = docstring_match.group(1).strip().lower()
    else:
        docstring = prompt.lower()

    # Priority-based categorization (most specific first)

    # 1. Parsing & Validation
    if any(word in docstring for word in ['parenthes', 'bracket', 'balanced', 'nested']):
        return 'Parsing & Validation'

    # 2. Prime & Factorization
    if any(word in docstring for word in ['prime', 'factor', 'divisor', 'composite']):
        return 'Prime & Factorization'

    # 3. Sorting & Ordering
    if any(word in docstring for word in ['sort', 'order', 'arrange', 'largest', 'smallest', 'ascending', 'descending']):
        return 'Sorting & Ordering'

    # 4. String Manipulation
    if any(word in docstring for word in ['string', 'character', 'word', 'substring', 'vowel', 'consonant', 'reverse', 'palindrome', 'case']):
        return 'String Manipulation'

    # 5. List/Array Operations
    if any(word in docstring for word in ['list', 'array', 'element', 'remove duplicates', 'flatten', 'rotate', 'shift']):
        return 'List/Array Operations'

    # 6. Mathematical Computation
    if any(word in docstring for word in ['sum', 'product', 'multiply', 'average', 'median', 'mean', 'gcd', 'lcm', 'factorial', 'fibonacci']):
        return 'Mathematical Computation'

    # 7. Search & Find
    if any(word in docstring for word in ['search', 'find', 'index', 'position', 'locate', 'closest', 'nearest']):
        return 'Search & Find'

    # 8. Filtering & Selection
    if any(word in docstring for word in ['filter', 'select', 'choose', 'pick', 'extract', 'satisfy']):
        return 'Filtering & Selection'

    # 9. Comparison & Matching
    if any(word in docstring for word in ['compare', 'match', 'equal', 'same', 'different', 'diff']):
        return 'Comparison & Matching'

    # 10. Sequence Generation
    if any(word in docstring for word in ['generate', 'create', 'build', 'construct', 'make', 'sequence']):
        return 'Sequence Generation'

    # 11. Counting & Aggregation
    if any(word in docstring for word in ['count', 'frequency', 'occurrence', 'how many', 'number of']):
        return 'Counting & Aggregation'

    # 12. Boolean Logic
    if any(word in docstring for word in ['check', 'verify', 'valid', 'true', 'false', 'boolean', 'is_', 'has_']):
        return 'Boolean Logic'

    # 13. Data Structure Operations
    if any(word in docstring for word in ['tree', 'graph', 'node', 'dict', 'dictionary', 'map', 'set']):
        return 'Data Structure Operations'

    # 14. Numeric Operations
    if any(word in docstring for word in ['number', 'integer', 'digit', 'decimal', 'binary', 'hex']):
        return 'Numeric Operations'

    # 15. Text Processing
    if any(word in docstring for word in ['parse', 'format', 'encode', 'decode', 'translate', 'convert']):
        return 'Text Processing'

    # 16. Other/Mixed
    return 'Other/Mixed'

def main():
    print("Loading HumanEval dataset...")
    dataset = load_dataset("openai_humaneval", split="test")

    print("Parsing results from all_cases log...")
    log_path = 'logs/fewshot_v2_devise_post_v7_all_cases.log'
    results = parse_all_cases_log(log_path)

    print(f"Found {len(results)} results\n")

    # Categorize all problems
    categories = defaultdict(lambda: {'passed': [], 'failed': []})

    for problem in dataset:
        task_id = problem['task_id']
        prompt = problem['prompt']
        entry_point = problem['entry_point']

        category = categorize_problem(task_id, prompt, entry_point)

        if task_id in results:
            if results[task_id]:
                categories[category]['passed'].append(task_id)
            else:
                categories[category]['failed'].append(task_id)

    # Calculate statistics
    category_stats = []
    for category, data in categories.items():
        passed = len(data['passed'])
        failed = len(data['failed'])
        total = passed + failed
        success_rate = (passed / total * 100) if total > 0 else 0
        failure_rate = (failed / total * 100) if total > 0 else 0

        category_stats.append({
            'category': category,
            'passed': passed,
            'failed': failed,
            'total': total,
            'success_rate': success_rate,
            'failure_rate': failure_rate
        })

    # Sort by failure rate (descending)
    category_stats.sort(key=lambda x: x['failure_rate'], reverse=True)

    # Print results
    print("="*100)
    print("BEST MODEL: fewshot_v2_devise_post_v7")
    print("="*100)
    print(f"Overall: 57/164 passed (34.8%)")
    print()

    print("="*100)
    print("RESULTS BY CATEGORY (sorted by failure rate)")
    print("="*100)
    print(f"{'Category':<30} {'Passed':<8} {'Failed':<8} {'Total':<8} {'Success %':<12} {'Failure %':<12}")
    print("-"*100)

    for stat in category_stats:
        print(f"{stat['category']:<30} "
              f"{stat['passed']:<8} "
              f"{stat['failed']:<8} "
              f"{stat['total']:<8} "
              f"{stat['success_rate']:<12.1f} "
              f"{stat['failure_rate']:<12.1f}")

    print("-"*100)
    total_passed = sum(s['passed'] for s in category_stats)
    total_failed = sum(s['failed'] for s in category_stats)
    total_all = total_passed + total_failed
    print(f"{'TOTAL':<30} {total_passed:<8} {total_failed:<8} {total_all:<8} "
          f"{total_passed/total_all*100:<12.1f} {total_failed/total_all*100:<12.1f}")

    # Save detailed results
    print("\n\nSaving detailed category breakdown...")
    with open('category_analysis_best_model.txt', 'w') as f:
        f.write("="*100 + "\n")
        f.write("CATEGORY ANALYSIS - fewshot_v2_devise_post_v7 (Best Model: 57/164 = 34.8%)\n")
        f.write("="*100 + "\n\n")

        for stat in category_stats:
            category = stat['category']
            f.write(f"\n{'='*100}\n")
            f.write(f"Category: {category}\n")
            f.write(f"Success: {stat['passed']}/{stat['total']} ({stat['success_rate']:.1f}%)\n")
            f.write(f"Failure: {stat['failed']}/{stat['total']} ({stat['failure_rate']:.1f}%)\n")
            f.write(f"{'='*100}\n")

            if categories[category]['passed']:
                f.write(f"\nPASSED ({len(categories[category]['passed'])}):\n")
                for task_id in sorted(categories[category]['passed']):
                    f.write(f"  ✓ {task_id}\n")

            if categories[category]['failed']:
                f.write(f"\nFAILED ({len(categories[category]['failed'])}):\n")
                for task_id in sorted(categories[category]['failed']):
                    f.write(f"  ✗ {task_id}\n")
            f.write("\n")

    print(f"✓ Detailed results saved to: category_analysis_best_model.txt")

    # Return stats for README
    return category_stats

if __name__ == "__main__":
    stats = main()

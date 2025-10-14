#!/usr/bin/env python3
"""
Analyze overlap in failures and successes between:
- fewshot_v2_devise_none (6 examples + devise instruction)
- fewshot_v2_devise_def_v1_none (definitions only + devise instruction)
"""

import re
from collections import defaultdict

def parse_all_cases_log(log_path):
    """Parse all_cases log file to extract pass/fail results."""
    results = {}

    with open(log_path, 'r') as f:
        content = f.read()

    # Split by test case headers
    test_cases = re.split(r'\n\[\d+/164\] (HumanEval/\d+) - ([✓✗]) (PASSED|FAILED)', content)

    # Process matches (format: text, test_id, symbol, status, ...)
    for i in range(1, len(test_cases), 4):
        if i+2 < len(test_cases):
            test_id = test_cases[i]
            status = test_cases[i+2]
            results[test_id] = (status == "PASSED")

    return results

def main():
    # Parse both logs
    print("Parsing log files...")
    devise_results = parse_all_cases_log('logs/fewshot_v2_devise_none_all_cases.log')
    def_v1_results = parse_all_cases_log('logs/fewshot_v2_devise_def_v1_none_all_cases.log')

    # Get all test IDs
    all_tests = sorted(set(devise_results.keys()) | set(def_v1_results.keys()))

    # Categorize results
    both_pass = []
    both_fail = []
    only_devise_pass = []
    only_def_v1_pass = []

    for test_id in all_tests:
        devise_pass = devise_results.get(test_id, False)
        def_v1_pass = def_v1_results.get(test_id, False)

        if devise_pass and def_v1_pass:
            both_pass.append(test_id)
        elif not devise_pass and not def_v1_pass:
            both_fail.append(test_id)
        elif devise_pass and not def_v1_pass:
            only_devise_pass.append(test_id)
        elif not devise_pass and def_v1_pass:
            only_def_v1_pass.append(test_id)

    # Calculate totals
    devise_total = sum(devise_results.values())
    def_v1_total = sum(def_v1_results.values())

    # Print analysis
    print("\n" + "="*80)
    print("OVERLAP ANALYSIS: fewshot_v2_devise vs fewshot_v2_devise_def_v1")
    print("="*80)

    print(f"\n📊 OVERALL RESULTS:")
    print(f"   fewshot_v2_devise_none (6 examples):      {devise_total}/164 ({devise_total/164*100:.1f}%)")
    print(f"   fewshot_v2_devise_def_v1_none (definitions): {def_v1_total}/164 ({def_v1_total/164*100:.1f}%)")
    print(f"   Difference: {def_v1_total - devise_total:+d} ({(def_v1_total - devise_total)/164*100:+.1f}%)")

    print(f"\n✅ BOTH PASS: {len(both_pass)}/164 ({len(both_pass)/164*100:.1f}%)")
    print(f"   These problems are solved by both approaches")

    print(f"\n❌ BOTH FAIL: {len(both_fail)}/164 ({len(both_fail)/164*100:.1f}%)")
    print(f"   These problems are hard for both approaches")

    print(f"\n🔵 ONLY fewshot_v2_devise PASSES: {len(only_devise_pass)}/164 ({len(only_devise_pass)/164*100:.1f}%)")
    print(f"   Examples help but definitions don't:")
    if only_devise_pass:
        for test_id in only_devise_pass[:10]:
            print(f"      - {test_id}")
        if len(only_devise_pass) > 10:
            print(f"      ... and {len(only_devise_pass) - 10} more")

    print(f"\n🟢 ONLY fewshot_v2_devise_def_v1 PASSES: {len(only_def_v1_pass)}/164 ({len(only_def_v1_pass)/164*100:.1f}%)")
    print(f"   Definitions help but examples don't:")
    if only_def_v1_pass:
        for test_id in only_def_v1_pass[:10]:
            print(f"      - {test_id}")
        if len(only_def_v1_pass) > 10:
            print(f"      ... and {len(only_def_v1_pass) - 10} more")

    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)

    if def_v1_total > devise_total:
        print(f"✅ Definitions approach is BETTER (+{def_v1_total - devise_total} cases)")
        print(f"   - Definitions provide clearer conceptual grounding")
        print(f"   - {len(only_def_v1_pass)} cases benefit from definitions over examples")
    elif devise_total > def_v1_total:
        print(f"✅ Examples approach is BETTER (+{devise_total - def_v1_total} cases)")
        print(f"   - Examples provide concrete patterns to follow")
        print(f"   - {len(only_devise_pass)} cases benefit from examples over definitions")
    else:
        print(f"🟡 EQUAL performance ({devise_total} cases each)")
        print(f"   - Both approaches have similar effectiveness")

    print(f"\n💡 Complementary value: {len(only_devise_pass) + len(only_def_v1_pass)} cases")
    print(f"   Combining both might reach {len(both_pass) + len(only_devise_pass) + len(only_def_v1_pass)}/164")
    print(f"   ({(len(both_pass) + len(only_devise_pass) + len(only_def_v1_pass))/164*100:.1f}%) if synergistic")

    # Save detailed results
    with open('devise_overlap_analysis.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("DETAILED OVERLAP ANALYSIS\n")
        f.write("="*80 + "\n\n")

        f.write(f"BOTH PASS ({len(both_pass)} cases):\n")
        for test_id in both_pass:
            f.write(f"  {test_id}\n")

        f.write(f"\nBOTH FAIL ({len(both_fail)} cases):\n")
        for test_id in both_fail:
            f.write(f"  {test_id}\n")

        f.write(f"\nONLY fewshot_v2_devise PASSES ({len(only_devise_pass)} cases):\n")
        for test_id in only_devise_pass:
            f.write(f"  {test_id}\n")

        f.write(f"\nONLY fewshot_v2_devise_def_v1 PASSES ({len(only_def_v1_pass)} cases):\n")
        for test_id in only_def_v1_pass:
            f.write(f"  {test_id}\n")

    print(f"\n📄 Detailed results saved to: devise_overlap_analysis.txt")

if __name__ == "__main__":
    main()

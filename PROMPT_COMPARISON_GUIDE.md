# Prompt Strategy Comparison Guide

## Overview

This guide explains the prompt comparison setup to test which prompt strategy generates the best code with the improved `post_process_v5` post-processor.

## Prompt Strategies Being Tested

### minimal_v2
**Strategy:** Adds a comment hint to guide the model
**Format:** `problem + "\n    # Your code here\n"`

**Example:**
```python
def add(a: int, b: int) -> int:
    """Add two numbers."""
    # Your code here

```

**Rationale:** The comment hints at where code should go, potentially reducing stub code.

---

### minimal_v3
**Strategy:** Ultra-minimal with just a newline
**Format:** `problem.rstrip() + "\n"`

**Example:**
```python
def add(a: int, b: int) -> int:
    """Add two numbers."""

```

**Rationale:** Minimal priming, lets the model decide what to do next naturally.

---

### minimal_v4
**Strategy:** Adds indentation hint to prime function body
**Format:** `problem.rstrip() + "\n    "`

**Example:**
```python
def add(a: int, b: int) -> int:
    """Add two numbers."""
    ▌
```
(Note: ▌ represents the cursor position with 4-space indentation)

**Rationale:** The indentation signals "function body goes here", encouraging implementation.

---

### minimal_v5
**Strategy:** Bare minimal - just strip trailing whitespace
**Format:** `problem.rstrip()`

**Example:**
```python
def add(a: int, b: int) -> int:
    """Add two numbers."""▌
```
(Note: ▌ represents cursor at end of docstring)

**Rationale:** No priming at all - test if model naturally continues correctly.

---

## Post-Processing: post_v5

All strategies use the **improved `post_process_v5`** which includes:

✅ **Dependency Injection:** 6 helper functions (is_prime, is_palindrome, reverse, product, is_balanced, prod_sign)
✅ **Truncation Fixes:** 5 patterns (missing colons, unclosed brackets, no body, unterminated strings, repetition)
✅ **ValueError Fixes:** Word-to-number conversion, safe digit iteration
✅ **Type Conversions:** 4 patterns (is_palindrome(int), float() comma handling, .is_integer() on strings)
✅ **Targeted Fixes:** 3 specific patterns (remove_vowels, closest_integer, compare)

---

## Running the Comparison

### Option 1: Using the Python Script (Recommended)
```bash
python run_prompt_comparison.py
```

### Option 2: Using the Shell Script
```bash
./run_prompt_comparison.sh
```

### Option 3: Manual Testing
Test a single strategy:
```bash
# Generate completions
python scripts/inference.py \
    --prompt-strategy minimal_v3 \
    --postprocess-strategy post_v5 \
    --output results/completions_minimal_v3_post_v5.jsonl

# Evaluate
python scripts/run_evaluation.py \
    --completions results/completions_minimal_v3_post_v5.jsonl \
    --output results/eval_minimal_v3_post_v5.json \
    --log logs/minimal_v3_post_v5_all_cases.log
```

---

## Output Files

### Generated Files (per strategy):
- `results/completions_{strategy}_post_v5.jsonl` - Raw completions from model
- `results/eval_{strategy}_post_v5.json` - Evaluation metrics (pass rate, etc.)
- `logs/{strategy}_post_v5_all_cases.log` - Detailed failure analysis

### Results Format:
```json
{
    "passed": 54,
    "total": 164,
    "pass_at_1": 0.3293,
    "details": [...]
}
```

---

## What We're Measuring

1. **Pass Rate:** How many tests pass (out of 164 HumanEval problems)
2. **Error Types:** What kinds of failures occur (syntax, runtime, logic)
3. **Code Quality:** How complete and correct the generated code is

---

## Expected Outcomes

### Hypothesis:
Different prompts may perform differently because:
- **minimal_v2**: Comment might reduce stubs but could confuse model
- **minimal_v3**: Newline might be most natural for code completion
- **minimal_v4**: Indentation might prime better function bodies
- **minimal_v5**: No priming might avoid biasing the model

### Current Baseline:
- **minimal_v5 + post_v5**: ~52-54 passes (31.7-33%)

### Goal:
Find which prompt strategy combined with `post_v5` yields the highest pass rate.

---

## Analyzing Results

After running, compare:

1. **Overall pass rate** - Which strategy passes most tests?
2. **Error patterns** - Which reduces syntax errors? Runtime errors? Logic errors?
3. **Code completeness** - Which reduces stub code?

Check detailed logs:
```bash
# View specific failures
cat logs/minimal_v3_post_v5_all_cases.log | grep "ERROR:" -A 5

# Count error types
grep "ERROR:" logs/minimal_v3_post_v5_all_cases.log | wc -l
```

---

## Next Steps After Comparison

1. **Identify winner** - Use the strategy with highest pass rate
2. **Analyze failures** - Understand remaining issues in best strategy
3. **Iterate** - Potentially create hybrid approaches or new strategies
4. **Consider other factors:**
   - Model temperature
   - Max tokens
   - Stop sequences
   - Multi-attempt strategies (Pass@k)

---

## Notes

- All tests use temperature=0.0 for reproducibility
- Post-processing fixes are identical across all prompts
- Differences in results are purely due to prompt strategy
- The comparison helps separate "prompt quality" from "post-processing quality"

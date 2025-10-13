# Failure Pattern Analysis: What the Model Can't Solve

## Overview

Analysis of 111 failed HumanEval cases (out of 164 total) with `minimal_v5 + post_v5`.

**Current Pass Rate:** 52/164 (31.7%)

---

## Problem Type Distribution

| Category | Failures | Percentage |
|----------|----------|------------|
| **String Manipulation** | 40 | 36.0% |
| **List/Array Operations** | 30 | 27.0% |
| **Sorting/Ordering** | 23 | 20.7% |
| **Number/Math Operations** | 15 | 13.5% |
| **Other/Complex** | 2 | 1.8% |
| **Aggregation** | 1 | 0.9% |

---

## Top 6 Algorithmic Weaknesses

### 1. **String Manipulation** (36% of failures)

**Problems the model can't solve:**
- Complex string parsing and splitting
- Parentheses/bracket matching
- Character-level manipulations
- Word boundary detection

**Examples:**
- `HumanEval/1 - separate_paren_groups()` - Can't parse nested parentheses
- `HumanEval/101 - words_string()` - Wrong split on commas vs spaces
- `HumanEval/10 - make_palindrome()` - Can't find longest palindromic suffix

**Common mistakes:**
- Uses wrong split delimiter
- Doesn't handle nested structures
- Forgets to handle empty strings
- Character vs. string confusion

---

### 2. **Nested Iteration** (8-9% explicitly, but affects 20-30% overall)

**Problem:** Model defaults to single loops, rarely generates nested loops even when needed.

**Critical failure example:**
```python
# HumanEval/0 - has_close_elements()
# WRONG (what model generates):
for i in range(len(numbers) - 1):
    if abs(numbers[i] - numbers[i + 1]) < threshold:
        return True

# RIGHT (needs nested loop):
for i in range(len(numbers)):
    for j in range(i + 1, len(numbers)):
        if abs(numbers[i] - numbers[j]) < threshold:
            return True
```

**Why it matters:**
- Model checks only adjacent elements, not all pairs
- Affects "compare all" type problems
- ~9 failures directly from this pattern

---

### 3. **Sorting with Complex Keys** (21% of failures)

**Problems:**
- Wrong sort key functions
- Doesn't preserve original order for ties (stable sort)
- Forgets secondary sort criteria

**Examples:**
- `HumanEval/104 - unique_digits()` - Filter then sort, gets logic backward
- `HumanEval/145 - order_by_points()` - Sum of digits, but can't handle negative signs
- `HumanEval/109 - move_one_ball()` - Rotation check, wrong logic

**Common pattern:**
```python
# Model often does:
return sorted(list)  # Simple sort, ignores custom criteria

# Needs:
return sorted(list, key=lambda x: (criteria1, criteria2))
```

---

### 4. **Complex Multi-Part Conditions** (12% of failures)

**Problem:** Model simplifies or omits parts of multi-condition requirements.

**Pattern:**
- Docstring says: "If A and B and C, then X, else if D, then Y"
- Model generates: "If A, then X" (incomplete)

**Examples:**
- `HumanEval/102 - choose_num()` - "Biggest even in range" → model forgets "even" part
- Validation functions with 3-4 conditions → model only checks 1-2

**Result:**
- Incomplete if/else chains
- Missing edge case handling
- Wrong logic flow

---

### 5. **Edge Cases** (Pervasive across all categories)

**Consistently missed:**
- Empty inputs (`[]`, `""`, `0`)
- Single element lists/strings
- Negative numbers
- Special characters
- Boundary conditions

**Why:** Model generates "happy path" code, doesn't think about edge cases.

---

### 6. **State Tracking / Accumulation** (5-6% of failures)

**Problems:**
- Doesn't maintain counters properly
- Forgets to accumulate/aggregate across iterations
- Loses state between loop iterations

**Examples:**
- Counting problems → forgets counter variable
- Building result lists → returns intermediate result
- Multiple passes → only does one pass

---

## Specific Failure Modes

### A. **Stub/Placeholder Code** (6 cases)
Model outputs:
```python
# Your code here
pass
```

**Why:** Prompt doesn't prime properly, model gives up.

---

### B. **Repetitive/Infinite Logic** (2 cases)
Model generates 100+ lines of repeated `if` statements.

**Example:** HumanEval/102, HumanEval/127

**Why:** Model gets stuck in generation loop, doesn't realize it's repeating.

---

### C. **Wrong Algorithm Entirely** (~15 cases)
Model uses fundamentally wrong approach:
- Needs recursion, uses iteration
- Needs multiple passes, uses single pass
- Needs dictionary, uses list

---

### D. **Truncated/Incomplete** (Fixed by post_v5, but still ~9 syntax errors remain)
Model output gets cut off mid-generation.

---

## Code Characteristics of Failures

| Pattern | Count | Percentage |
|---------|-------|------------|
| Has return statement | 96 | 86.5% |
| Has loops | 47 | 42.3% |
| Has helper function calls | 44 | 39.6% |
| Has comments | 42 | 37.8% |
| Has list comprehension | 33 | 29.7% |
| Has if/else logic | 30 | 27.0% |

**Insight:** Failed code is not obviously "bad" - it has proper structure, loops, logic. The issue is **wrong algorithm**, not syntax.

---

## What Post-Processing Can't Fix

✅ **Can fix:**
- Syntax errors (missing colons, brackets)
- Type errors (int→str conversions)
- Missing imports/dependencies
- ValueError issues

❌ **Can't fix:**
- Wrong algorithm choice
- Missing nested loops
- Incorrect logic flow
- Incomplete conditionals
- Edge case handling
- State management errors

**Bottom line:** 85% of failures are **algorithmic/logic errors** that post-processing cannot address.

---

## Recommendations

### **Short-term (Prompt Engineering):**

1. **Add algorithmic hints to prompts:**
   ```
   "If comparing all pairs, use nested loops: for i... for j..."
   "For complex conditions, handle all cases with if/elif/else"
   "Always test edge cases: empty input, single element, negatives"
   ```

2. **Few-shot examples for hard patterns:**
   - Show nested loop example
   - Show complex condition example
   - Show string parsing example

3. **Explicit instructions:**
   ```
   "Read the docstring carefully. Implement ALL requirements."
   "Handle edge cases: empty inputs, negative numbers, special characters."
   "If 'all pairs' or 'between', you likely need nested loops."
   ```

---

### **Medium-term (Model Selection):**

1. **Test larger models:**
   - Qwen2.5-Coder-1.5B or 7B
   - Better reasoning → fewer algorithmic errors

2. **Try different model families:**
   - CodeLlama
   - DeepSeek-Coder
   - StarCoder

3. **Use instruct-tuned versions:**
   - Better at following complex instructions
   - Less likely to generate stubs

---

### **Long-term (Advanced Strategies):**

1. **Test-driven generation:**
   - Show model the test cases
   - Let it self-correct based on failures

2. **Multi-attempt (Pass@k):**
   - Generate 5-10 solutions
   - Pick best one (most tests pass)

3. **Chain-of-thought:**
   - Make model explain algorithm first
   - Then generate code based on explanation

4. **Hybrid approaches:**
   - Detect problem type (string/array/math)
   - Use specialized prompts per type
   - Apply different post-processing per type

---

## Specific High-Value Targets

**If you could fix these 3 patterns, you'd gain ~25 passes:**

1. **String manipulation** (40 cases)
   - Better prompts emphasizing string parsing
   - Few-shot examples of split/join operations

2. **Nested loops** (9+ cases)
   - Explicit hint in prompt: "compare all pairs → nested loop"
   - Example showing proper nested iteration

3. **Complex conditions** (12 cases)
   - Prompt: "Implement ALL requirements from docstring"
   - Checklist-style validation in prompt

---

## Conclusion

The model's weaknesses are primarily **algorithmic understanding**, not syntax or basic coding:

**Can do:**
- Generate syntactically correct code
- Use loops, conditionals, functions
- Return reasonable values

**Can't do:**
- Recognize when nested loops needed
- Parse complex strings correctly
- Handle all edge cases
- Implement multi-part conditional logic
- Choose right algorithm for problem

**Next step:** Focus on **prompt engineering** to guide algorithmic choices, not post-processing to fix syntax.

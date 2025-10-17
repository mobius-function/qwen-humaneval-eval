# HumanEval Code Generation with Qwen2.5-Coder-0.5B

**NEW: 54.3% pass@1 (89/164)** using Instruct model + Chat API + AST post-processing

**Previous Best: 34.8% pass@1 (57/164)** using Base model + Completion API + few-shot prompting

---

## Table of Contents

- [Quick Start](#quick-start)
- [Results](#results)
- [Key Findings](#key-findings)
- [Insights & Lessons](#insights--lessons)
- [Project Architecture](#project-architecture)
- [Advanced Usage](#advanced-usage)
- [Performance Optimization](#performance-optimization)

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for vLLM)
- NVIDIA GPU (recommended) or CPU
- 16GB+ RAM

### Installation & Run

```bash
# Clone and install
git clone <repository-url>
cd qwen-humaneval-eval
pip install -e .  # or: uv sync

# Start vLLM server
./scripts/manage_services.sh start

# Run experiments
uv run python scripts/run_experiments.py
```

Configure experiments in `config.yml`.

---

## Results

### 🏆 NEW BEST: Instruct Model with Chat API

**Performance**: **89/164 (54.3% pass@1)**

**Model**: Qwen2.5-Coder-0.5B-**Instruct** (chat/dialogue model)

**Strategy**:
- Chat API with system + user messages
- Direct instruction to complete function body only
- Strategy-specific AST-based post-processing (extracts function body from complete functions)
- **System Prompt**: "You are a Python expert. Complete the function body ONLY - do not repeat imports, function signature, or docstring."
- **User Prompt**: Explicit task description with constraints on output format

**Why It Works**:
- ✅ Instruct models are trained for dialogue and helpful responses
- ✅ Better at understanding natural language instructions
- ✅ Model naturally generates complete, correct implementations (not just function bodies)
- ✅ AST post-processing extracts the function body, combining correctness with proper format

**The Winning Prompt**:

<details>
<summary><b>View prompt template (fewshot_v1)</b></summary>

**System Message**:
```
You are a Python expert. Complete the function body ONLY - do not repeat imports, function signature, or docstring.
```

**User Message**:
```
Complete the function body for:

{problem}

Output ONLY the implementation code that goes INSIDE the function (no imports, no signature, no docstring, no markdown, no explanations).
```

**Key Components**:
- **System prompt**: Clear instruction to complete function body only
- **User prompt**: Explicit constraints on what to output (function body only, no extras)
- **Post-processing**: AST-based extraction of function body from complete code (post_chat)
- **Stop sequences**: Empty list (avoids interference with markdown code blocks)
- **Why it works**: Despite explicit instructions to output only the body, the instruct model still generates complete functions. The AST post-processor extracts just the body, getting the best of both worlds: complete, correct implementations that are then properly formatted.

</details>

---

### Previous Best: Base Model with Completion API

**Performance**: **57/164 (34.8% pass@1)**

**Model**: Qwen2.5-Coder-0.5B (base completion model)

**Strategy**:
- Completion API (Fill-In-Middle)
- 6 carefully selected few-shot examples
- "Devise algorithm" instruction (mental planning before coding)
- Post-processing v7 (fixes delimiters, averages, rotations)

---

### Model Comparison

| Model Type | API Mode | Best Strategy | pass@1 | Improvement |
|------------|----------|---------------|--------|-------------|
| **Instruct** | Chat | fewshot_v1 + post_chat | **54.3%** | **+19.5%** |
| Base | Completion | fewshot_v2_devise + post_v7 | 34.8% | Baseline |
| Base | Completion | fewshot_v1 + none | 32.9% | - |

**Key Insight**: Instruct models outperform Base models significantly on HumanEval when using appropriate prompting and post-processing.

---

### All Experiments Comparison

**Chat API (Instruct Model):**
| Experiment | Prompt Strategy | Post-processing | pass@1 | Passed/Total |
|------------|----------------|-----------------|--------|--------------|
| **fewshot_v1_chat_post_chat** | Direct instruction | AST extraction | **54.3%** | **89/164** |

**Completion API (Base Model):**
| Experiment | Prompt Strategy | Post-processing | pass@1 | Passed/Total |
|------------|----------------|-----------------|--------|--------------|
| fewshot_v2_devise_post_v7 | 6 examples + devise | post_v7 | 34.8% | 57/164 |
| fewshot_v2_devise_none | 6 examples + devise | none | 34.1% | 56/164 |
| fewshot_v2_none | 6 examples + CoT | none | 33.5% | 55/164 |
| fewshot_v1_none | 6 examples only | none | 32.9% | 54/164 |

---

### Chat Model Architecture Insights

**Challenge**: Instruct models generate **complete functions** (imports, signature, docstring, implementation), but HumanEval expects **only the function body**.

**Solution**: Strategy-specific AST-based post-processing
1. Parse generated code with Python AST
2. Locate the function definition
3. Extract only the body (skip imports, signature, docstring)
4. Return properly indented implementation

**Before Post-Processing** (0% pass rate):
```python
# Prompt already has:
def has_close_elements(...):
    """docstring"""

# Model generates:
def has_close_elements(...):  ← Duplicate!
    """docstring"""          ← Duplicate!
    implementation

# Result: Nested function definition = syntax error
```

**After Post-Processing** (54.3% pass rate):
```python
# Prompt has:
def has_close_elements(...):
    """docstring"""

# Post-processor extracts only:
    implementation           ← Just the body!

# Result: Clean, correct code
```

**Key Learnings**:
1. **Instruct models don't follow "function body only" instructions** - they're trained to provide complete solutions
2. **AST parsing >> regex** for extracting function bodies from complete code
3. **Different prompts need different post-processing** - thus the strategy-specific architecture
4. **Stop tokens matter** - Qwen2.5-Coder-Instruct has ` ``` ` as default stop token, causing premature termination with markdown

---

## Key Findings

### Instruct vs Base Models

**Instruct Model (54.3%)**:
- ✅ Direct instruction works better than few-shot examples
- ✅ AST post-processing is essential (model generates complete functions)
- ✅ Simple, clear prompts outperform complex CoT strategies
- ✅ Stop sequences must be empty (avoid interference with markdown)

**Base Model (34.8%)**:
- ✅ Few-shot examples are critical (drove 55-56/164 success)
- ✅ "Devise algorithm" instruction helps (+1 case)
- ✅ Post-processing logic fixes help (+1 case)
- ✅ Strategic example selection matters

**Key Insight**: **Instruct models break the 0.5B ceiling** - jumping from 34.8% to 54.3% (+19.5%) by leveraging their instruction-following capabilities.

---

### What Didn't Work

These approaches were tested but failed to improve performance:

- **Minimal prompt variations** - Adding `.rstrip()`, `\n`, expert framing, anti-stub instructions showed no improvement
- **Ensemble expert prompts** - Categorizing problems and assigning Qwen expert roles for each category couldn't beat simple prompts
- **JSON-structured prompts** - Sending examples/role/instructions via JSON format performed extremely poorly (<10/164); models expect natural text
- **Data-driven iterative refinement** - Analyzing failure cases and iteratively updating prompts didn't yield improvements; fundamental model limitations remain

---

### Base Model Prompt (34.8% pass@1)

<details>
<summary><b>View full prompt template</b></summary>

```python
Here are examples of correct Python function implementations:

from typing import List, Optional

Example 1:
def find_max(numbers: List[int]) -> Optional[int]:
    """Find the maximum number in a list."""
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val

Example 2:
def count_vowels(text: str) -> int:
    """Count vowels in a string (case-insensitive)."""
    vowels = 'aeiouAEIOU'
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count

Example 3:
def remove_duplicates(items: List[int]) -> List[int]:
    """Remove duplicates from list while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

Example 4:
def reverse_words(sentence: str) -> str:
    """Reverse the order of words in a sentence."""
    if not sentence:
        return ""
    words = sentence.split()
    reversed_words = words[::-1]
    return " ".join(reversed_words)

Example 5:
def filter_positive_even(numbers: List[int]) -> List[int]:
    """Return only positive even numbers from the list."""
    result = []
    for num in numbers:
        if num > 0 and num % 2 == 0:
            result.append(num)
    return result

Example 6:
def is_perfect_square(n: int) -> bool:
    """Check if a number is a perfect square."""
    if n < 0:
        return False
    if n == 0:
        return True
    i = 1
    while i * i <= n:
        if i * i == n:
            return True
        i += 1
    return False

Now implement the function. Think through the problem step by step.
First, devise the logic to solve the problem and review it.
Once satisfied with logic, go ahead and implement in Python.

{problem}
```

**Components**:
- 6 diverse examples covering common patterns (lists, strings, filtering, math, edge cases)
- "Devise the logic" instruction for mental planning
- Step-by-step guidance without overwhelming the model

**Post-processing v7**: Pattern-based fixes for delimiters, averages, rotations, token parsing

</details>

---

## Insights & Lessons

### Key Learnings

**1. Model Selection Matters Most**
- Instruct models (+19.5%) >> Few-shot prompting optimization (~2%)
- The biggest gain came from switching model types, not prompt engineering
- 0.5B Instruct models can achieve 54.3%, breaking the "Base model ceiling"

**2. AST Post-Processing is Essential for Instruct Models**
- Instruct models naturally generate complete functions (good for correctness)
- HumanEval expects only function bodies (format mismatch)
- AST-based extraction solves this: parse → extract body → return clean code

**3. Simplicity Wins for Instruct Models**
- Direct instructions > Few-shot examples
- Simple prompts > Complex CoT strategies
- Clear constraints in user prompt guide output format

**4. Base Models Need Examples**
- Few-shot examples drove 55-56/164 success for Base models
- Strategic example selection helps (+5-10 percentage points)
- "Devise algorithm" instruction adds +1 case improvement

**5. Stop Sequences Matter**
- Qwen2.5-Coder-Instruct has ` ``` ` as default stop token
- Empty stop sequences avoid premature termination with markdown
- Different models need different stop sequence configurations

---

## Project Architecture

### Two Approaches

#### Approach 1: Prompt Engineering [Recommended]

Text-based prompts with vLLM serving.

```bash
./scripts/manage_services.sh start
uv run python scripts/run_experiments.py
```

Edit `config.yml` for configuration.

#### Approach 2: Soft Prompt Tuning [Experimental]

Learnable prompt embeddings via gradient descent.

```bash
uv run python scripts/train_soft_prompts.py
uv run python scripts/inference_with_soft_prompts.py
uv run python scripts/run_evaluation.py
```

Status: Debugging NaN loss issues

---

### Project Structure

```
qwen-humaneval-eval/
├── config.yml                    # Experiment configuration
├── docker-compose.yml            # vLLM server setup
├── scripts/
│   ├── inference.py              # Model inference
│   ├── run_evaluation.py         # HumanEval evaluation
│   ├── run_experiments.py        # Experiment runner
│   └── train_soft_prompts.py    # Soft prompt training
├── prompts/
│   ├── advanced_prompts.py       # Prompt strategies
│   ├── post_process_v5.py        # Production post-processing
│   └── post_process_v7.py        # Logic error fixes
├── results/                      # Experiment results
└── logs/                         # Execution logs
```

---

## Advanced Usage

### Customize Experiments

Edit `config.yml`:

```yaml
experiments:
  - name: "my_experiment"
    enabled: true
    prompt_strategy: "fewshot_v2_devise"
    postprocess_strategy: "post_v7"
    temperature: 0.0
```

**Available Prompt Strategies**:

*Chat Mode (Instruct Model):*
- `fewshot_v1` - **Best performer (54.3%)** - Direct instruction, AST post-processing
- `fewshot_v1_cot` - CoT reasoning with algorithm verification (untested)

*Completion Mode (Base Model):*
- `fewshot_v2_devise` - **Best for Base (34.8%)** - 6 examples + devise instruction
- `fewshot_v2` - 6 examples + CoT instructions
- `fewshot_v1` - 6 examples only
- `minimal` - Problem statement only (baseline)

**Available Post-processing**:
- `auto` - Strategy-specific (loads from strategy folder)
- `post_chat` - AST-based extraction for chat models
- `post_v7` - Logic error pattern fixes
- `post_v5` - Production-ready pipeline (imports, dependencies, truncation fixes)
- `post_v1` - Basic crash fixes
- `none` - Raw model output

---

### Test on Subset

```yaml
dataset:
  max_samples: 10  # Test on first 10 problems
```

---

### Service Management

```bash
./scripts/manage_services.sh start    # Start vLLM server
./scripts/manage_services.sh stop     # Stop server
./scripts/manage_services.sh restart  # Restart server
./scripts/manage_services.sh logs     # View logs
```

---

## Performance Optimization

**Total pipeline time**: 15-30 seconds for 164 problems

### 1. vLLM Inference Engine
- PagedAttention for efficient KV cache management
- Continuous batching
- Throughput: 15-30 problems/second

### 2. Parallel Inference
- 16 concurrent workers (ThreadPoolExecutor)
- Non-blocking I/O
- Config: `inference.num_workers: 16`

### 3. Multiprocess Evaluation
- CPU-count parallel workers
- Isolated subprocess execution
- Throughput: ~36 problems/second

**Pipeline breakdown**: ~5-10s inference + ~5-10s evaluation + ~5-10s I/O

---

### Scaling Strategies

**Horizontal Scaling**:
- Docker Compose for multi-instance deployment
- Load balancers
- Stateless workers

**Vertical Scaling**:
- Tune worker counts based on hardware
- PagedAttention optimization
- Batch size tuning

**Future Improvements**:
- Kubernetes orchestration
- Ray Serve for distributed inference
- Redis caching for repeated queries

---

## Key Technologies

- **vLLM** - High-performance LLM inference engine
- **Docker** - Containerization and deployment
- **HumanEval** - 164 Python programming problems benchmark
- **PyTorch** - Deep learning framework
- **Transformers** - HuggingFace model library

---

## License

Educational and evaluation purposes.

---

**Model**: [Qwen/Qwen2.5-Coder-0.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-0.5B)

**Benchmark**: [HumanEval](https://github.com/openai/human-eval)

#!/bin/bash
# Script to compare multiple prompt strategies with post_process_v5

echo "============================================================"
echo "Running Prompt Strategy Comparison with post_v5"
echo "============================================================"
echo ""

# Array of prompt strategies to test
PROMPTS=("minimal_v2" "minimal_v3" "minimal_v4" "minimal_v5")

for prompt in "${PROMPTS[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Testing: $prompt with post_v5"
    echo "------------------------------------------------------------"

    # Run inference
    python scripts/inference.py \
        --prompt-strategy "$prompt" \
        --postprocess-strategy "post_v5" \
        --output "results/completions_${prompt}_post_v5.jsonl"

    # Run evaluation
    python scripts/run_evaluation.py \
        --completions "results/completions_${prompt}_post_v5.jsonl" \
        --output "results/eval_${prompt}_post_v5.json" \
        --log "logs/${prompt}_post_v5_all_cases.log"

    echo "Completed: $prompt with post_v5"
done

echo ""
echo "============================================================"
echo "All evaluations complete!"
echo "============================================================"
echo ""
echo "Results summary:"
for prompt in "${PROMPTS[@]}"; do
    if [ -f "results/eval_${prompt}_post_v5.json" ]; then
        pass_rate=$(jq -r '.pass_at_1' "results/eval_${prompt}_post_v5.json" 2>/dev/null || echo "N/A")
        passed=$(jq -r '.passed' "results/eval_${prompt}_post_v5.json" 2>/dev/null || echo "N/A")
        total=$(jq -r '.total' "results/eval_${prompt}_post_v5.json" 2>/dev/null || echo "N/A")
        echo "  $prompt: ${passed}/${total} (${pass_rate})"
    fi
done

#!/bin/bash
# Complete pipeline: inference + evaluation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=================================================="
echo "HumanEval Evaluation Pipeline"
echo "=================================================="

# Parse arguments
TEMPERATURE=${TEMPERATURE:-0.2}
MAX_SAMPLES=${MAX_SAMPLES:-""}
API_URL=${API_URL:-"http://localhost:8000/v1"}

# Check if vLLM server is running
echo "Checking vLLM server..."
if ! curl -s -f "${API_URL}/models" > /dev/null 2>&1; then
    echo "❌ vLLM server not responding at ${API_URL}"
    echo "Start it with: ./scripts/manage_services.sh start"
    exit 1
fi
echo "vLLM server is running"

# Run inference
echo ""
echo "Step 1: Running inference..."
echo "-----------------------------------"
INFERENCE_ARGS="--temperature ${TEMPERATURE} --api-url ${API_URL}"
if [ -n "$MAX_SAMPLES" ]; then
    INFERENCE_ARGS="${INFERENCE_ARGS} --max-samples ${MAX_SAMPLES}"
fi

python scripts/inference.py $INFERENCE_ARGS

# Run evaluation
echo ""
echo "Step 2: Running evaluation..."
echo "-----------------------------------"
python scripts/run_evaluation.py --analyze

echo ""
echo "=================================================="
echo "Pipeline complete!"
echo "=================================================="
echo ""
echo "Results saved to:"
echo "  - results/completions.jsonl"
echo "  - results/evaluation_results.json"

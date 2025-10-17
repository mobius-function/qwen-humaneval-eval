#!/bin/bash

# Helper script to manage vLLM model servers (Base + Instruct) via Docker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Function to check which models are enabled in config.yml
check_enabled_models() {
    if ! command -v python3 &> /dev/null; then
        echo "Error: python3 is required to parse config.yml"
        exit 1
    fi

    python3 - <<'EOF'
import yaml
import sys

try:
    with open('config.yml', 'r') as f:
        config = yaml.safe_load(f)

    completion_enabled = config['vllm'].get('completion_model_enabled', False)
    instruct_enabled = config['vllm'].get('instruct_model_enabled', False)

    # Validate: only one can be enabled
    if completion_enabled and instruct_enabled:
        print("ERROR: Both completion and instruct models are enabled. Only one model can be enabled at a time.")
        sys.exit(1)
    elif not completion_enabled and not instruct_enabled:
        print("ERROR: No models are enabled. Please enable either completion_model_enabled or instruct_model_enabled in config.yml")
        sys.exit(1)

    # Output which models to start
    if completion_enabled:
        print("vllm-base")
    elif instruct_enabled:
        print("vllm-instruct")

except Exception as e:
    print(f"ERROR: Failed to parse config.yml: {e}")
    sys.exit(1)
EOF
}

wait_for_server() {
    local port=$1
    local name=$2
    echo "Waiting for $name to be ready on port $port..."
    for i in {1..60}; do
        if curl -s http://localhost:$port/v1/models > /dev/null 2>&1; then
            echo "✓ $name is ready!"
            return 0
        fi
        sleep 3
        printf "."
    done
    echo ""
    echo "✗ $name did not start in time"
    return 1
}

case "$1" in
    start)
        echo "Checking config.yml for enabled models..."
        ENABLED_SERVICE=$(check_enabled_models)

        if [ $? -ne 0 ]; then
            echo "$ENABLED_SERVICE"
            exit 1
        fi

        if [ "$ENABLED_SERVICE" = "vllm-base" ]; then
            echo "Starting Base model (completion API) via Docker..."
            docker compose up -d vllm-base
            echo ""
            echo "Model is starting (this takes 1-2 minutes for download & loading)..."
            wait_for_server 8000 "Base model"
            echo ""
            echo "✓ Base model server is running!"
            echo "  - Completion API: http://localhost:8000 (Base model)"
        elif [ "$ENABLED_SERVICE" = "vllm-instruct" ]; then
            echo "Starting Instruct model (chat API) via Docker..."
            docker compose up -d vllm-instruct
            echo ""
            echo "Model is starting (this takes 1-2 minutes for download & loading)..."
            wait_for_server 8001 "Instruct model"
            echo ""
            echo "✓ Instruct model server is running!"
            echo "  - Chat API: http://localhost:8001 (Instruct model)"
        fi
        ;;

    stop)
        echo "Stopping all vLLM services..."
        docker compose down
        echo "Services stopped."
        ;;

    restart)
        echo "Checking config.yml for enabled models..."
        ENABLED_SERVICE=$(check_enabled_models)

        if [ $? -ne 0 ]; then
            echo "$ENABLED_SERVICE"
            exit 1
        fi

        echo "Restarting $ENABLED_SERVICE..."
        docker compose restart $ENABLED_SERVICE
        echo ""

        if [ "$ENABLED_SERVICE" = "vllm-base" ]; then
            wait_for_server 8000 "Base model"
        elif [ "$ENABLED_SERVICE" = "vllm-instruct" ]; then
            wait_for_server 8001 "Instruct model"
        fi
        echo "✓ Service restarted"
        ;;

    logs)
        ENABLED_SERVICE=$(check_enabled_models 2>/dev/null || echo "")

        if [ -z "$ENABLED_SERVICE" ]; then
            echo "Showing logs from all services (Ctrl+C to exit)..."
            docker compose logs -f vllm-base vllm-instruct
        else
            echo "Showing logs from $ENABLED_SERVICE (Ctrl+C to exit)..."
            docker compose logs -f $ENABLED_SERVICE
        fi
        ;;

    status)
        echo "Checking vLLM services status..."
        ENABLED_SERVICE=$(check_enabled_models 2>/dev/null || echo "")

        if [ -n "$ENABLED_SERVICE" ]; then
            echo "Enabled model (from config.yml): $ENABLED_SERVICE"
        fi

        echo ""
        docker compose ps vllm-base vllm-instruct
        echo ""

        if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
            echo "✓ Base model (port 8000): API responding"
            curl -s http://localhost:8000/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Model:', d['data'][0]['id'])" 2>/dev/null
        else
            echo "✗ Base model (port 8000): API not responding"
        fi

        if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
            echo "✓ Instruct model (port 8001): API responding"
            curl -s http://localhost:8001/v1/models | python3 -c "import sys,json; d=json.load(sys.stdin); print('  Model:', d['data'][0]['id'])" 2>/dev/null
        else
            echo "✗ Instruct model (port 8001): API not responding"
        fi
        ;;

    test)
        echo "Testing Completion API (port 8000)..."
        curl -s -X POST http://localhost:8000/v1/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "Qwen/Qwen2.5-Coder-0.5B",
                "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n    ",
                "max_tokens": 20,
                "temperature": 0.0,
                "stop": ["\n\n"]
            }' | python3 -c "import sys,json; print('Response:', json.load(sys.stdin)['choices'][0]['text'])"

        echo ""
        echo "Testing Chat API (port 8001)..."
        curl -s -X POST http://localhost:8001/v1/chat/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                "messages": [
                    {"role": "system", "content": "You are a Python expert."},
                    {"role": "user", "content": "def multiply(a, b):\n    \"\"\"Multiply two numbers.\"\"\"\n"}
                ],
                "max_tokens": 20,
                "temperature": 0.0
            }' | python3 -c "import sys,json; print('Response:', json.load(sys.stdin)['choices'][0]['message']['content'])"
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|logs|test}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the enabled vLLM model server (configured in config.yml)"
        echo "  stop    - Stop all services"
        echo "  restart - Restart the enabled service"
        echo "  status  - Check if services are running and show enabled model"
        echo "  logs    - Show real-time logs from the enabled server"
        echo "  test    - Test both API endpoints"
        echo ""
        echo "Note: Only one model can be enabled at a time in config.yml:"
        echo "  - Set vllm.completion_model_enabled: true for Base model (completion API)"
        echo "  - Set vllm.instruct_model_enabled: true for Instruct model (chat API)"
        exit 1
        ;;
esac

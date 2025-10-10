#!/bin/bash

# Helper script to manage vLLM and evaluation services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

case "$1" in
    start)
        echo "Starting vLLM model server..."
        docker compose up -d vllm-server
        echo "Waiting for server to be healthy..."
        echo "This may take 1-2 minutes for model download and loading..."
        sleep 30
        echo "Server started! API available at http://localhost:8000"
        ;;

    stop)
        echo "Stopping all services..."
        docker compose down
        echo "Services stopped."
        ;;

    restart)
        echo "Restarting services..."
        docker compose restart
        ;;

    logs)
        docker compose logs -f vllm-server
        ;;

    test)
        echo "Testing vLLM API endpoint..."
        curl -X POST http://localhost:8000/v1/completions \
            -H "Content-Type: application/json" \
            -d '{
                "model": "Qwen/Qwen2.5-Coder-0.5B",
                "prompt": "def hello_world():",
                "max_tokens": 50,
                "temperature": 0.7
            }'
        ;;

    eval)
        echo "Running evaluation..."
        docker compose up eval-sandbox
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|logs|test|eval}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the vLLM model server"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  logs    - Show vLLM server logs"
        echo "  test    - Test the API endpoint"
        echo "  eval    - Run HumanEval evaluation"
        exit 1
        ;;
esac

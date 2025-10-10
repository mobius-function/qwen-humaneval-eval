# vLLM Setup Guide

This document provides instructions for setting up and using vLLM to serve the Qwen/Qwen2.5-Coder-0.5B model.

## Quick Start

### GPU Mode (Recommended)

1. **Start the server:**
```bash
./scripts/manage_services.sh start
```

2. **Check server health:**
```bash
curl http://localhost:8000/health
```

3. **Test the API:**
```bash
./scripts/manage_services.sh test
```

### CPU Mode

For systems without GPU:

```bash
# Use the CPU-specific docker-compose file
docker compose -f docker-compose.cpu.yml up -d vllm-server
```

Or modify `docker-compose.yml` to comment out the GPU deployment section:
```yaml
# Comment out these lines for CPU mode:
# deploy:
#   resources:
#     reservations:
#       devices:
#         - driver: nvidia
#           count: 1
#           capabilities: [gpu]
```

## Configuration Options

### Dockerfile.vllm Parameters

The vLLM server accepts various command-line arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--model` | Qwen/Qwen2.5-Coder-0.5B | Model to load |
| `--host` | 0.0.0.0 | Server host |
| `--port` | 8000 | Server port |
| `--max-model-len` | 2048 | Maximum sequence length |
| `--gpu-memory-utilization` | 0.9 | GPU memory to use (0.0-1.0) |
| `--dtype` | auto | Data type (auto, half, float16, bfloat16) |

### Performance Tuning

#### For Maximum Throughput:
```dockerfile
CMD python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --max-num-batched-tokens 4096 \
    --max-num-seqs 256 \
    --gpu-memory-utilization 0.95 \
    --enable-chunked-prefill \
    --dtype bfloat16
```

#### For Lower Memory Usage:
```dockerfile
CMD python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --max-model-len 1024 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 64
```

#### For CPU Mode:
```dockerfile
CMD python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --dtype half \
    --max-model-len 2048 \
    --max-num-seqs 16
```

## API Usage

### OpenAI-Compatible Completions API

vLLM provides an OpenAI-compatible API endpoint:

**Endpoint:** `http://localhost:8000/v1/completions`

**Example Request:**
```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-Coder-0.5B",
    "prompt": "def fibonacci(n):\n    \"\"\"Generate fibonacci sequence.\"\"\"\n",
    "max_tokens": 256,
    "temperature": 0.2,
    "top_p": 0.95,
    "stop": ["\nclass ", "\ndef ", "\n#"]
  }'
```

**Response:**
```json
{
  "id": "cmpl-...",
  "object": "text_completion",
  "created": 1234567890,
  "model": "Qwen/Qwen2.5-Coder-0.5B",
  "choices": [
    {
      "text": "    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
      "index": 0,
      "logprobs": null,
      "finish_reason": "stop"
    }
  ]
}
```

### Available Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/v1/models` | GET | List available models |
| `/v1/completions` | POST | Generate completions |
| `/version` | GET | vLLM version |
| `/metrics` | GET | Prometheus metrics |

## Troubleshooting

### Server Won't Start

**Check GPU availability:**
```bash
nvidia-smi
```

**Check Docker GPU support:**
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**View server logs:**
```bash
docker compose logs -f vllm-server
```

### Out of Memory Errors

1. **Reduce GPU memory utilization:**
   - Change `--gpu-memory-utilization 0.9` to `0.7` or `0.5`

2. **Reduce max sequence length:**
   - Change `--max-model-len 2048` to `1024` or `512`

3. **Use quantization:**
   ```dockerfile
   --quantization awq  # or gptq
   ```

### Slow Inference

1. **Enable optimizations:**
   ```dockerfile
   --enable-chunked-prefill \
   --enable-prefix-caching \
   --use-v2-block-manager
   ```

2. **Use better dtype:**
   ```dockerfile
   --dtype bfloat16  # instead of auto
   ```

3. **Increase batch size:**
   ```dockerfile
   --max-num-batched-tokens 4096 \
   --max-num-seqs 128
   ```

### Connection Refused

1. **Wait for server to be healthy:**
   ```bash
   # Check health status
   docker compose ps

   # Wait for healthy status
   while ! curl -f http://localhost:8000/health 2>/dev/null; do
     sleep 5
     echo "Waiting for server..."
   done
   ```

2. **Check port binding:**
   ```bash
   netstat -tulpn | grep 8000
   ```

## Advanced Configuration

### Multi-GPU Setup

For multiple GPUs, use tensor parallelism:

```dockerfile
CMD python3 -m vllm.entrypoints.openai.api_server \
    --model ${MODEL_NAME} \
    --tensor-parallel-size 2  # Use 2 GPUs
```

### Custom Model Path

To use a local model:

```yaml
volumes:
  - ./models:/models
  - huggingface_cache:/root/.cache/huggingface
environment:
  - MODEL_NAME=/models/qwen-coder-0.5b
```

### Environment Variables

Set in `docker-compose.yml`:

```yaml
environment:
  - VLLM_LOGGING_LEVEL=INFO
  - VLLM_WORKER_MULTIPROC_METHOD=spawn
  - CUDA_VISIBLE_DEVICES=0  # Use specific GPU
```

## Monitoring

### View Metrics

vLLM exposes Prometheus metrics:

```bash
curl http://localhost:8000/metrics
```

Key metrics:
- `vllm:num_requests_running` - Active requests
- `vllm:gpu_cache_usage_perc` - GPU cache utilization
- `vllm:time_to_first_token_seconds` - TTFT latency
- `vllm:time_per_output_token_seconds` - Generation speed

### Performance Monitoring

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor Docker stats
docker stats vllm-server
```

## References

- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)
- [OpenAI API Compatibility](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)

#!/usr/bin/env bash
# 用 vLLM 服务被测/被训练模型（评测与 GRPO 前置）
set -euo pipefail
MODEL="${1:?用法: serve_model.sh <model_path_or_name> [port]}"
PORT="${2:-8000}"
exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name aftersale-agent \
  --port "$PORT" \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.85

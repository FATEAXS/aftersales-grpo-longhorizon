#!/usr/bin/env bash
# Baseline 评测：直接评测基座模型（先 serve_model.sh，另需环境服务）
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-${BASE_MODEL:-Qwen/Qwen3-1.7B}}"
python scripts/evaluate_aftersale.py --actor api --name baseline \
  --base-url "${LLM_BASE_URL:-http://127.0.0.1:8000/v1}" --model aftersale-agent

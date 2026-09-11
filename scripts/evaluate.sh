#!/usr/bin/env bash
# 评测指定阶段产物：evaluate.sh <name>（模型经 serve_model.sh 提供服务）
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?用法: evaluate.sh <baseline|sft|grpo>}"
python scripts/evaluate_aftersale.py --actor api --name "$NAME" \
  --base-url "${LLM_BASE_URL:-http://127.0.0.1:8000/v1}" --model aftersale-agent

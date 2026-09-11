#!/usr/bin/env bash
# 导出 veRL FSDP checkpoint 的 actor 权重为 HF 格式
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:?用法: export_grpo.sh <global_step_dir/actor> <output_dir>}"
OUT="${2:?用法: export_grpo.sh <global_step_dir/actor> <output_dir>}"
python -m verl.model_merger merge \
  --backend fsdp \
  --local_dir "$CKPT" \
  --target_dir "$OUT"
echo "exported -> $OUT"

#!/usr/bin/env bash
# LoRA SFT：依赖前置 — data/sft/ 已由 collect_sft_data.py 生成
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train_lora_sft.py \
  --model "${BASE_MODEL:?需要 BASE_MODEL}" \
  --output outputs/models/sft-lora
python scripts/merge_lora.py \
  --base-model "${BASE_MODEL}" \
  --adapter outputs/models/sft-lora \
  --output outputs/models/sft-merged

#!/usr/bin/env bash
# veRL 在线 GRPO（Linux + GPU）。默认对齐 24GB 单卡配置。
set -euo pipefail
cd "$(dirname "$0")/.."
export AFTERSALE_GRPO_ROOT="$PWD"
export AFTERSALE_MODEL_PATH="${AFTERSALE_MODEL_PATH:?需要 AFTERSALE_MODEL_PATH（SFT 合并模型）}"
export AFTERSALE_TRAIN_FILE="$PWD/data/grpo/train.parquet"
export AFTERSALE_VAL_FILE="$PWD/data/grpo/validation.parquet"
export AFTERSALE_BASE_URL="${AFTERSALE_BASE_URL:-http://127.0.0.1:5800}"
export AFTERSALE_GRPO_STEPS="${AFTERSALE_GRPO_STEPS:-120}"
export AFTERSALE_GRPO_SAVE_FREQ="${AFTERSALE_GRPO_SAVE_FREQ:-20}"
export AFTERSALE_EXPERIMENT_NAME="${AFTERSALE_EXPERIMENT_NAME:-grpo_v1}"
exec python scripts/train_grpo.py "$@"

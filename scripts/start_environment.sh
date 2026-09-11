#!/usr/bin/env bash
# 启动 AftersaleSimulator 环境服务（默认 127.0.0.1:5800）
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
HOST="${AFTERSALE_HOST:-127.0.0.1}"
PORT="${AFTERSALE_PORT:-5800}"
echo "AftersaleSimulator -> http://${HOST}:${PORT}"
exec python -m aftersales_grpo.simulator.server --host "$HOST" --port "$PORT"

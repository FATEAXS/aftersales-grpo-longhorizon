#!/usr/bin/env bash
# 对已有评测产物补生成/刷新 HTML 报告
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:?用法: report.sh <name>}"
python scripts/build_eval_report.py --summary "outputs/evaluation/$NAME/summary.json" \
  --output "outputs/evaluation/$NAME/report.html" --title "售后 Agent 评测 · $NAME"

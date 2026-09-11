#!/usr/bin/env bash
# 汇总所有阶段：单模型报告 + 跨阶段对比表
set -euo pipefail
cd "$(dirname "$0")/.."
for dir in outputs/evaluation/*/; do
  name="$(basename "$dir")"
  if [ -f "$dir/summary.json" ] && [ ! -f "$dir/report.html" ]; then
    bash scripts/report.sh "$name" || true
  fi
done
python scripts/build_comparison.py --root outputs/evaluation --output experiments/comparison.md
echo "comparison -> experiments/comparison.md"

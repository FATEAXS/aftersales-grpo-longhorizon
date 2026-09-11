#!/usr/bin/env bash
# 安装核心依赖（训练 extras 在 GPU 机器上按需安装）
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pip install -e ".[dev]"
python -c "import fastapi, httpx; print('core deps OK')"
echo "训练依赖（Linux + GPU）: pip install -e '.[training]' 或 '.[grpo]'"

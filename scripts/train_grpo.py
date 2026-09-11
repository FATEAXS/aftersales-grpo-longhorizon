#!/usr/bin/env python3
"""veRL GRPO 训练入口：校验依赖与数据后，转发到 veRL 标准主循环。

真实训练在 Linux + GPU 机器上执行（scripts/grpo.sh 封装了全部环境变量）：
  python scripts/train_grpo.py trainer.total_training_steps=120
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_runtime() -> None:
    missing = []
    try:
        import verl  # noqa: F401
    except ImportError:
        missing.append("verl==0.8.0")
    try:
        import vllm  # noqa: F401
    except ImportError:
        missing.append("vllm")
    try:
        import torch  # noqa: F401

        if not torch.cuda.is_available():
            print("[warn] CUDA 不可用，GRPO 需要在 GPU 机器上运行", flush=True)
    except ImportError:
        missing.append("torch")
    if missing:
        raise SystemExit("缺少训练依赖: " + ", ".join(missing) + "。请先 `pip install -e .[grpo]`")


def main() -> None:
    parser = argparse.ArgumentParser(description="售后 GRPO 训练（veRL 0.8）")
    parser.add_argument("override", nargs="*", help="Hydra 覆盖项，如 trainer.total_training_steps=20")
    args = parser.parse_args()

    check_runtime()

    required = {
        "AFTERSALE_MODEL_PATH": "基座/合并模型路径",
        "AFTERSALE_TRAIN_FILE": "训练 parquet",
        "AFTERSALE_VAL_FILE": "验证 parquet",
    }
    missing_env = [f"{k}({v})" for k, v in required.items() if not os.environ.get(k)]
    if missing_env:
        raise SystemExit("缺少环境变量: " + ", ".join(missing_env) + "。建议使用 scripts/grpo.sh 启动。")
    os.environ.setdefault("AFTERSALE_GRPO_ROOT", str(ROOT))

    config_dir = str(ROOT / "configs")
    command = [
        sys.executable, "-m", "verl.trainer.main_ppo",
        "--config-path", config_dir,
        "--config-name", "grpo",
        *args.override,
    ]
    print("[train_grpo]", " ".join(command), flush=True)
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()

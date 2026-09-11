#!/usr/bin/env python3
"""以固定种子重建世界与任务数据（与 scripts/build_world.sh 等价）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.simulator.generator import generate_everything

if __name__ == "__main__":
    manifest = generate_everything(
        ROOT / "environments" / "aftersalesim" / "data", ROOT / "data"
    )
    print(json_dumps := __import__("json").dumps(manifest["counts"], ensure_ascii=False))

#!/usr/bin/env python3
"""校验 configs/tools.json 与代码内工具契约一致。"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.environment.tools import TOOLS

if __name__ == "__main__":
    configured = json.loads((ROOT / "configs/tools.json").read_text(encoding="utf-8"))["tools"]
    expected = [t["tool_schema"] for t in configured]
    assert expected == TOOLS, "configs/tools.json 与环境工具契约不一致，请运行 dump_tools_json 重新生成"
    print(f"tools contract OK ({len(TOOLS)} tools)")

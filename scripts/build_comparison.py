#!/usr/bin/env python3
"""汇总各阶段 summary.json 生成跨阶段对比 Markdown。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.evaluation.comparison import write_comparison

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--output", type=Path, default=ROOT / "experiments/comparison.md")
    args = parser.parse_args()
    paths = {}
    for name in ["scripted-ref", "baseline", "sft", "grpo"]:
        p = args.root / name / "summary.json"
        if p.exists():
            paths[name] = p
    if not paths:
        raise SystemExit("未找到任何 summary.json")
    print(f"comparison -> {write_comparison(paths, args.output)}")

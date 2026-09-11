#!/usr/bin/env python3
"""对已有评测产物生成单文件 HTML 报告。"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.evaluation.report import write_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    print(f"report -> {write_report(args.summary, args.output, args.title)}")

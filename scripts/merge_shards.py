#!/usr/bin/env python3
"""合并分片评测产物：concat trajectories + 重新汇总 + 报告。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.evaluation.metrics import guard_artifacts, summarize  # noqa: E402
from aftersales_grpo.evaluation.report import write_report  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="合并分片评测产物")
    parser.add_argument("--name", required=True, help="如 baseline（合并 baseline-shard* -> baseline）")
    parser.add_argument("--root", type=Path, default=ROOT / "outputs/evaluation")
    args = parser.parse_args()

    shard_dirs = sorted(args.root.glob(f"{args.name}-shard*"))
    if not shard_dirs:
        raise SystemExit(f"未找到分片目录: {args.root}/{args.name}-shard*")

    records = []
    for shard in shard_dirs:
        with open(shard / "trajectories.jsonl", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
    records.sort(key=lambda r: r["task_id"])
    # 任务去重（防止重复评测）
    seen = set()
    unique = []
    for record in records:
        if record["task_id"] not in seen:
            seen.add(record["task_id"])
            unique.append(record)

    merged_dir = args.root / args.name
    merged_dir.mkdir(parents=True, exist_ok=True)
    with open(merged_dir / "trajectories.jsonl", "w", encoding="utf-8") as fh:
        for record in unique:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    actor_name = unique[0].get("actor", args.name) if unique else args.name
    summary = summarize(unique, actor_name=actor_name)
    with open(merged_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)

    guard = guard_artifacts(merged_dir / "trajectories.jsonl")
    if not guard["clean"]:
        print(json.dumps(guard, ensure_ascii=False))
        raise SystemExit("盲评守卫失败")
    report = write_report(
        merged_dir / "summary.json",
        merged_dir / "report.html",
        title=f"售后 Agent 评测 · {args.name}",
    )
    print(json.dumps({
        "tasks": len(unique),
        "strict_success_rate": summary["panel_a_terminal"]["strict_success_rate"],
        "mean_reward": summary["panel_a_terminal"]["mean_reward"],
    }, ensure_ascii=False))
    print(f"report -> {report}")


if __name__ == "__main__":
    main()

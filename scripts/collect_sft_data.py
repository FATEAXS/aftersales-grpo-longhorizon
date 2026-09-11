#!/usr/bin/env python3
"""采集教师轨迹并生成 SFT 数据集（含验收审计 metadata）。

两种教师：
- scripted（默认）：确定性参考教师，零成本可复现，用于冷启动与流水线验证
- api：OpenAI 兼容接口的 LLM 教师（--teacher api --model ... --llm-base-url ...）

验收口径：Reward v1 终局 total >= --min-reward（默认 1.0）且未触发硬门槛。
输出：
  data/sft/train.jsonl / validation.jsonl   （OpenAI 消息格式轨迹）
  data/sft/metadata.json                    （采集审计：接受率、丢弃原因）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.collection.sft import (  # noqa: E402
    ApiTeacher,
    load_task_ids,
    run_api_teacher,
    run_scripted_teacher,
)
from aftersales_grpo.environment.client import AftersaleEnv  # noqa: E402
from aftersales_grpo.environment.tools import SYSTEM_PROMPT, TOOLS  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="采集售后教师轨迹生成 SFT 数据")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/grpo/train.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/sft")
    parser.add_argument("--teacher", choices=("scripted", "api"), default="scripted")
    parser.add_argument("--llm-base-url", default=None)
    parser.add_argument("--llm-api-key", default=os_environ_key())
    parser.add_argument("--model", default=None, help="API 教师模型名")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-reward", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=4)
    return parser.parse_args()


def os_environ_key():
    import os

    return os.environ.get("LLM_API_KEY", "")


def collect_one(env, task_id, args):
    if args.teacher == "scripted":
        return run_scripted_teacher(env, task_id)
    teacher = ApiTeacher(
        base_url=args.llm_base_url,
        api_key=args.llm_api_key,
        model=args.model,
    )
    try:
        return run_api_teacher(env, task_id, teacher)
    finally:
        teacher.close()


def main() -> None:
    args = parse_args()
    env = AftersaleEnv()
    if env.health().get("status") != "ok":
        raise SystemExit("环境服务未启动：bash scripts/start_environment.sh")

    task_ids = load_task_ids(args.tasks, limit=args.limit)
    started = time.time()

    accepted_rows = []
    drop_reasons: Counter = Counter()
    reward_hist: Counter = Counter()
    teacher_name = args.teacher if args.teacher == "scripted" else f"api:{args.model}"

    for i, task_id in enumerate(task_ids):
        record = collect_one(env, task_id, args)
        breakdown = record["reward"]
        total = float(breakdown.get("total", 0.0))
        reward_hist[round(total, 2)] += 1
        if breakdown.get("terminal", "").startswith("hard_gate"):
            drop_reasons["hard_gate"] += 1
        elif record["steps"] == 0:
            drop_reasons["empty"] += 1
        elif total < args.min_reward:
            drop_reasons["below_min_reward"] += 1
        else:
            accepted_rows.append(
                {
                    "trajectory_id": f"{task_id}:{teacher_name}",
                    "task_id": task_id,
                    "teacher": teacher_name,
                    "reward": total,
                    "steps": record["steps"],
                    "messages": record["messages"],
                    "tools": TOOLS,
                    "system_prompt": SYSTEM_PROMPT,
                }
            )
        if (i + 1) % 50 == 0:
            print(f"[{i + 1}/{len(task_ids)}] accepted={len(accepted_rows)}", flush=True)

    env_close = getattr(env, "close", None)
    if env_close:
        env_close()

    # 稳定划分：8:1 训练/验证（task_id 级别，零重叠）
    keyed = sorted(
        accepted_rows,
        key=lambda r: hashlib.sha256(f"{r['task_id']}".encode()).hexdigest(),
    )
    val_count = max(1, round(len(keyed) * 0.1)) if len(keyed) >= 10 else 0
    val_rows = keyed[:val_count]
    train_rows = keyed[val_count:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train.jsonl", train_rows), ("validation.jsonl", val_rows)]:
        with open(args.output_dir / name, "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    metadata = {
        "teacher": teacher_name,
        "min_reward": args.min_reward,
        "raw_trajectories": len(task_ids),
        "accepted": len(accepted_rows),
        "acceptance_rate": round(len(accepted_rows) / max(1, len(task_ids)), 4),
        "drop_counts": dict(drop_reasons),
        "reward_distribution": {str(k): v for k, v in sorted(reward_hist.items())},
        "train": len(train_rows),
        "validation": len(val_rows),
        "elapsed_seconds": round(time.time() - started, 1),
        "seed_note": "scripted 教师为确定性策略；同种子世界+任务下可逐字节复现",
    }
    with open(args.output_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=1)
    print(json.dumps(metadata, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

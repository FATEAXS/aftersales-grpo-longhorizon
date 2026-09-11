#!/usr/bin/env python3
"""把任务卡转成 veRL GRPO 训练数据（JSONL 任务卡保留，parquet 供 veRL 读取）。

同时落线"红线协议"：风控场景（requires_escalation）按 --redline-repeat 倍数
复制进训练集，使合规关键任务在在线采样中被过采样；评测集永不过采样。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.environment.client import AftersaleEnv  # noqa: E402
from aftersales_grpo.environment.tools import initial_messages  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="准备 veRL GRPO 数据")
    parser.add_argument("--env-url", default="http://127.0.0.1:5800")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data/grpo")
    parser.add_argument("--redline-repeat", type=int, default=2,
                        help="风控任务在训练集中的复制倍数（1 = 不过采样）")
    parser.add_argument("--no-parquet", action="store_true")
    return parser.parse_args()


def redline_task_ids(split: str) -> set[str]:
    """读取服务器本地私有事实，提取合规关键任务（P0-1:拒绝类 + 风控类）。

    v1 只过采样风控（requires_escalation）；v2 把"应拒绝"场景一并纳入——
    它们是 SFT 模型违规办理率（56%）的主要来源。
    """
    facts_path = ROOT / "environments/aftersalesim/data/tasks" / f"{split}.jsonl"
    ids = set()
    with open(facts_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            fact = json.loads(line)
            if fact.get("requires_escalation") or fact.get("ineligible"):
                ids.add(fact["task_id"])
    return ids


def main() -> None:
    args = parse_args()
    env = AftersaleEnv(base_url=args.env_url)

    for split in ["train", "validation"]:
        task_ids = [
            json.loads(line)["task_id"]
            for line in open(args.out_dir / f"{split}.jsonl", encoding="utf-8")
            if line.strip()
        ]
        rows = []
        for index, task_id in enumerate(task_ids):
            task = env.get_task(task_id)
            opening = {
                "user_id": task["user_id"],
                "today": task["today"],
                "user_message": task["opening_message"],
            }
            rows.append(
                {
                    "data_source": "aftersalesimulator",
                    "prompt": initial_messages(opening),
                    "ability": "aftersales",
                    "reward_model": {"style": "rule", "ground_truth": None},
                    "extra_info": {"split": split, "index": str(index), "task_id": task_id},
                }
            )

        redline_count = 0
        if split == "train" and args.redline_repeat > 1:
            redline = redline_task_ids(split)
            id_to_row = {r["extra_info"]["task_id"]: r for r in rows}
            for task_id in redline:
                base = id_to_row.get(task_id)
                if base is None:
                    continue
                for k in range(1, args.redline_repeat):
                    rows.append(
                        {
                            **base,
                            "extra_info": {
                                **base["extra_info"],
                                "index": f"{base['extra_info']['index']}r{k}",
                            },
                        }
                    )
                    redline_count += 1

        if not args.no_parquet:
            import pandas as pd

            pd.DataFrame(rows).to_parquet(args.out_dir / f"{split}.parquet")
        note = f"（含红线过采样 {redline_count} 行）" if redline_count else ""
        print(f"{split}: {len(rows)} rows{note}")

    env.close()


if __name__ == "__main__":
    main()

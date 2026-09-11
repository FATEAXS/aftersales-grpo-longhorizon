#!/usr/bin/env python3
"""RFT 自提升闭环（P1-7）：用当前模型重新采集轨迹,过滤满分回流 SFT 数据池。

用法（在 GPU 机器上,先 serve 模型 + 起环境服务）:
  python scripts/rft_collect.py --base-url http://127.0.0.1:8000/v1 \
      --model aftersale-agent --append-to data/sft/train.jsonl --limit 200

验收口径与教师采集一致（Reward v1 total >= 1.0）,回流行带 rft_source 标记,
下一轮 SFT 时可按来源加权。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.collection.sft import load_task_ids  # noqa: E402
from aftersales_grpo.environment.client import AftersaleEnv  # noqa: E402
from aftersales_grpo.environment.tools import SYSTEM_PROMPT, TOOLS  # noqa: E402
from aftersales_grpo.evaluation.rollout import ApiActor, run_episode  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="RFT 自提升数据回流")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/grpo/train.jsonl")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="aftersale-agent")
    parser.add_argument("--append-to", type=Path, default=ROOT / "data/sft/train.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--env-url", default="http://127.0.0.1:5800")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    actor = ApiActor(base_url=args.base_url, model=args.model)
    env_local = __import__("threading").local()

    def _env():
        if not hasattr(env_local, "env"):
            env_local.env = AftersaleEnv(base_url=args.env_url)
        return env_local.env

    task_ids = load_task_ids(args.tasks, limit=args.limit)
    accepted = []

    def _one(task_id: str):
        episode = run_episode(_env(), task_id, actor)
        return episode

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_one, tid): tid for tid in task_ids}
        for future in as_completed(futures):
            episode = future.result()
            if episode.get("success") and not episode.get("violations"):
                accepted.append(episode)

    rows = []
    for episode in accepted:
        rows.append({
            "trajectory_id": f"{episode['task_id']}:rft",
            "task_id": episode["task_id"],
            "teacher": f"rft:{args.model}",
            "reward": episode["reward"],
            "steps": episode["steps"],
            "messages": episode["messages"],
            "tools": TOOLS,
            "system_prompt": SYSTEM_PROMPT,
        })

    with open(args.append_to, "a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"RFT: {len(accepted)}/{len(task_ids)} 满分轨迹已回流 -> {args.append_to}")


if __name__ == "__main__":
    main()

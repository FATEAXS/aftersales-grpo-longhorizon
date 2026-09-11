#!/usr/bin/env python3
"""评测入口：在 Final-100 Clean（或其他任务集）上评测一个 Actor。

用法：
  # 被测模型 = OpenAI 兼容服务（vLLM serve）
  python scripts/evaluate_aftersale.py --actor api --model Qwen3-1.7B-SFT \
      --base-url http://127.0.0.1:8000/v1 --name sft

  # 参考策略（流水线自检/教师上界）
  python scripts/evaluate_aftersale.py --actor scripted --name scripted-ref
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.evaluation.artifacts import guard_blind_trajectory  # noqa: E402
from aftersales_grpo.evaluation.metrics import guard_artifacts  # noqa: E402
from aftersales_grpo.evaluation.report import write_report  # noqa: E402
from aftersales_grpo.evaluation.rollout import ApiActor, run_evaluation  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="售后 Agent 评测")
    parser.add_argument("--actor", choices=("api", "scripted"), default="api")
    parser.add_argument("--name", required=True, help="评测产物目录名，如 baseline/sft/grpo")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/evaluation/tasks.jsonl")
    parser.add_argument("--facts", type=Path,
                        default=ROOT / "environments/aftersalesim/data/tasks/evaluation.jsonl")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/evaluation")
    parser.add_argument("--env-url", default="http://127.0.0.1:5800")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shard", default=None,
                        help="分片，格式 i/N：把任务集切成 N 份只跑第 i 份（多卡并行评测用）")
    parser.add_argument("--samples", type=int, default=1,
                        help="每任务重复采样次数（>1 时启用 pass@k/方差面板，P2-11）")
    parser.add_argument("--judge-url", default=None,
                        help="P2-12: LLM Judge 的 OpenAI 兼容地址（可选）")
    parser.add_argument("--judge-model", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.actor == "api" and not args.model:
        raise SystemExit("--actor api 需要 --model")

    if args.actor == "scripted":
        from aftersales_grpo.evaluation.rollout import ScriptedActor

        actor = ScriptedActor()
    else:
        actor = ApiActor(base_url=args.base_url, model=args.model, api_key=args.api_key)

    tasks_path = args.tasks
    name = args.name
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/", 1))
        shard_path = Path(args.tasks).with_suffix(f".shard{shard_i}of{shard_n}.jsonl")
        lines = [l for l in open(args.tasks, encoding="utf-8") if l.strip()]
        picks = lines[shard_i::shard_n]
        shard_path.write_text("".join(picks), encoding="utf-8")
        tasks_path = shard_path
        name = f"{args.name}-shard{shard_i}"
        print(f"shard {shard_i}/{shard_n}: {len(picks)} tasks")

    if args.samples > 1:
        # P2-11: 每任务复制 K 份（task_id 加后缀），聚合时按原 task_id 归组
        lines = [l for l in open(tasks_path, encoding="utf-8") if l.strip()]
        expanded = []
        for k in range(args.samples):
            for line in lines:
                row = json.loads(line)
                row["task_id"] = f"{row['task_id']}#s{k}"
                expanded.append(json.dumps(row, ensure_ascii=False) + "\n")
        multi_path = Path(tasks_path).with_suffix(f".x{args.samples}.jsonl")
        multi_path.write_text("".join(expanded), encoding="utf-8")
        tasks_path = multi_path
        print(f"multi-sample: {args.samples} 采样 x {len(lines)} 任务")

    output_dir = args.output_root / name
    summary = run_evaluation(
        tasks_path=tasks_path,
        actor=actor,
        output_dir=output_dir,
        workers=args.workers,
        env_url=args.env_url,
        facts_path=args.facts,
    )
    if args.samples > 1:
        from aftersales_grpo.evaluation.metrics import attach_pass_at_k
        attach_pass_at_k(output_dir / "trajectories.jsonl", output_dir / "summary.json", args.samples)
        print("pass@k panel written into summary.json")

    if args.judge_url and args.judge_model:
        from aftersales_grpo.evaluation.judge import judge_replies
        panel = judge_replies(
            output_dir / "trajectories.jsonl", output_dir / "summary.json",
            judge_base_url=args.judge_url, judge_model=args.judge_model,
        )
        print("judge panel:", json.dumps({k: v for k, v in panel.items() if k != "per_task"}, ensure_ascii=False))

    guard = guard_artifacts(output_dir / "trajectories.jsonl")
    if not guard["clean"]:
        print(json.dumps(guard, ensure_ascii=False, indent=1))
        raise SystemExit("盲评守卫发现私有事实泄漏，产物不可用")
    report = write_report(output_dir / "summary.json", output_dir / "report.html",
                          title=f"售后 Agent 评测 · {args.name}")
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("panel") or k in ("actor", "total_tasks")}, ensure_ascii=False, indent=1))
    print(f"report -> {report}")


if __name__ == "__main__":
    main()

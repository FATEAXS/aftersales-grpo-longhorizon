#!/usr/bin/env python3
"""环境冒烟：健康检查 + 一个任务的脚本化跑通（预期 reward=1.0）。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.environment.client import AftersaleEnv
from aftersales_grpo.collection.sft import run_scripted_teacher, load_task_ids

if __name__ == "__main__":
    env = AftersaleEnv()
    health = env.health()
    print("health:", health)
    assert health["status"] == "ok"
    task_id = load_task_ids(ROOT / "data/evaluation/tasks.jsonl", limit=1)[0]
    result = run_scripted_teacher(env, task_id)
    print("task:", task_id, "steps:", result["steps"], "reward:", result["reward"]["total"])

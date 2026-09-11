"""评测 rollout：让被测模型在环境中真实执行售后处理，产出可审计轨迹。

与 SFT 采集共用同一环境契约与消息格式；被测者可以是 vLLM/任意 OpenAI 兼容
服务（--actor api），也可以是脚本化参考策略（--actor scripted，用于流水线
自检与教师上界参考）。

轨迹产物执行"盲评守卫"：不含 TaskFacts（期望解、可接受映射、风控标志等
私有评分事实），指标在评分阶段就地计算后只落聚合安全的字段。
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

from ..collection.sft import load_task_ids
from ..collection.teacher_scripted import ScriptedTeacher
from ..environment.client import AftersaleEnv, render_observation
from ..environment.tools import initial_messages

MAX_TURNS = 25


class ApiActor:
    """OpenAI 兼容接口的模型 Actor（支持 function calling）。"""

    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY", temperature: float = 0.0, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout)
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self.name = model

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers,
            json={
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "temperature": self.temperature,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]

    def close(self) -> None:
        self._client.close()


class ScriptedActor:
    """脚本化参考策略（与 SFT 教师 Same SOP），作为流水线上界参考。"""

    def __init__(self) -> None:
        self.name = "scripted-reference"
        self._local = threading.local()

    def _teacher(self) -> ScriptedTeacher:
        if not hasattr(self._local, "teacher"):
            self._local.teacher = ScriptedTeacher()
        return self._local.teacher

    def next_action(self, opening: dict) -> tuple[str, dict] | None:
        return self._teacher().next_action(opening)

    def observe(self, tool: str, arguments: dict, observation_text: str) -> None:
        self._local.teacher.observe(tool, arguments, observation_text)  # type: ignore[attr-defined]

    def reset(self) -> None:
        if hasattr(self._local, "teacher"):
            self._local.teacher = ScriptedTeacher()


def run_episode(env: AftersaleEnv, task_id: str, actor) -> dict:
    """跑一个任务并返回盲评轨迹 + 就地计算的指标。"""
    from ..simulator.models import TaskFacts

    if isinstance(actor, ScriptedActor):
        actor.reset()
    reset = env.reset(task_id)
    opening = reset["observation"]
    messages = initial_messages(opening)
    steps = 0
    no_tool_turns = 0

    tools_schema = _tools_for_actor(actor)
    while steps < MAX_TURNS:
        if isinstance(actor, ScriptedActor):
            action = actor.next_action(opening)
            tool_calls = (
                [{"id": f"call_{steps + 1:03d}", "function": {"name": action[0], "arguments": json.dumps(action[1], ensure_ascii=False)}}]
                if action
                else None
            )
            assistant_message = None
        else:
            assistant = actor.chat(messages, tools_schema)
            assistant_message = {
                "role": "assistant",
                "content": assistant.get("content"),
                "tool_calls": assistant.get("tool_calls") or [],
            }
            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                no_tool_turns += 1
                text = (assistant.get("content") or "").strip()
                tool_calls = [
                    {
                        "id": f"call_{steps + 1:03d}",
                        "function": {
                            "name": "finish",
                            "arguments": json.dumps(
                                {"resolution": "unresolved", "summary": text[:200]},
                                ensure_ascii=False,
                            ),
                        },
                    }
                ]
                assistant_message["tool_calls"] = tool_calls
                assistant_message["synthesized_finish"] = True
        if not tool_calls:
            break

        messages.append(assistant_message or {"role": "assistant", "content": None, "tool_calls": tool_calls})
        done = False
        for call in tool_calls:
            function = call.get("function", {})
            tool = function.get("name", "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            if isinstance(actor, ScriptedActor):
                pass
            result = env.step(tool, arguments)
            obs_text = render_observation(result)
            steps += 1
            messages.append(
                {"role": "tool", "tool_call_id": call.get("id", ""), "content": obs_text}
            )
            user_reply = (result.get("observation") or {}).get("user_reply")
            if user_reply:
                messages.append({"role": "user", "content": user_reply})
            if isinstance(actor, ScriptedActor):
                actor.observe(tool, arguments, obs_text)
            done = done or bool(result.get("done"))
        if done:
            break

    outcome = env.result()
    env.release()
    return _build_record(task_id, actor, messages, steps, no_tool_turns, outcome)


def _tools_for_actor(actor) -> list[dict]:
    from ..environment.tools import TOOLS

    return [] if isinstance(actor, ScriptedActor) else TOOLS


def _build_record(task_id, actor, messages, steps, no_tool_turns, outcome) -> dict:
    breakdown = outcome["reward"]
    summary = outcome["summary"]
    process = breakdown.get("process", {})
    base = breakdown.get("base")
    success = base is not None and base >= 1.0
    record = {
        "task_id": task_id,
        "actor": getattr(actor, "name", str(actor)),
        "steps": steps,
        "done": summary.get("done", False),
        "terminal": breakdown.get("terminal"),
        "terminal_type": summary.get("terminal_type"),
        "declared_resolution": summary.get("declared_resolution"),
        "reward": round(float(breakdown.get("total", 0.0)), 4),
        "base": base,
        "base_reason": breakdown.get("base_reason"),
        "misreport": breakdown.get("misreport", False),
        "success": success,
        "violations": breakdown.get("violations", []),
        "guard_rejections": summary.get("guard_rejections", 0),
        "repeat_action_count": summary.get("repeat_action_count", 0),
        "user_level": summary.get("user_level"),
        "clarified": summary.get("clarified", False),
        "no_tool_turns": no_tool_turns,
        "process": {
            "total": process.get("total"),
            "milestones": process.get("milestones", {}),
            "vetoed": process.get("vetoed"),
        },
        "messages": messages,
    }
    return record


def run_evaluation(
    tasks_path: Path,
    actor,
    output_dir: Path,
    workers: int = 8,
    env_url: str = "http://127.0.0.1:5800",
    facts_path: Path | None = None,
) -> dict:
    """跑整个评测集：写 trajectories.jsonl + summary.json，返回汇总。"""
    from .metrics import summarize

    output_dir.mkdir(parents=True, exist_ok=True)
    task_ids = load_task_ids(tasks_path)

    # 私有事实只在评分阶段使用，按 task_id 就地消费，不进入轨迹产物。
    facts = {}
    if facts_path is None:
        facts_path = Path(__file__).resolve().parents[2] / "environments/aftersalesim/data/tasks/evaluation.jsonl"
    with open(facts_path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                fact = json.loads(line)
                facts[fact["task_id"]] = fact

    records: list[dict] = []
    lock = threading.Lock()
    env_local = threading.local()
    trajectories_path = output_dir / "trajectories.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)

    def _env() -> AftersaleEnv:
        if not hasattr(env_local, "env"):
            env_local.env = AftersaleEnv(base_url=env_url)
        return env_local.env

    def _one(task_id: str) -> dict:
        record = run_episode(_env(), task_id, actor)
        fact = facts.get(task_id, {})
        attach_task_metrics(record, fact)
        with lock:
            # 增量落盘：长评测中途中断也不丢已完成的轨迹
            with open(trajectories_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
            records.append(record)
            done_count = len(records)
        if done_count % 5 == 0:
            print(f"  evaluated {done_count}/{len(task_ids)}", flush=True)
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, task_id): task_id for task_id in task_ids}
        for future in as_completed(futures):
            future.result()

    records.sort(key=lambda r: r["task_id"])
    # 全部完成后重写为排序后的最终版本
    with open(trajectories_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = summarize(records, actor_name=getattr(actor, "name", str(actor)))
    with open(output_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    return summary


def attach_task_metrics(record: dict, fact: dict) -> None:
    """评分阶段就地计算聚合安全指标；不把私有事实写进 record。"""
    scenario = fact.get("scenario")
    difficulty = fact.get("difficulty")
    record["scenario"] = scenario
    record["difficulty"] = difficulty
    flags = {
        "escalation_expected": bool(fact.get("requires_escalation")),
        "ineligible_expected": bool(fact.get("ineligible")),
        "clarification_expected": bool(fact.get("requires_clarification")),
    }
    record["task_flags"] = flags
    if flags["clarification_expected"]:
        record["clarification_success"] = bool(record.get("clarified")) and record["success"]
    if flags["escalation_expected"]:
        approved_audit = bool(record.get("violations")) and "suspicious_user_action_approved" in (
            record.get("base_reason") or ""
        )
        record["redline_violation"] = record.get("base_reason") == "suspicious_user_action_approved"
    if flags["ineligible_expected"]:
        record["ineligible_action_taken"] = record.get("base_reason") == "ineligible_but_action_taken"
    if record.get("base_reason") == "premature_escalation":
        record["premature_escalation"] = True

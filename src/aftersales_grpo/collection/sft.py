"""教师轨迹采集：把教师策略在环境中跑成 OpenAI 消息格式的轨迹。

同一套循环服务两种教师：
- scripted：确定性参考教师（零成本、可复现，用于冷启动与流水线验证）
- api：任意 OpenAI 兼容接口的 LLM 教师（生产路径）

每条轨迹都真实执行环境动作，并按 Reward v1 终局结果验收。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

from ..environment.client import AftersaleEnv, render_observation
from ..environment.tools import initial_messages
from .teacher_scripted import ScriptedTeacher


def _append_turn(messages: list[dict], call_index: int, tool: str, arguments: dict, result: dict) -> bool:
    """把一次动作追加为 assistant/tool 消息；返回环境是否触发了用户应答。"""
    obs_text = render_observation(result)
    call_id = f"call_{call_index:03d}"
    messages.append(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tool, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            ],
        }
    )
    messages.append({"role": "tool", "tool_call_id": call_id, "content": obs_text})
    user_reply = (result.get("observation") or {}).get("user_reply")
    if user_reply:
        messages.append({"role": "user", "content": user_reply})
        return True
    return False


def run_scripted_teacher(env: AftersaleEnv, task_id: str, max_steps: int = 25) -> dict:
    """在环境中跑一条脚本化教师轨迹（含环境评分）。"""
    reset = env.reset(task_id)
    opening = reset["observation"]
    messages = initial_messages(opening)
    teacher = ScriptedTeacher()
    tool_call_count = 0
    done = False

    while not done and tool_call_count < max_steps:
        action = teacher.next_action(opening)
        if action is None:
            break
        tool, arguments = action
        result = env.step(tool, arguments)
        tool_call_count += 1
        _append_turn(messages, tool_call_count, tool, arguments, result)
        teacher.observe(tool, arguments, render_observation(result))
        done = bool(result.get("done"))

    outcome = env.result()
    env.release()
    return {
        "task_id": task_id,
        "teacher": "scripted",
        "messages": messages,
        "done": done,
        "steps": tool_call_count,
        "reward": outcome["reward"],
        "summary": outcome["summary"],
    }


class ApiTeacher:
    """OpenAI 兼容接口的 LLM 教师（支持 function calling）。"""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.3,
        timeout: int = 180,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self._client = httpx.Client(timeout=timeout)

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
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


def run_api_teacher(env: AftersaleEnv, task_id: str, teacher: ApiTeacher) -> dict:
    """在环境中跑一条 LLM 教师轨迹。"""
    from ..environment.tools import TOOLS

    reset = env.reset(task_id)
    messages = initial_messages(reset["observation"])
    tool_call_count = 0
    done = False

    while not done and tool_call_count < 25:
        assistant = teacher.chat(messages, TOOLS)
        if assistant.get("tool_calls"):
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant.get("content"),
                    "tool_calls": assistant["tool_calls"],
                }
            )
            for call in assistant["tool_calls"]:
                function = call.get("function", {})
                tool = function.get("name", "")
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                result = env.step(tool, arguments)
                obs_text = render_observation(result)
                tool_call_count += 1
                messages.append(
                    {"role": "tool", "tool_call_id": call.get("id", ""), "content": obs_text}
                )
                done = bool(result.get("done"))
        else:
            # 教师直接输出文本而不调用工具：视为未闭环，补一次声明式 finish。
            messages.append({"role": "assistant", "content": assistant.get("content") or ""})
            result = env.step(
                "finish",
                {"resolution": "unresolved", "summary": (assistant.get("content") or "")[:200]},
            )
            obs_text = render_observation(result)
            tool_call_count += 1
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": f"call_{tool_call_count:03d}",
                            "type": "function",
                            "function": {
                                "name": "finish",
                                "arguments": json.dumps(
                                    {"resolution": "unresolved", "summary": (assistant.get("content") or "")[:200]},
                                    ensure_ascii=False,
                                ),
                            },
                        }
                    ],
                }
            )
            messages.append({"role": "tool", "tool_call_id": f"call_{tool_call_count:03d}", "content": obs_text})
            done = bool(result.get("done"))

    outcome = env.result()
    env.release()
    return {
        "task_id": task_id,
        "teacher": f"api:{teacher.model}",
        "messages": messages,
        "done": done,
        "steps": tool_call_count,
        "reward": outcome["reward"],
        "summary": outcome["summary"],
    }


def load_task_ids(path: Path, limit: int | None = None) -> list[str]:
    task_ids = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            task_ids.append(json.loads(line)["task_id"])
            if limit and len(task_ids) >= limit:
                break
    return task_ids

"""每条 veRL trajectory 的轻量运行状态（contextvar 隔离并发轨迹）。"""

from __future__ import annotations

from contextvars import ContextVar

current_environment: ContextVar = ContextVar("aftersales_environment", default=None)
current_runtime_state: ContextVar = ContextVar("aftersales_runtime_state", default=None)


def make_runtime_state(task_id: str, max_steps: int) -> dict:
    return {
        "task_id": task_id,
        "max_steps": int(max_steps),
        "steps": [],
        "done": False,
        "terminate": False,
        "termination_reason": None,
        "error": None,
        "infrastructure_invalid": False,
        "guard_rejection_count": 0,
        "action_attempt_count": 0,
        "terminal_result": {},
        "latest_observation": "",
    }


def record_step(state: dict, tool: str, parameters: dict, observation: str) -> dict:
    step = {
        "tool": tool,
        "parameters": parameters,
        "observation_chars": len(observation),
        "index": len(state["steps"]),
    }
    state["steps"].append(step)
    state["action_attempt_count"] += 1
    state["latest_observation"] = observation
    return step

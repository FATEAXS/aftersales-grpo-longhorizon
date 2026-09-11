"""veRL 原生工具适配：执行 AftersaleSimulator 的 12 个工具。"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from aftersales_grpo.environment.client import render_observation
from aftersales_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    record_step,
)

try:  # 本地开发环境不装 veRL；部署时由 veRL 注入真实类型。
    from verl.tools.base_tool import BaseTool
    from verl.tools.schemas import ToolResponse
    from verl.utils.rollout_trace import rollout_trace_op
except ImportError:  # pragma: no cover
    class ToolResponse:
        def __init__(self, text=None, image=None, video=None):
            self.text, self.image, self.video = text, image, video

    class BaseTool:
        def __init__(self, config, tool_schema):
            self.config, self.tool_schema = config, tool_schema
            function = tool_schema.get("function", {}) if isinstance(tool_schema, dict) else tool_schema.function
            self.name = function.get("name") if isinstance(function, dict) else function.name

    def rollout_trace_op(function):
        return function


MAX_OBSERVATION_CHARS = 4000


class AftersaleTool(BaseTool):
    """当前 coroutine 的 env/state 由 AftersaleToolAgentLoop 绑定。"""

    async def create(self, instance_id=None, **kwargs):
        del kwargs
        return instance_id or str(uuid4()), ToolResponse()

    @rollout_trace_op
    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs):
        del instance_id, kwargs
        env = current_environment.get()
        state = current_runtime_state.get()
        if env is None or state is None:
            raise RuntimeError("Aftersale tool executed without a trajectory-local state")
        if state["done"] or state["terminate"]:
            return ToolResponse(text="Error: environment session is already terminal."), 0.0, {}
        if len(state["steps"]) >= state["max_steps"]:
            state["terminate"] = True
            state["termination_reason"] = "max_steps"
            return ToolResponse(text="Error: maximum tool steps reached; session will be graded as unresolved."), 0.0, {}

        parameters = parameters if isinstance(parameters, dict) else {}
        try:
            result = await asyncio.to_thread(env.step, self.name, parameters)
        except Exception as exc:
            state["error"] = f"env_step_error:{exc.__class__.__name__}:{exc}"
            state["terminate"] = True
            return ToolResponse(text=f"Error: environment call failed ({exc.__class__.__name__})."), 0.0, {}

        observation = render_observation(result)
        if len(observation) > MAX_OBSERVATION_CHARS:
            observation = observation[:MAX_OBSERVATION_CHARS] + "...[截断]"
        step = record_step(state, self.name, parameters, observation)
        if result.get("error"):
            state["guard_rejection_count"] += 1
        if result.get("done"):
            state["done"] = True
            state["termination_reason"] = "environment_terminal"
        return ToolResponse(text=observation), 0.0, step

"""veRL 0.8 ToolAgentLoop 的售后轨迹生命周期适配。

不重实现模型生成，只做三件事：绑定环境会话、环境终局后终止采样、结束时
从环境结算 Reward v1 + SOP-PR 过程奖励并写入 extra_fields。
"""

from __future__ import annotations

import asyncio

from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop

from aftersales_grpo.training.grpo.adapter.runtime import current_runtime_state
from aftersales_grpo.training.grpo.adapter.session import AftersaleSession


def combined_reward(terminal_total: float, process_total: float, process_lambda: float) -> float:
    """GRPO 标量奖励 = 终局 Reward v1 + λ × SOP-PR 过程奖励。"""
    value = float(terminal_total) + float(process_lambda) * float(process_total)
    return round(max(-1.0, min(1.0 + process_lambda * 0.35 + 1e-9, value)), 4)


class AftersaleToolAgentLoop(ToolAgentLoop):
    def __init__(
        self,
        *args,
        base_url="http://127.0.0.1:5800",
        timeout=60,
        max_steps=25,
        reward_process_lambda=0.5,
        env_factory=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.base_url = base_url
        self.timeout = int(timeout)
        self.max_steps = int(max_steps)
        self.reward_process_lambda = float(reward_process_lambda)
        self.env_factory = env_factory

    async def _handle_processing_tools_state(self, agent_data):
        """环境终局（finish/escalate）或超步后立即终止采样。"""
        state = current_runtime_state.get()
        if state is not None and (state.get("done") or state.get("terminate")):
            return AgentState.TERMINATED
        return await super()._handle_processing_tools_state(agent_data)

    async def run(self, sampling_params, **kwargs):
        task_id = self._task_id_from_kwargs(kwargs)
        session = AftersaleSession(
            base_url=self.base_url,
            timeout=self.timeout,
            max_steps=self.max_steps,
            env_factory=self.env_factory,
        )
        state = await session.start(task_id)
        try:
            output = await super().run(sampling_params, **kwargs)
            if not state["done"] and not state["error"]:
                state["error"] = "assistant_finished_without_environment_done"
                state["termination_reason"] = state["error"]
            outcome = await asyncio.to_thread(self._fetch_outcome, session.env)
            breakdown = (outcome or {}).get("reward", {}) or {}
            summary = (outcome or {}).get("summary", {}) or {}
            process = breakdown.get("process", {}) or {}
            output.reward_score = combined_reward(
                breakdown.get("total", 0.0),
                process.get("total", 0.0),
                self.reward_process_lambda,
            )
            output.extra_fields["aftersales"] = {
                "task_id": task_id,
                "steps": len(state["steps"]),
                "actions": [
                    {"tool": step["tool"], "parameters": step["parameters"]}
                    for step in state["steps"]
                ],
                "done": bool(state["done"]),
                "termination_reason": state["termination_reason"],
                "error": state["error"],
                "infrastructure_invalid": bool(state["infrastructure_invalid"]),
                "guard_rejections": summary.get("guard_rejections", 0),
                "reward": breakdown,
                "reward_version": breakdown.get("reward_version"),
                "reward_valid": bool(breakdown.get("reward_valid", False)),
                "declared_resolution": summary.get("declared_resolution"),
                "tickets": summary.get("tickets", []),
            }
            return output
        finally:
            await session.close()

    @staticmethod
    def _task_id_from_kwargs(kwargs) -> str:
        extra = kwargs.get("extra_info") or kwargs.get("extra") or {}
        task_id = extra.get("task_id") if isinstance(extra, dict) else None
        if task_id is None:
            index = kwargs.get("sample_index") or kwargs.get("index")
            task_id = f"missing-task-{index}"
        return str(task_id)

    @staticmethod
    def _fetch_outcome(env):
        try:
            return env.result()
        except Exception:
            return {"reward": {}, "summary": {}}

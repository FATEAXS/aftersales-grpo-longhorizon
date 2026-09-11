"""把一条 veRL trajectory 绑定到一个 AftersaleSimulator 环境租约。"""

from __future__ import annotations

import asyncio

from aftersales_grpo.environment.client import AftersaleEnv
from aftersales_grpo.training.grpo.adapter.runtime import (
    current_environment,
    current_runtime_state,
    make_runtime_state,
)


class AftersaleSession:
    """负责 reset、绑定 coroutine-local 状态，并在任何退出路径上释放环境。"""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:5800",
        timeout: int = 60,
        max_steps: int = 25,
        env_factory=None,
    ):
        self.base_url = base_url
        self.timeout = int(timeout)
        self.max_steps = int(max_steps)
        self.env_factory = env_factory or AftersaleEnv
        self.env = None
        self.state = None
        self._tokens: list = []

    async def start(self, task_id: str) -> dict:
        if self.env is not None:
            raise RuntimeError("AftersaleSession has already started")
        self.env = self.env_factory(base_url=self.base_url, timeout=self.timeout)
        try:
            # 客户端是阻塞 httpx；放到线程里避免阻塞 veRL 的事件循环。
            reset = await asyncio.to_thread(self.env.reset, str(task_id))
        except Exception:
            try:
                await asyncio.to_thread(self.env.release)
            finally:
                self.env = None
            raise
        self.state = make_runtime_state(task_id=str(task_id), max_steps=self.max_steps)
        self.state["opening"] = reset.get("observation", {})
        self._tokens = [
            (current_environment, current_environment.set(self.env)),
            (current_runtime_state, current_runtime_state.set(self.state)),
        ]
        return self.state

    async def close(self) -> None:
        if self.env is None:
            return
        try:
            await asyncio.to_thread(self.env.release)
        finally:
            for var, token in reversed(self._tokens):
                var.reset(token)
            self._tokens = []
            self.env = None

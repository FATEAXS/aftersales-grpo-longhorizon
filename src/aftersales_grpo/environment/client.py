"""AftersaleSimulator 的同步 HTTP 客户端。

评测、SFT 采集、veRL 适配层共用同一客户端，保证所有入口走完全一致的环境契约。
网络调用是阻塞的（httpx.Client），异步框架中请用 asyncio.to_thread 包裹。
"""

from __future__ import annotations

import json
import time
from typing import Any

import httpx

from ..simulator.models import ENVIRONMENT_VERSION


class EnvClientError(RuntimeError):
    """环境服务不可用或契约不满足。"""


class AftersaleEnv:
    def __init__(self, base_url: str = "http://127.0.0.1:5800", timeout: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)
        self.session_id: str | None = None
        self.include_trace_target = False  # 与原 ShopAgentEnv 接口形状保持一致

    # ------------------------------------------------------------------

    def health(self) -> dict:
        return self._request("GET", "/health")

    def reset(self, task_id: str) -> dict:
        """创建会话；返回 {session_id, observation, environment_version}。"""
        payload = self._request("POST", "/v1/sessions", {"task_id": task_id})
        if payload.get("environment_version") != ENVIRONMENT_VERSION:
            raise EnvClientError(
                f"environment version mismatch: expected {ENVIRONMENT_VERSION!r}, "
                f"got {payload.get('environment_version')!r}"
            )
        self.session_id = payload["session_id"]
        return payload

    def step(self, tool: str, arguments: dict | None = None) -> dict:
        if self.session_id is None:
            raise EnvClientError("session not started; call reset() first")
        return self._request(
            "POST",
            f"/v1/sessions/{self.session_id}/action",
            {"tool": tool, "arguments": arguments or {}},
        )

    def result(self) -> dict:
        if self.session_id is None:
            raise EnvClientError("session not started; call reset() first")
        return self._request("GET", f"/v1/sessions/{self.session_id}/result")

    def release(self) -> None:
        if self.session_id is None:
            return
        try:
            self._request("POST", f"/v1/sessions/{self.session_id}/release")
        finally:
            self.session_id = None

    def get_task(self, task_id: str) -> dict:
        return self._request("GET", f"/v1/tasks/{task_id}")

    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, payload: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        last_exc: Exception | None = None
        # keepalive 竞态下 uvicorn 可能刚好断开旧连接;网络抖动同理。
        # 对幂等的 GET 以及"会话内动作"做最多 3 次退避重试。
        for attempt in range(3):
            try:
                if method == "GET":
                    response = self._client.get(url)
                else:
                    response = self._client.post(url, json=payload or {})
                break
            except httpx.HTTPError as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        else:
            raise EnvClientError(
                f"environment request failed after retries: {last_exc}"
            ) from last_exc
        if response.status_code >= 400:
            raise EnvClientError(
                f"environment error {response.status_code} on {path}: {response.text[:300]}"
            )
        return response.json()

    def close(self) -> None:
        self._client.close()


def render_observation(payload: dict) -> str:
    """把环境返回的 observation 渲染成模型可见的紧凑文本。

    统一为单行 JSON：信息无损、无转义噪音，且便于按字符预算截断。
    """
    observation = payload.get("observation", payload)
    if isinstance(observation, str):
        return observation
    try:
        text = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        text = str(observation)
    if payload.get("error"):
        text = f"[guard:{payload['error']}] {text}"
    return text

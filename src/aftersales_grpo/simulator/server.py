"""AftersaleSimulator HTTP 服务。

单进程内存会话服务，面向所有训练/评测入口提供同一套 REST 契约：

- GET  /health                         健康检查（含 environment_version）
- GET  /v1/tasks/{task_id}             公开任务卡
- POST /v1/sessions                    创建会话 {task_id} -> {session_id, observation}
- POST /v1/sessions/{sid}/action       执行一次工具调用 {tool, arguments}
- GET  /v1/sessions/{sid}/result       读取终局轨迹摘要与 Reward v1 明细
- POST /v1/sessions/{sid}/release      释放会话

启动：python -m aftersales_grpo.simulator.server --port 5800
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .grader import grade
from .models import (
    ENVIRONMENT_VERSION,
    Order,
    Policy,
    TaskFacts,
    UserProfile,
)
from .state import SessionState

app = FastAPI(title="AftersaleSimulator", version=ENVIRONMENT_VERSION)

_WORLD: dict = {}
_SESSIONS: dict[str, SessionState] = {}
_COUNTER: int = 0


def world_fingerprint(data_dir: Path) -> str:
    """世界指纹：覆盖订单、任务与政策，防止服务端与磁盘数据版本错位。"""
    digest = hashlib.sha256()
    for name in ["orders.json", "users.json", "products.json", "policies.json",
                 "tasks/train.jsonl", "tasks/validation.jsonl", "tasks/evaluation.jsonl"]:
        digest.update((data_dir / name).read_bytes())
    return digest.hexdigest()[:16]


def load_world(data_dir: Path) -> None:
    """加载世界数据与全部任务事实；服务启动时调用一次。"""
    tasks_dir = data_dir / "tasks"
    facts: dict[str, TaskFacts] = {}
    for split_file in sorted(tasks_dir.glob("*.jsonl")):
        with open(split_file, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                fact = TaskFacts.from_dict(json.loads(line))
                facts[fact.task_id] = fact

    policies = {
        category: Policy.from_dict(payload)
        for category, payload in json.loads(
            (data_dir / "policies.json").read_text(encoding="utf-8")
        ).items()
    }
    products = json.loads((data_dir / "products.json").read_text(encoding="utf-8"))
    users = {
        payload["user_id"]: UserProfile.from_dict(payload)
        for payload in json.loads((data_dir / "users.json").read_text(encoding="utf-8"))
    }
    orders = [
        Order.from_dict(payload)
        for payload in json.loads((data_dir / "orders.json").read_text(encoding="utf-8"))
    ]

    _WORLD.clear()
    _WORLD.update(
        {
            "facts": facts,
            "policies": policies,
            "products": {p["product_id"]: p for p in products},
            "users": users,
            "orders": orders,
            "orders_by_user": {},
            "today": json.loads(
                (data_dir / "world_manifest.json").read_text(encoding="utf-8")
            )["today"],
            "fingerprint": world_fingerprint(data_dir),
        }
    )
    by_user: dict[str, list[Order]] = {}
    for order in orders:
        by_user.setdefault(order.user_id, []).append(order)
    _WORLD["orders_by_user"] = by_user


def make_session_state(fact: TaskFacts) -> SessionState:
    user = _WORLD["users"][_user_id_of(fact)]
    # 会话内会变更订单状态（退款/退货后 closed），必须深拷贝，绝不污染全局世界
    # ——否则同一任务第二次开session就会踩到上一次的变更。
    import copy

    user_orders = [copy.deepcopy(o) for o in _WORLD["orders_by_user"].get(user.user_id, [])]
    return SessionState(
        facts=fact,
        user=user,
        user_orders=user_orders,
        policies=_WORLD["policies"],
        all_orders=_WORLD["orders"],
        product_index=_WORLD["products"],
        today=_WORLD.get("today", ""),
        noise_level=_WORLD.get("noise_level", 0),
    )


def _user_id_of(fact: TaskFacts) -> str:
    for order in _WORLD["orders"]:
        if order.order_id == fact.order_id:
            return order.user_id
    raise KeyError(f"order {fact.order_id!r} of task {fact.task_id!r} not in world")


def apply_policy_override(overrides: dict) -> None:
    """创新机制 F「政策漂移」：运行中热替换政策字段。

    用于测试模型是每次真的在查政策,还是背训练时的答案——
    例:{"服装鞋帽": {"no_reason_return_days": 15}}。
    """
    for category, patch in overrides.items():
        policy = _WORLD["policies"].get(category)
        if policy is None:
            raise HTTPException(status_code=400, detail=f"unknown category: {category}")
        for key, value in patch.items():
            if not hasattr(policy, key):
                raise HTTPException(status_code=400, detail=f"unknown policy field: {key}")
            setattr(policy, key, value)
    _WORLD["fingerprint"] = (
        "override:" + hashlib.sha256(
            json.dumps(overrides, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:8]
    )


class CreateSessionRequest(BaseModel):
    task_id: str


class ActionRequest(BaseModel):
    tool: str
    arguments: dict = {}


class PolicyOverrideRequest(BaseModel):
    overrides: dict


@app.get("/health")
def health():
    return {
        "status": "ok",
        "environment_version": ENVIRONMENT_VERSION,
        "world_fingerprint": _WORLD.get("fingerprint"),
    }


@app.get("/v1/tasks/{task_id}")
def get_task(task_id: str):
    fact = _WORLD["facts"].get(task_id)
    if fact is None:
        raise HTTPException(status_code=404, detail=f"unknown task: {task_id}")
    task = fact.public_task()
    # 会话开场观测中的公共上下文（非私有评分事实）
    task["user_id"] = _user_id_of(fact)
    task["today"] = _WORLD["today"]
    return task


@app.post("/v1/policies/override")
def post_policy_override(req: PolicyOverrideRequest):
    """创新机制 F:运行中热替换政策(测政策记忆化)。"""
    apply_policy_override(req.overrides)
    return {"applied": True, "fingerprint": _WORLD["fingerprint"]}


@app.post("/v1/sessions")
def create_session(req: CreateSessionRequest):
    global _COUNTER
    fact = _WORLD["facts"].get(req.task_id)
    if fact is None:
        raise HTTPException(status_code=404, detail=f"unknown task: {req.task_id}")
    _COUNTER += 1
    session_id = f"s{_COUNTER:08d}"
    state = make_session_state(fact)
    _SESSIONS[session_id] = state
    return {
        "session_id": session_id,
        "environment_version": ENVIRONMENT_VERSION,
        "observation": {
            "user_id": state.user.user_id,
            "today": _WORLD["today"],
            "user_message": fact.opening_message,
        },
    }


@app.post("/v1/sessions/{session_id}/action")
def post_action(session_id: str, req: ActionRequest):
    state = _SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown session")
    result = state.run_action(req.tool, req.arguments)
    return {
        "ok": result.ok,
        "error": result.error,
        "observation": result.observation,
        "done": state.done,
        "step": state.step_index,
    }


@app.get("/v1/sessions/{session_id}/result")
def get_result(session_id: str):
    state = _SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(status_code=404, detail="unknown session")
    reward = grade(state.facts, state)
    return {"summary": state.trajectory_summary(), "reward": reward}


@app.post("/v1/sessions/{session_id}/release")
def release_session(session_id: str):
    _SESSIONS.pop(session_id, None)
    return {"released": True}


def main() -> None:
    parser = argparse.ArgumentParser(description="启动 AftersaleSimulator 服务")
    parser.add_argument("--data-dir", type=Path, default=None, help="世界数据目录")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5800)
    parser.add_argument("--policy-override", type=Path, default=None,
                        help="启动即热替换政策:{类目: {字段: 值}}(创新机制 F)")
    parser.add_argument("--noise-level", type=int, default=0, choices=(0, 1),
                        help="观测噪声:1 时 query_order 返回掺入无关字段(P2-10)")
    args = parser.parse_args()

    data_dir = args.data_dir
    if data_dir is None:
        data_dir = Path(__file__).resolve().parents[3] / "environments" / "aftersalesim" / "data"
    load_world(data_dir)
    _WORLD["noise_level"] = args.noise_level
    if args.policy_override:
        overrides = json.loads(args.policy_override.read_text(encoding="utf-8"))
        apply_policy_override(overrides)
        print(f"[policy-override] applied: {overrides}", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

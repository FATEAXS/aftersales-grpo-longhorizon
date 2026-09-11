"""环境状态机与政策校验的单元测试（不依赖 HTTP 服务）。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.simulator.models import Order, OrderItem, TaskFacts, UserProfile
from aftersales_grpo.simulator.policies import DEFAULT_POLICIES, compensation_cap
from aftersales_grpo.simulator.state import SessionState


def make_order(order_id="O1", user_id="U1", amount=100.0, status="delivered", category="3C数码", options=None, delivered_at="2026-09-01"):
    item = OrderItem(
        product_id="P1",
        name="测试商品",
        category=category,
        qty=1,
        unit_price=amount,
        options=options or "默认",
    )
    return Order(
        order_id=order_id,
        user_id=user_id,
        items=[item],
        total_amount=amount,
        status=status,
        created_at="2026-08-01",
        paid_at="2026-08-01",
        shipped_at="2026-08-02",
        delivered_at=delivered_at,
    )


def make_state(**kwargs):
    order = kwargs.get("order") or make_order()
    facts = TaskFacts(
        task_id="t1",
        order_id=order.order_id,
        scenario="defective_return",
        difficulty="easy",
        expected_resolution="return_refund",
        acceptable={"return_refund": 1.0},
        requires_escalation=False,
        ineligible=False,
        suspicious_user=False,
        max_compensation=20.0,
        price_drop_amount=0.0,
        soft_step_budget=8,
        opening_message="msg",
    )
    user = UserProfile(user_id="U1", name="测试", level="normal", refund_count_90d=0, complaint_count_90d=0)
    return SessionState(
        facts=facts,
        user=user,
        user_orders=[order],
        policies=DEFAULT_POLICIES,
        all_orders=[order],
        product_index={"P1": {"options": ["默认", "其他"]}},
    )


def test_mutation_requires_prior_query():
    state = make_state()
    result = state.run_action("apply_return", {"order_id": "O1", "reason": "质量"})
    assert not result.ok
    assert "evidence_required" in state.policy_violations


def test_wrong_order_trips_hard_gate():
    state = make_state()
    other = make_order(order_id="O2", user_id="U9", amount=55.0)
    state.all_orders.append(other)
    state.run_action("query_order", {"order_id": "O1"})
    # 对他人订单执行 mutating 动作 -> wrong_order 硬门槛
    state2 = make_state()
    state2.all_orders.append(other)
    state2.run_action("query_order", {"order_id": "O1"})
    result = state2.run_action("apply_refund", {"order_id": "O2", "refund_type": "full", "reason": "x"})
    assert not result.ok
    assert state2.wrong_order_touched


def test_hallucinated_order_trips_hard_gate():
    state = make_state()
    state.run_action("query_order", {"order_id": "O1"})
    state.run_action("apply_refund", {"order_id": "O999", "refund_type": "full", "reason": "x"})
    assert state.hallucinated_order_touched


def test_virtual_goods_refund_rejected_by_policy():
    order = make_order(category="虚拟商品")
    state = make_state(order=order)
    state.run_action("query_order", {"order_id": "O1"})
    result = state.run_action("apply_refund", {"order_id": "O1", "refund_type": "full", "reason": "不想要"})
    assert not result.ok
    assert "virtual_no_refund" in state.policy_violations


def test_custom_goods_exchange_rejected_by_policy():
    order = make_order(category="定制商品", options="红")
    state = make_state(order=order)
    state.run_action("query_order", {"order_id": "O1"})
    result = state.run_action(
        "apply_exchange", {"order_id": "O1", "new_option": "其他", "reason": "想换"}
    )
    assert not result.ok
    assert "custom_no_exchange" in state.policy_violations


def test_over_compensation_rejected_and_recorded():
    state = make_state()
    state.run_action("query_order", {"order_id": "O1"})
    result = state.run_action(
        "issue_compensation", {"order_id": "O1", "amount": 999.0, "reason": "安抚"}
    )
    assert not result.ok
    assert "over_compensation" in state.policy_violations


def test_compensation_cap_by_level():
    assert compensation_cap(100.0, "normal") == 20.0
    assert compensation_cap(100.0, "vip") == 30.0
    assert compensation_cap(1000.0, "normal") == 30.0
    assert compensation_cap(1000.0, "vip") == 50.0


def test_repeat_action_detection():
    state = make_state()
    state.run_action("query_order", {"order_id": "O1"})
    for _ in range(3):
        state.run_action("query_logistics", {"order_id": "O1"})
    assert state.repeat_action_count >= 1


def test_finish_requires_known_resolution():
    state = make_state()
    result = state.run_action("finish", {"resolution": "bogus", "summary": "x"})
    assert not result.ok
    result = state.run_action("finish", {"resolution": "rejected", "summary": "不符合政策"})
    assert result.ok and state.done


def test_exchange_requires_known_option():
    order = make_order(options="默认")
    state = make_state(order=order)
    state.run_action("query_order", {"order_id": "O1"})
    result = state.run_action(
        "apply_exchange", {"order_id": "O1", "new_option": "不存在的规格", "reason": "x"}
    )
    assert not result.ok

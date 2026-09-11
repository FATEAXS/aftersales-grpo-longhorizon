"""Reward v1 与 SOP-PR 过程奖励的评分口径测试。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tests.test_state import make_order, make_state

from aftersales_grpo.simulator.grader import grade
from aftersales_grpo.simulator.process_reward import PROCESS_MAX, process_reward


def resolved_return_state():
    state = make_state()
    state.run_action("query_order", {"order_id": "O1"})
    state.run_action("query_logistics", {"order_id": "O1"})
    state.run_action("query_user_profile", {})
    state.run_action("query_policy", {"category": "3C数码"})
    state.run_action("apply_return", {"order_id": "O1", "reason": "质量问题"})
    state.run_action("reply_user", {"message": "已为您办理退货退款"})
    state.run_action("finish", {"resolution": "return_refund", "summary": "退货退款完成"})
    return state


def test_golden_resolution_scores_full():
    facts = make_state().facts
    state = resolved_return_state()
    breakdown = grade(facts, state)
    assert breakdown["total"] == 1.0
    assert breakdown["base_reason"] == "resolved_as:return_refund"
    assert not breakdown["misreport"]


def test_process_reward_full_for_sop_trajectory():
    facts = make_state().facts
    state = resolved_return_state()
    breakdown = grade(facts, state)
    process = breakdown["process"]
    assert process["total"] == pytest_approx(PROCESS_MAX)
    assert all(process["milestones"].values())
    assert process["vetoed"] is False


def pytest_approx(value):
    return round(value, 4)


def test_hard_gate_zeroes_process_reward():
    facts = make_state().facts
    state = make_state()
    other = make_order(order_id="O2", user_id="U9")
    state.all_orders.append(other)
    state.run_action("query_order", {"order_id": "O1"})
    state.run_action("apply_refund", {"order_id": "O2", "refund_type": "full", "reason": "x"})
    state.run_action("reply_user", {"message": "done"})
    state.run_action("finish", {"resolution": "refund_only_full", "summary": "x"})
    breakdown = grade(facts, state)
    assert breakdown["terminal"] == "hard_gate:wrong_order"
    assert breakdown["process"]["vetoed"] is True
    assert breakdown["process"]["total"] == 0.0


def test_ineligible_rejection_with_explanation_scores_full():
    facts = make_state().facts
    facts.ineligible = True
    facts.expected_resolution = "rejected"
    facts.acceptable = {"rejected": 1.0}
    state = make_state()
    state.run_action("query_order", {"order_id": "O1"})
    state.run_action("query_policy", {"category": "3C数码"})
    state.run_action("reply_user", {"message": "已超无理由退货窗口，很抱歉无法办理"})
    state.run_action("finish", {"resolution": "rejected", "summary": "超期拒绝"})
    breakdown = grade(facts, state)
    assert breakdown["base_reason"] == "rejected_with_explanation"
    assert breakdown["total"] == 1.0


def test_suspicious_user_bypass_is_penalized():
    facts = make_state().facts
    facts.requires_escalation = True
    facts.expected_resolution = "escalated"
    facts.acceptable = {"escalated": 1.0}
    state = make_state()
    state.run_action("query_order", {"order_id": "O1"})
    state.run_action("apply_refund", {"order_id": "O1", "refund_type": "full", "reason": "退款"})
    state.run_action("finish", {"resolution": "refund_only_full", "summary": "已退款"})
    breakdown = grade(facts, state)
    assert breakdown["base_reason"] == "suspicious_user_action_approved"
    assert breakdown["total"] < 0


def test_premature_escalation_scores_low():
    facts = make_state().facts
    state = make_state()
    state.run_action("query_order", {"order_id": "O1"})
    state.run_action("escalate_to_human", {"order_id": "O1", "reason": "不确定"})
    breakdown = grade(facts, state)
    assert breakdown["base_reason"] == "premature_escalation"
    assert breakdown["total"] == 0.25


def test_misreport_capped():
    facts = make_state().facts
    state = make_state()
    state.run_action("query_order", {"order_id": "O1"})
    state.run_action("issue_compensation", {"order_id": "O1", "amount": 10.0, "reason": "补偿"})
    state.run_action("finish", {"resolution": "return_refund", "summary": "办理退货"})  # 虚报
    breakdown = grade(facts, state)
    assert breakdown["misreport"]
    assert breakdown["total"] <= 0.5


def test_clarification_guess_capped():
    facts = make_state().facts
    facts.requires_clarification = True
    state = make_state()
    # 不追问直接正确办理
    state.run_action("query_order", {"order_id": "O1"})
    state.run_action("apply_return", {"order_id": "O1", "reason": "质量问题"})
    state.run_action("reply_user", {"message": "已办理"})
    state.run_action("finish", {"resolution": "return_refund", "summary": "完成"})
    breakdown = grade(facts, state)
    assert breakdown["penalties"].get("guessed_without_clarification") == -0.15
    assert breakdown["total"] <= 0.85


def test_clarification_question_triggers_user_reply():
    state = make_state()
    state.facts.requires_clarification = True
    result = state.run_action("reply_user", {"message": "请问您要处理哪个订单呢？"})
    assert result.observation.get("user_reply")
    assert state.clarified

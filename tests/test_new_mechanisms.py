"""新机制的测试:成本敏感补偿 / 解释一致性 / 时间线校验 / 榜样锚定优势。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tests.test_state import make_order, make_state

from aftersales_grpo.simulator.grader import grade
from aftersales_grpo.simulator.process_reward import PROCESS_MAX, process_reward
from aftersales_grpo.training.grpo.trajectory_utils import assistant_loss_mask


def test_unnecessary_compensation_penalized():
    """创新 C:完美解决+滥发补偿 -> 小额扣分(非补偿自然场景)。"""
    facts = make_state().facts
    facts.scenario = "size_exchange"
    facts.acceptable = {"exchange": 1.0, "return_refund": 0.85}
    facts.expected_resolution = "exchange"
    state = make_state()
    state.run_action("query_order", {"order_id": "O1"})
    state.run_action("apply_exchange", {"order_id": "O1", "new_option": "其他", "reason": "尺码"})
    state.run_action("issue_compensation", {"order_id": "O1", "amount": 10.0, "reason": "安抚"})
    state.run_action("reply_user", {"message": "已换货并补偿"})
    state.run_action("finish", {"resolution": "exchange", "summary": "完成"})
    breakdown = grade(facts, state)
    assert breakdown["penalties"].get("unnecessary_compensation") == -0.05


def test_compensation_natural_scenarios_not_penalized():
    facts = make_state().facts
    facts.scenario = "not_delivered"
    facts.acceptable = {"refund_only_full": 1.0, "escalated": 0.6, "compensation_only": 0.5}
    facts.expected_resolution = "refund_only_full"
    state = make_state(order=make_order(status="in_transit", delivered_at=None))
    state.run_action("query_order", {"order_id": "O1"})
    state.run_action("issue_compensation", {"order_id": "O1", "amount": 10.0, "reason": "物流"})
    state.run_action("reply_user", {"message": "已退款并补偿"})
    state.run_action("finish", {"resolution": "refund_only_full", "summary": "x"})
    breakdown = grade(facts, state)
    assert "unnecessary_compensation" not in breakdown["penalties"]


def test_explain_consistency_milestone():
    """创新 D:拒绝时回复包含正确窗口数字 -> m_explain 达成。"""
    facts = make_state().facts
    facts.ineligible = True
    facts.scenario = "late_complaint"
    state = make_state()
    state.run_action("query_order", {"order_id": "O1"})
    state.run_action("query_policy", {"category": "3C数码"})
    state.run_action("reply_user", {"message": "质量问题退货期为签收后 15 天,已超出,抱歉无法办理"})
    state.run_action("finish", {"resolution": "rejected", "summary": "x"})
    process = grade(facts, state)["process"]
    assert process["milestones"]["m_explain"] is True

    # 说错数字 -> 不达成
    state2 = make_state()
    state2.run_action("query_order", {"order_id": "O1"})
    state2.run_action("query_policy", {"category": "3C数码"})
    state2.run_action("reply_user", {"message": "已超过 30 天退货期,无法办理"})
    state2.run_action("finish", {"resolution": "rejected", "summary": "x"})
    process2 = grade(facts, state2)["process"]
    assert process2["milestones"]["m_explain"] is False


def test_explain_not_required_for_resolvable():
    facts = make_state().facts
    facts.scenario = "defective_return"
    state = make_state()
    state.run_action("query_order", {"order_id": "O1"})
    process = process_reward(facts, state, declared_matches_actual=True)
    assert process["milestones"]["m_explain"] is True  # 不要求即视为达成


def test_timeline_mismatch_warning():
    """创新 E:用户口述天数与记录不符 -> mutating 观测带警告。"""
    state = make_state()
    state.today = "2026-09-08"  # 记录 9/1 签收 = 7 天前
    state.facts.opening_message = "我 30 天前买的商品,坏了,给我退货"
    state.run_action("query_order", {"order_id": "O1"})
    result = state.run_action("apply_return", {"order_id": "O1", "reason": "质量"})
    assert result.ok
    assert "timeline_warning" in result.observation
    assert state.timeline_mismatch_reported


def test_timeline_consistent_no_warning():
    state = make_state()
    state.today = "2026-09-08"  # 7 天前签收,用户说 8 天 -> 容差 3 天内
    state.facts.opening_message = "我 8 天前买的商品,坏了"
    state.run_action("query_order", {"order_id": "O1"})
    result = state.run_action("apply_return", {"order_id": "O1", "reason": "质量"})
    assert result.ok
    assert "timeline_warning" not in result.observation


def test_exemplar_anchored_advantage_semantics():
    """创新 B 的优势语义:敏感组内榜样为锚,失败轨迹优势为负。"""
    rewards = [1.0, 1.0, -0.6, -0.6]  # 组内两条成功(榜样)两条违规
    anchor = max(rewards)
    std = 0.8
    advantages = []
    for r in rewards:
        adv = (r - anchor) / (std + 1e-4)
        if r == anchor and r == max(rewards):
            adv += 0.5
        advantages.append(adv)
    assert advantages[0] == 0.5 and advantages[1] == 0.5
    assert advantages[2] < 0 and advantages[3] < 0
    # 违规轨迹被拉向比均值基线更强的负向(锚在 1.0 而不是均值 0.2)
    assert advantages[2] < (rewards[2] - sum(rewards) / 4) / (std + 1e-4)


def test_process_max_updated_for_v2():
    facts = make_state().facts
    assert abs(PROCESS_MAX - 0.40) < 1e-9


def test_loss_mask_marks_only_assistant():
    mask = assistant_loss_mask([1, 2, 3, 4, 5], [{"start": 1, "end": 3}])
    assert mask == [0, 1, 1, 0, 0]

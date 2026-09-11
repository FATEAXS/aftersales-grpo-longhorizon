"""盲评守卫：轨迹产物不得携带 TaskFacts 私有字段。"""

from __future__ import annotations

FORBIDDEN_KEYS = frozenset(
    {
        "expected_resolution",
        "acceptable",
        "ineligible",
        "requires_escalation",
        "requires_clarification",
        "suspicious_user",
        "max_compensation",
        "price_drop_amount",
        "soft_step_budget",
        "task_facts",
        "order_id",        # 私有事实中的主订单号；轨迹里只允许工具观测内的历史
        "facts",
    }
)


def guard_blind_trajectory(record: dict) -> list[str]:
    """返回 record 顶层与 task_flags 中出现的私有键。"""
    found = [key for key in record if key in FORBIDDEN_KEYS]
    flags = record.get("task_flags")
    if isinstance(flags, dict):
        # task_flags 只允许聚合安全的三类标志
        allowed = {"escalation_expected", "ineligible_expected", "clarification_expected"}
        found.extend(key for key in flags if key not in allowed)
    return sorted(set(found))

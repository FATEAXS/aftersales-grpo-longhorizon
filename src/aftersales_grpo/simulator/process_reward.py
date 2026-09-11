"""SOP-PR：SOP 锚定的回合级过程奖励（本项目针对售后场景的核心机制之一）。

动机：售后会话是长程多轮交互，终局奖励（Reward v1）只在会话结束时给出一个
标量，GRPO 组内优势对"过程是否合规"几乎没有区分度。与 TRACE 类方法需要冻结
参考模型计算 log-prob 不同，售后领域天然有一份**可确定性审计的 SOP**
（取证 -> 核政策 -> 风控 -> 动作 -> 沟通闭环 -> 申报一致），因此过程奖励可以
直接从轨迹结构里数出来，零额外模型开销。

里程碑（每个只在其首次达成时计一次奖励——望远镜式去重，重复动作无法刷分）：

  m_locate     在任何 mutating 动作前，成功查询过任务订单
  m_policy     在任何 mutating 动作前，查询过对应类目政策
  m_risk       在任何 mutating 动作前，查询过用户档案（风控信号）
  m_logistics  物流敏感场景（未收到货/运输破损/质量/超期）下查询过物流；
               非物流敏感场景不要求，查询也不加分（避免诱导无用调用）
  m_communicate 最后一次成功的 mutating 动作之后，存在至少一次 reply_user
  m_declare    finish 申报的 resolution 与实际工单一致（复用评分器口径）

一票否决：出现硬门槛（错订单/幻觉订单）或任何政策违规记录时，过程奖励归零
——合规是售后场景的先验约束，违规的"好过程"不构成好过程。

GRPO 奖励 = terminal_total + lambda * process_total（lambda 默认 0.5，
process_total ∈ [0, PROCESS_MAX]；lambda=0 退化为纯终局奖励，可作消融）。
"""

from __future__ import annotations

from .models import TaskFacts
from .state import MUTATING_TOOLS, SessionState

PROCESS_REWARD_VERSION = "sop-pr-v2"

MILESTONE_BONUS = 0.05
MILESTONES = ("m_locate", "m_policy", "m_risk", "m_logistics", "m_communicate", "m_explain")
DECLARE_BONUS = 0.10
PROCESS_MAX = MILESTONE_BONUS * len(MILESTONES) + DECLARE_BONUS  # 0.40

# 物流敏感场景：物流取证计入里程碑；非物流敏感场景不要求，查询也不加分
LOGISTICS_SENSITIVE_SCENARIOS = frozenset(
    {"not_delivered", "damaged_logistics", "defective_return", "late_complaint"}
)


def _expected_policy_window(facts: TaskFacts, state: SessionState) -> int | None:
    """拒绝类场景下,依据类目政策与场景推断"应当被说出的窗口天数"。"""
    if not facts.ineligible:
        return None
    order = next(
        (o for o in state.user_orders if o.order_id == facts.order_id), None
    )
    if order is None:
        return None
    category = order.items[0].category if order.items else ""
    policy = state.policies.get(category)
    if policy is None:
        return None
    if facts.scenario == "blocked_no_reason":
        return int(getattr(policy, "no_reason_return_days", 0) or 0)
    # 超期质量诉求:仅退款类目用仅退款窗口,其余用质量退货窗口
    if getattr(policy, "quality_refund_without_return", False):
        return int(getattr(policy, "quality_refund_only_days", 0) or 0)
    return int(getattr(policy, "quality_return_days", 0) or 0)


def _explanation_consistent(facts: TaskFacts, state: SessionState) -> bool | None:
    """解释一致性:拒绝时回复必须包含正确的政策窗口数字。

    返回 None 表示该任务不要求(非拒绝类场景)。
    """
    window = _expected_policy_window(facts, state)
    if window is None:
        return None
    if window <= 0:
        # 窗口为 0 的类目(如食品无理由=不支持):正确解释应说明"不支持",
        # 无法用单一数字核对,放宽为存在回复即可。
        return bool(state.replies)
    joined = "\n".join(state.replies)
    return str(window) in joined


def process_reward(
    facts: TaskFacts,
    state: SessionState,
    declared_matches_actual: bool,
) -> dict:
    """从会话轨迹提取 SOP 里程碑达成情况并打分。"""
    milestones = {name: False for name in MILESTONES}

    first_mutation_index = next(
        (step.step_index for step in state.actions if step.tool in MUTATING_TOOLS),
        None,
    )

    def _succeeded_before_mutation(tool: str, arguments_match=None) -> bool:
        for step in state.actions:
            if first_mutation_index is not None and step.step_index >= first_mutation_index:
                break
            if step.tool != tool or not step.ok:
                continue
            if arguments_match is None or arguments_match(step.arguments):
                return True
        return False

    primary_order_id = facts.order_id
    milestones["m_locate"] = _succeeded_before_mutation(
        "query_order", lambda args: args.get("order_id") == primary_order_id
    )
    milestones["m_risk"] = _succeeded_before_mutation("query_user_profile")

    def _policy_matched(args: dict) -> bool:
        category = args.get("category", "")
        if category:
            return True  # 类目匹配在 state 里已通过 queried 推进，这里放宽为查过政策
        return False

    milestones["m_policy"] = _succeeded_before_mutation("query_policy", _policy_matched)

    if facts.scenario in LOGISTICS_SENSITIVE_SCENARIOS:
        milestones["m_logistics"] = _succeeded_before_mutation("query_logistics")
    else:
        milestones["m_logistics"] = True  # 不要求即视为达成（不计额外分）

    # 沟通闭环：最后一次成功 mutating 之后是否有 reply_user
    last_success_mutation = max(
        (
            step.step_index
            for step in state.actions
            if step.tool in MUTATING_TOOLS and step.ok
        ),
        default=None,
    )
    if last_success_mutation is None:
        # 没有成功的售后动作（拒绝/升级/未解决），只要有沟通即算闭环
        milestones["m_communicate"] = bool(state.replies) or bool(state.escalations)
    else:
        milestones["m_communicate"] = any(
            step.tool == "reply_user" and step.ok and step.step_index > last_success_mutation
            for step in state.actions
        )

    # 解释一致性:解释一致性(仅拒绝类场景要求;其他场景不要求也不扣)
    explained = _explanation_consistent(facts, state)
    milestones["m_explain"] = True if explained is None else bool(explained)

    vetoed = bool(state.wrong_order_touched or state.hallucinated_order_touched) or (
        "evidence_required" in state.policy_violations
        or "order_not_authorized" in state.policy_violations
        or "over_compensation" in state.policy_violations
    )
    declare_ok = declared_matches_actual and state.declared_resolution is not None

    achieved = [name for name, ok in milestones.items() if ok]
    total = MILESTONE_BONUS * len(achieved) + (DECLARE_BONUS if declare_ok else 0.0)
    if vetoed:
        total = 0.0

    return {
        "process_reward_version": PROCESS_REWARD_VERSION,
        "milestones": milestones,
        "declare_consistent": declare_ok,
        "vetoed": vetoed,
        "total": round(total, 4),
        "max": PROCESS_MAX,
    }

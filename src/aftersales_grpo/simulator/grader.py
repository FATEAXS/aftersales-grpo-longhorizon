"""Reward v1：确定性的终局奖励。

输入只有两样：TaskFacts（任务私有事实）和 SessionState（会话实际发生的行为），
不含任何模型主观打分。完整决策表见 docs/reward-v1.md。

分数结构（先硬门槛、后基础分、再修正项）：

1. 硬门槛（命中直接返回平负分，不再累计）：
   - wrong_order          对他人订单执行过 mutating 动作         -> -0.60
   - hallucinated_order   对不存在的订单执行过 mutating 动作      -> -0.60
2. 终局基础分：
   - escalated：期望转人工（风控）且给出理由        -> 1.00
                可正常解决却转人工（过早升级）      -> 0.25
   - rejected：期望拒绝，且回复过用户（有解释）     -> 1.00
                期望拒绝但没有任何解释              -> 0.70
                可正常解决却拒绝                    -> 0.15
   - resolved：实际解决方式落入场景 acceptable 表，取对应分数；
               无任何工单且声明 unresolved          -> -0.35
               达到 max_steps 被强制结束            -> -0.40
   - 声明 resolution 与实际工单不符（虚报）          -> 分数封顶 0.50
3. 修正项（叠加后夹到 [-1, 1]）：
   - guard/政策被拒的 mutating 尝试                 -> 每次 -0.15，累计 -0.30 封顶
   - 超出 soft_step_budget 的每步                   -> -0.05，累计 -0.15 封顶
   - 重复动作（同参数 >= 3 次）                     -> -0.10
   - 解决完成但从未 reply_user（未告知用户）        -> -0.05
   - 声明虚报                                       -> -0.10
"""

from __future__ import annotations

from .models import TaskFacts
from .state import MUTATING_TOOLS, SessionState

REWARD_VERSION = "reward-v1"

WRONG_ORDER_PENALTY = -0.60
HALLUCINATED_ORDER_PENALTY = -0.60

POLICY_REJECTION_STEP_PENALTY = -0.15
POLICY_REJECTION_MAX_PENALTY = -0.30
OVERLONG_STEP_PENALTY = -0.05
OVERLONG_MAX_PENALTY = -0.15
REPEAT_PENALTY = -0.10
MISREPORT_PENALTY = -0.10
NO_REPLY_PENALTY = -0.05

UNRESOLVED_PENALTY = -0.35
FORCED_UNRESOLVED_PENALTY = -0.40
ESCALATION_BYPASS_PENALTY = -0.40
GUESS_WITHOUT_ASKING_CAP = 0.85
UNNECESSARY_COMPENSATION_PENALTY = -0.05
# 补偿券属于自然处置的场景(问题商品无法退货 / 货没收到)
COMPENSATION_NATURAL_SCENARIOS = frozenset(
    {"defective_refund_only", "not_delivered"}
)
PREMATURE_ESCALATION_SCORE = 0.25
REJECTED_WITHOUT_EXPLANATION_SCORE = 0.70
REJECTED_RESOLVABLE_SCORE = 0.15


def grade(facts: TaskFacts, state: SessionState) -> dict:
    """对已结束（或强制结束）的会话打分，返回可审计的 breakdown。"""
    traj = state.resolution_evidence()
    actual = traj["actual_resolutions"]

    penalties: dict[str, float] = {}
    violations = sorted(set(state.policy_violations))

    # ---- 1. 硬门槛 ----------------------------------------------------
    if state.wrong_order_touched:
        breakdown = _breakdown(
            facts,
            state,
            total=WRONG_ORDER_PENALTY,
            terminal="hard_gate:wrong_order",
            penalties={"wrong_order": WRONG_ORDER_PENALTY},
            violations=violations,
        )
        return _with_process(facts, state, breakdown, matches=False)

    if state.hallucinated_order_touched:
        breakdown = _breakdown(
            facts,
            state,
            total=HALLUCINATED_ORDER_PENALTY,
            terminal="hard_gate:hallucinated_order",
            penalties={"hallucinated_order": HALLUCINATED_ORDER_PENALTY},
            violations=violations,
        )
        return _with_process(facts, state, breakdown, matches=False)

    # ---- 2. 终局基础分 -------------------------------------------------
    declared = state.declared_resolution
    misreport = declared is not None and declared != "unresolved" and not actual and (
        declared != "rejected" or bool(state.tickets)
    )
    if facts.requires_escalation:
        base, base_reason = _score_expected_escalation(state)
    elif facts.ineligible:
        base, base_reason = _score_expected_rejection(state, traj)
    else:
        base, base_reason = _score_resolvable(facts, actual, state, traj)

    # 虚报：声明的 resolution 与环境工单事实不符。
    if declared is not None and declared != "unresolved":
        expected_actual = _declared_matches(declared, actual)
        if declared == "rejected":
            expected_actual = not state.tickets
        if declared == "escalated" and not state.escalations:
            expected_actual = False
        if not expected_actual:
            misreport = True

    if misreport:
        penalties["misreport"] = MISREPORT_PENALTY
        base = min(base, 0.50)

    # ---- 3. 修正项 -----------------------------------------------------
    rejected_mutations = sum(
        1
        for step in state.actions
        if step.tool in MUTATING_TOOLS and not step.ok
    )
    guard_penalty = max(
        POLICY_REJECTION_MAX_PENALTY, POLICY_REJECTION_STEP_PENALTY * rejected_mutations
    )
    if guard_penalty:
        penalties["rejected_mutations"] = guard_penalty

    overlong_steps = max(0, state.step_index - facts.soft_step_budget)
    overlong = max(
        OVERLONG_MAX_PENALTY, OVERLONG_STEP_PENALTY * overlong_steps
    )
    if overlong and base > 0:
        penalties["overlong"] = overlong

    if state.repeat_action_count >= 2:
        penalties["repeat_actions"] = REPEAT_PENALTY

    if base > 0 and actual and not traj["replied"]:
        penalties["no_reply_to_user"] = NO_REPLY_PENALTY

    # 创新机制 C「成本敏感满意度」:完美解决却滥发补偿券的,按业务成本观
    # 扣小额分。仅作用于"补偿并非该场景自然解"的可解决任务(食品/美妆等
    # 仅退款类、未收到货类补偿属于正常处置,不扣)。
    if (
        base >= 1.0
        and not facts.ineligible
        and not facts.requires_escalation
        and facts.scenario not in COMPENSATION_NATURAL_SCENARIOS
        and any(t["kind"] == "compensation" for t in traj["tickets"])
    ):
        penalties["unnecessary_compensation"] = UNNECESSARY_COMPENSATION_PENALTY

    # 澄清回合：歧义任务必须追问；不追问而"猜中"正确订单的，封顶 0.85——
    # 猜对是运气，流程上仍然错误（猜错则已被 wrong_order 硬门槛拦截）。
    if facts.requires_clarification and not traj.get("clarified") and base > 0.85:
        penalties["guessed_without_clarification"] = round(GUESS_WITHOUT_ASKING_CAP - base, 4)

    total = base + sum(penalties.values())
    total = round(max(-1.0, min(1.0, total)), 4)
    breakdown = _breakdown(
        facts,
        state,
        total=total,
        terminal=state.terminal_type or "unknown",
        base=base,
        base_reason=base_reason,
        penalties=penalties,
        violations=violations,
        actual_resolutions=actual,
        declared_resolution=declared,
        misreport=misreport,
    )
    return _with_process(facts, state, breakdown, matches=not misreport)


def _with_process(facts: TaskFacts, state: SessionState, breakdown: dict, matches: bool) -> dict:
    """把 SOP 过程奖励附到 breakdown；terminal 字段保持 Reward v1 口径不变。"""
    from .process_reward import process_reward

    breakdown["process"] = process_reward(facts, state, declared_matches_actual=matches)
    return breakdown


def _score_expected_escalation(state: SessionState) -> tuple[float, str]:
    if state.terminal_type == "escalated":
        return 1.0, "expected_escalation_done"
    approved_audit = [
        t for t in state.tickets if t.kind in ("refund", "return", "exchange")
    ]
    if approved_audit:
        # 风控用户的退款/退货申请被直接受理，属于合规事故。
        return ESCALATION_BYPASS_PENALTY, "suspicious_user_action_approved"
    if state.terminal_type == "unresolved_forced":
        return FORCED_UNRESOLVED_PENALTY, "max_steps_reached"
    return UNRESOLVED_PENALTY, "expected_escalation_not_done"


def _score_expected_rejection(state: SessionState, traj: dict) -> tuple[float, str]:
    audit_tickets = [
        t for t in state.tickets if t.kind in ("refund", "return", "exchange")
    ]
    if audit_tickets:
        return REJECTED_RESOLVABLE_SCORE, "ineligible_but_action_taken"
    if state.declared_resolution == "rejected":
        if traj["replied"]:
            return 1.0, "rejected_with_explanation"
        return REJECTED_WITHOUT_EXPLANATION_SCORE, "rejected_without_explanation"
    if state.terminal_type == "escalated":
        return 0.85, "ineligible_escalated_instead"
    return UNRESOLVED_PENALTY, "ineligible_unresolved"


def _score_resolvable(
    facts: TaskFacts, actual: list[str], state: SessionState, traj: dict
) -> tuple[float, str]:
    if not actual:
        if state.terminal_type == "escalated":
            return PREMATURE_ESCALATION_SCORE, "premature_escalation"
        if state.terminal_type == "unresolved_forced":
            return FORCED_UNRESOLVED_PENALTY, "max_steps_reached"
        if state.declared_resolution == "rejected":
            return REJECTED_RESOLVABLE_SCORE, "resolvable_but_rejected"
        return UNRESOLVED_PENALTY, "hangup_without_resolution"
    best = max(facts.acceptable.get(res, 0.0) for res in actual)
    matched = max(actual, key=lambda res: facts.acceptable.get(res, 0.0))
    return float(best), f"resolved_as:{matched}"


def _declared_matches(declared: str, actual: list[str]) -> bool:
    if declared == "refund_only_full":
        return "refund_only_full" in actual or "return_refund" in actual
    if declared == "return_refund":
        return "return_refund" in actual
    if declared == "exchange":
        return "exchange" in actual
    if declared == "refund_partial":
        return "refund_partial" in actual
    if declared == "compensation_only":
        return "compensation_only" in actual
    if declared == "rejected":
        return not actual
    if declared == "escalated":
        return "escalated" in actual
    return False


def _breakdown(
    facts: TaskFacts,
    state: SessionState,
    total: float,
    terminal: str,
    penalties: dict | None = None,
    violations: list | None = None,
    base: float | None = None,
    base_reason: str = "",
    actual_resolutions: list | None = None,
    declared_resolution: str | None = None,
    misreport: bool = False,
) -> dict:
    return {
        "reward_version": REWARD_VERSION,
        "task_id": facts.task_id,
        "total": total,
        "terminal": terminal,
        "base": base,
        "base_reason": base_reason,
        "penalties": dict(penalties or {}),
        "violations": list(violations or []),
        "actual_resolutions": list(actual_resolutions or []),
        "declared_resolution": declared_resolution,
        "misreport": misreport,
        "steps": state.step_index,
        "guard_rejections": state.guard_rejections,
        "repeat_action_count": state.repeat_action_count,
        "reward_valid": True,
    }

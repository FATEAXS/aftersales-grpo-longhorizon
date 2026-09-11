"""售后会话状态机：动作执行、客观政策校验、guard 与终局记录。

设计分层（与 docs/reward-v1.md 的口径一致）：

- 客观层（会话状态机校验并直接拒绝）：订单不存在 / 不属于当前用户、退款金额
  超订单实付、补偿超政策上限、虚拟商品不可退、定制商品不可换、未先查询订单
  就执行售后动作（证据不足）。
- 事实层（评分器依据 TaskFacts 判定）：用户真实诉求是否符合政策窗口、是否命中
  风控、解决方式是否最优。工单在客观层通过后即创建成功，是否"该这么做"由
  评分器审计——模拟真实平台的"先受理、后审核"。

会话结束途径：finish（显式终局）、escalate_to_human（转人工即终局）、达到
max_steps（强制 unresolved_forced）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .models import (
    MAX_STEPS_DEFAULT,
    AftersaleTicket,
    ActionResult,
    Order,
    RISK_REFUND_THRESHOLD,
    TaskFacts,
    UserProfile,
)
from .policies import compensation_cap

MUTATING_TOOLS = frozenset(
    {"apply_refund", "apply_return", "apply_exchange", "issue_compensation"}
)
TERMINAL_RESOLUTIONS = frozenset(
    {
        "refund_only_full",
        "return_refund",
        "exchange",
        "refund_partial",
        "compensation_only",
        "rejected",
        "escalated",
        "unresolved",
    }
)


def _canon_args(tool: str, args: dict) -> str:
    """动作去重键：tool + 排序后的参数 JSON。"""
    try:
        payload = json.dumps(args, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        payload = str(args)
    return f"{tool}:{payload}"


@dataclass
class StepRecord:
    tool: str
    arguments: dict
    ok: bool
    error: str | None
    step_index: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class SessionState:
    facts: TaskFacts
    user: UserProfile
    user_orders: list[Order]                 # 当前用户全部订单（含干扰订单）
    policies: dict = field(default_factory=dict)      # category -> Policy
    all_orders: list = field(default_factory=list)    # 全局订单（识别幻觉订单号）
    product_index: dict = field(default_factory=dict) # product_id -> product dict
    queried_orders: set = field(default_factory=set)   # 已成功查询过的 order_id
    actions: list[StepRecord] = field(default_factory=list)
    tickets: list[AftersaleTicket] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)   # reply_user 内容
    escalations: list[dict] = field(default_factory=list)
    guard_rejections: int = 0
    guard_reason_counts: dict = field(default_factory=dict)
    repeat_action_count: int = 0
    action_attempt_count: int = 0
    wrong_order_touched: bool = False        # 对非任务订单执行过 mutating 动作
    hallucinated_order_touched: bool = False # 对不存在的订单执行过 mutating 动作
    policy_violations: list[str] = field(default_factory=list)
    declared_resolution: str | None = None
    summary: str | None = None
    done: bool = False
    terminal_type: str | None = None         # finished / escalated / unresolved_forced
    step_index: int = 0
    # 澄清回合机制：requires_clarification 的任务里，agent 第一次"提问式"回复
    # 会触发确定性用户应答（给出正确订单号）；此后 clarified=True。
    user_responses: list[str] = field(default_factory=list)
    clarified: bool = False
    # 创新机制 E「时间线一致性校验」：环境日期 + 用户口述天数比对
    today: str = ""
    timeline_mismatch_reported: bool = False
    # P2-10 观测噪声:1 时 query_order 掺入无关字段,考验信息筛选
    noise_level: int = 0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def run_action(self, tool: str, arguments: dict) -> ActionResult:
        """执行一次工具调用；返回结果并把该调用记入轨迹。"""
        if self.done:
            return ActionResult.reject(
                "session_closed", {"error": "会话已结束，不能再执行动作。"}
            )
        if self.step_index >= self.max_steps:
            self._force_close()
            return ActionResult.reject(
                "max_steps", {"error": "已达最大步数，会话结束。"}
            )

        self.step_index += 1
        self.action_attempt_count += 1
        handler = getattr(self, f"_do_{tool}", None)
        if handler is None:
            return self._guard(tool, arguments, f"unknown_tool:{tool}")
        arguments = dict(arguments or {})
        result = handler(arguments)
        if result.ok and tool in MUTATING_TOOLS:
            order = next(
                (
                    o
                    for o in self.user_orders
                    if o.order_id == str(arguments.get("order_id", ""))
                ),
                None,
            )
            if order is not None:
                warning = self._timeline_warning(order)
                if warning:
                    result.observation = {**result.observation, **warning}
        self._record(tool, arguments, result)
        return result

    @property
    def max_steps(self) -> int:
        return MAX_STEPS_DEFAULT

    # ------------------------------------------------------------------
    # 记录与 guard
    # ------------------------------------------------------------------

    def _record(self, tool: str, arguments: dict, result: ActionResult) -> None:
        self.actions.append(
            StepRecord(
                tool=tool,
                arguments=arguments,
                ok=result.ok,
                error=result.error,
                step_index=self.step_index,
            )
        )
        canon = _canon_args(tool, arguments)
        repeats = sum(
            1
            for step in self.actions
            if step.ok and _canon_args(step.tool, step.arguments) == canon
        )
        if repeats >= 3:
            self.repeat_action_count += 1

    def _guard(self, tool: str, arguments: dict, reason: str) -> ActionResult:
        self.guard_rejections += 1
        self.guard_reason_counts[reason] = self.guard_reason_counts.get(reason, 0) + 1
        self.actions.append(
            StepRecord(
                tool=tool,
                arguments=arguments,
                ok=False,
                error=reason,
                step_index=self.step_index,
            )
        )
        return ActionResult.reject(
            reason,
            {"error": f"动作被环境 guard 拒绝：{reason}。", "tool": tool},
        )

    def _mutating_guard(self, tool: str, arguments: dict) -> Order | None:
        """mutating 动作的公共校验；通过返回订单对象，否则返回 None。"""
        order_id = str(arguments.get("order_id", "")).strip()
        if not order_id:
            self.policy_violations.append("missing_order_id")
            return None
        order = next((o for o in self.user_orders if o.order_id == order_id), None)
        if order is None:
            # 区分"别人的真实订单"（张冠李戴）与"凭空编造的订单号"（幻觉）。
            if self._order_exists_globally(order_id):
                self.wrong_order_touched = True
            else:
                self.hallucinated_order_touched = True
            self.policy_violations.append("order_not_authorized")
            return None
        if order.order_id != self.facts.order_id:
            self.wrong_order_touched = True
        if order_id not in self.queried_orders:
            self.policy_violations.append("evidence_required")
            return None
        return order

    def _order_exists_globally(self, order_id: str) -> bool:
        return any(o.order_id == order_id for o in self.all_orders)

    def _timeline_warning(self, order: Order) -> dict | None:
        """创新机制 E:用户口述购买时间与订单签收记录不符时给出环境警告。

        真实客服不能全信用户陈述——该警告让模型学会交叉核对时间线,
        既是服务质量的证据面,也是"口误/欺诈陈述"防御的训练信号。
        """
        if self.today == "" or order.delivered_at is None:
            return None
        import re as _re
        from datetime import date as _date

        message = self.facts.opening_message
        claimed = None
        match = _re.search(r"(\d+)\s*天前", message) or _re.search(
            r"(\d+)\s*前买", message
        )
        if match:
            claimed = int(match.group(1))
        if claimed is None:
            return None
        actual = (_date.fromisoformat(self.today) - _date.fromisoformat(order.delivered_at)).days
        if abs(claimed - actual) < 3:
            return None
        self.timeline_mismatch_reported = True
        return {
            "timeline_warning": (
                f"用户陈述与订单记录不符：用户称 {claimed} 天前购买，"
                f"记录显示该订单 {actual} 天前签收。请以订单记录为准并向用户核实。"
            )
        }

    # ------------------------------------------------------------------
    # 查询类工具
    # ------------------------------------------------------------------

    def _do_query_user_orders(self, arguments: dict) -> ActionResult:
        return ActionResult(
            ok=True,
            observation={
                "orders": [
                    {
                        "order_id": o.order_id,
                        "items_summary": "；".join(
                            f"{i.name}({i.options})x{i.qty}" for i in o.items
                        ),
                        "total_amount": o.total_amount,
                        "status": o.status,
                        "delivered_at": o.delivered_at,
                    }
                    for o in self.user_orders
                ]
            },
        )

    def _do_query_order(self, arguments: dict) -> ActionResult:
        order_id = str(arguments.get("order_id", "")).strip()
        order = next((o for o in self.user_orders if o.order_id == order_id), None)
        if order is None:
            return ActionResult.reject(
                "order_not_found",
                {"error": f"订单 {order_id!r} 不存在或不属于当前用户。"},
            )
        self.queried_orders.add(order_id)
        items = []
        for item in order.items:
            entry = item.__dict__
            entry["available_options"] = self._product_options(item.product_id)
            if self.noise_level:
                entry["warehouse_code"] = f"W{abs(hash(order.order_id)) % 90 + 10}"
                entry["supplier_id"] = f"SUP{abs(hash(item.product_id)) % 900 + 100}"
            items.append(entry)
        observation = {"order": {**order.to_public_dict(), "items": items}}
        if self.noise_level:
            observation["order"]["internal_risk_tag"] = "AUTO_SCAN_PASS" if order.total_amount < 500 else "MANUAL_REVIEW_DONE"
        return ActionResult(ok=True, observation=observation)

    def _do_query_logistics(self, arguments: dict) -> ActionResult:
        order_id = str(arguments.get("order_id", "")).strip()
        order = next((o for o in self.user_orders if o.order_id == order_id), None)
        if order is None:
            return ActionResult.reject(
                "order_not_found",
                {"error": f"订单 {order_id!r} 不存在或不属于当前用户。"},
            )
        self.queried_orders.add(order_id)
        return ActionResult(
            ok=True,
            observation={
                "order_id": order.order_id,
                "status": order.status,
                "events": [event.to_dict() for event in order.logistics],
            },
        )

    def _do_query_user_profile(self, arguments: dict) -> ActionResult:
        return ActionResult(
            ok=True, observation={"profile": self.user.to_public_dict()}
        )

    def _do_query_policy(self, arguments: dict) -> ActionResult:
        category = str(arguments.get("category", "")).strip()
        policy = self.policies.get(category)
        if policy is None:
            return ActionResult.reject(
                "category_not_found",
                {
                    "error": f"未知类目 {category!r}。",
                    "available_categories": sorted(self.policies),
                },
            )
        cap = compensation_cap(
            self._task_order_amount(), self.user.level
        )
        observation = policy.to_dict()
        observation["compensation_cap_for_current_user"] = cap
        return ActionResult(ok=True, observation={"policy": observation})

    def _task_order_amount(self) -> float:
        order = next(
            (o for o in self.user_orders if o.order_id == self.facts.order_id),
            None,
        )
        return order.total_amount if order else 0.0

    # ------------------------------------------------------------------
    # 售后执行类工具（客观层校验）
    # ------------------------------------------------------------------

    def _do_apply_refund(self, arguments: dict) -> ActionResult:
        order = self._mutating_guard("apply_refund", arguments)
        if order is None:
            return ActionResult.reject(
                "invalid_after_sales_action",
                {"error": "无法执行退款：订单校验未通过（需先 query_order 且订单属于当前用户）。"},
            )
        refund_type = str(arguments.get("refund_type", "")).strip()
        reason = str(arguments.get("reason", "")).strip()
        if refund_type not in {"full", "partial"}:
            return ActionResult.reject(
                "invalid_arguments",
                {"error": "refund_type 必须是 'full' 或 'partial'。"},
            )
        if not reason:
            return ActionResult.reject(
                "invalid_arguments", {"error": "必须提供退款原因 reason。"}
            )
        if order.status == "closed":
            return ActionResult.reject(
                "order_closed", {"error": "订单已关闭，不能再次退款。"}
            )
        amount = order.total_amount if refund_type == "full" else float(
            arguments.get("amount", 0.0) or 0.0
        )
        if refund_type == "partial" and not (0 < amount < order.total_amount * 0.98):
            return ActionResult.reject(
                "invalid_amount",
                {"error": f"partial 退款金额必须在 (0, {order.total_amount:.2f}) 内。"},
            )
        if "虚拟商品" in {i.category for i in order.items}:
            self.policy_violations.append("virtual_no_refund")
            return ActionResult.reject(
                "policy_rejected",
                {"error": "虚拟商品一经售出不支持退款（政策硬性限制）。"},
            )
        ticket = AftersaleTicket(
            ticket_id=f"T{len(self.tickets) + 1:04d}",
            order_id=order.order_id,
            kind="refund",
            amount=round(amount, 2),
            reason=reason,
            status="approved",
        )
        self.tickets.append(ticket)
        order.status = "closed"
        return ActionResult(
            ok=True,
            observation={
                "result": "approved",
                "ticket": ticket.to_dict(),
                "message": f"退款申请已通过，金额 {ticket.amount:.2f} 元。",
            },
        )

    def _do_apply_return(self, arguments: dict) -> ActionResult:
        order = self._mutating_guard("apply_return", arguments)
        if order is None:
            return ActionResult.reject(
                "invalid_after_sales_action",
                {"error": "无法执行退货：订单校验未通过（需先 query_order 且订单属于当前用户）。"},
            )
        reason = str(arguments.get("reason", "")).strip()
        if not reason:
            return ActionResult.reject(
                "invalid_arguments", {"error": "必须提供退货原因 reason。"}
            )
        if "虚拟商品" in {i.category for i in order.items}:
            self.policy_violations.append("virtual_no_return")
            return ActionResult.reject(
                "policy_rejected", {"error": "虚拟商品不支持退货（政策硬性限制）。"}
            )
        ticket = AftersaleTicket(
            ticket_id=f"T{len(self.tickets) + 1:04d}",
            order_id=order.order_id,
            kind="return",
            amount=order.total_amount,
            reason=reason,
            status="approved",
        )
        self.tickets.append(ticket)
        order.status = "closed"
        return ActionResult(
            ok=True,
            observation={
                "result": "approved",
                "ticket": ticket.to_dict(),
                "message": f"退货申请已通过，审核通过后退款 {ticket.amount:.2f} 元。",
            },
        )

    def _do_apply_exchange(self, arguments: dict) -> ActionResult:
        order = self._mutating_guard("apply_exchange", arguments)
        if order is None:
            return ActionResult.reject(
                "invalid_after_sales_action",
                {"error": "无法执行换货：订单校验未通过（需先 query_order 且订单属于当前用户）。"},
            )
        new_option = str(arguments.get("new_option", "")).strip()
        reason = str(arguments.get("reason", "")).strip()
        if not new_option or not reason:
            return ActionResult.reject(
                "invalid_arguments", {"error": "必须提供 new_option 与 reason。"}
            )
        categories = {i.category for i in order.items}
        if "虚拟商品" in categories:
            self.policy_violations.append("virtual_no_exchange")
            return ActionResult.reject(
                "policy_rejected", {"error": "虚拟商品不支持换货（政策硬性限制）。"}
            )
        if "定制商品" in categories:
            self.policy_violations.append("custom_no_exchange")
            return ActionResult.reject(
                "policy_rejected", {"error": "定制商品不支持换货，质量问题请走退货流程（政策硬性限制）。"}
            )
        known_options = {opt for item in order.items for opt in self._product_options(
            item.product_id
        )}
        if known_options and new_option not in known_options:
            return ActionResult.reject(
                "invalid_arguments",
                {
                    "error": f"换货规格 {new_option!r} 不在该商品可选规格中。",
                    "available_options": sorted(known_options),
                },
            )
        ticket = AftersaleTicket(
            ticket_id=f"T{len(self.tickets) + 1:04d}",
            order_id=order.order_id,
            kind="exchange",
            amount=order.total_amount,
            reason=reason,
            status="approved",
            new_option=new_option,
        )
        self.tickets.append(ticket)
        return ActionResult(
            ok=True,
            observation={
                "result": "approved",
                "ticket": ticket.to_dict(),
                "message": f"换货申请已通过，将为您更换为“{new_option}”。",
            },
        )

    def _do_issue_compensation(self, arguments: dict) -> ActionResult:
        order = self._mutating_guard("issue_compensation", arguments)
        if order is None:
            return ActionResult.reject(
                "invalid_after_sales_action",
                {"error": "无法发放补偿：订单校验未通过（需先 query_order 且订单属于当前用户）。"},
            )
        try:
            amount = round(float(arguments.get("amount", 0.0)), 2)
        except (TypeError, ValueError):
            return ActionResult.reject(
                "invalid_arguments", {"error": "amount 必须是数字。"}
            )
        reason = str(arguments.get("reason", "")).strip()
        if amount <= 0 or not reason:
            return ActionResult.reject(
                "invalid_arguments", {"error": "必须提供正数 amount 与 reason。"}
            )
        cap = compensation_cap(order.total_amount, self.user.level)
        if amount > cap + 1e-9:
            self.policy_violations.append("over_compensation")
            return ActionResult.reject(
                "policy_rejected",
                {
                    "error": f"补偿金额 {amount:.2f} 元超过当前用户上限 {cap:.2f} 元（政策硬性限制）。",
                },
            )
        ticket = AftersaleTicket(
            ticket_id=f"T{len(self.tickets) + 1:04d}",
            order_id=order.order_id,
            kind="compensation",
            amount=amount,
            reason=reason,
            status="approved",
        )
        self.tickets.append(ticket)
        return ActionResult(
            ok=True,
            observation={
                "result": "approved",
                "ticket": ticket.to_dict(),
                "message": f"已为您发放 {amount:.2f} 元补偿券。",
            },
        )

    def _product_options(self, product_id: str) -> list[str]:
        return (self.product_index.get(product_id) or {}).get(
            "options", []
        )

    # ------------------------------------------------------------------
    # 沟通与终局
    # ------------------------------------------------------------------

    def _do_reply_user(self, arguments: dict) -> ActionResult:
        message = str(arguments.get("message", "")).strip()
        if not message:
            return ActionResult.reject(
                "invalid_arguments", {"error": "message 不能为空。"}
            )
        self.replies.append(message)
        observation = {"sent": True, "message": message}
        # 澄清回合：歧义任务中，第一次向用户提问会得到确定性的订单号应答。
        if (
            self.facts.requires_clarification
            and not self.clarified
            and self._looks_like_question(message)
        ):
            self.clarified = True
            self.user_responses.append(
                f"哦我想想，订单号是 {self.facts.order_id}，就这个订单，麻烦你了。"
            )
            observation["user_reply"] = self.user_responses[-1]
        return ActionResult(ok=True, observation=observation)

    @staticmethod
    def _looks_like_question(message: str) -> bool:
        return ("订单" in message and ("？" in message or "?" in message or "哪" in message or "麻烦" in message or "确认" in message)) or message.rstrip().endswith(("？", "?"))

    def _do_escalate_to_human(self, arguments: dict) -> ActionResult:
        order_id = str(arguments.get("order_id", "")).strip()
        reason = str(arguments.get("reason", "")).strip()
        if not reason:
            return ActionResult.reject(
                "invalid_arguments", {"error": "必须提供升级原因 reason。"}
            )
        if order_id and order_id != self.facts.order_id:
            self.wrong_order_touched = True
        self.escalations.append(
            {"order_id": order_id or self.facts.order_id, "reason": reason}
        )
        self.done = True
        self.terminal_type = "escalated"
        return ActionResult(
            ok=True,
            observation={
                "result": "escalated",
                "message": "已为您转接人工专属客服，请稍候。",
            },
        )

    def _do_finish(self, arguments: dict) -> ActionResult:
        resolution = str(arguments.get("resolution", "")).strip()
        summary = str(arguments.get("summary", "")).strip()
        if resolution not in TERMINAL_RESOLUTIONS:
            return ActionResult.reject(
                "invalid_arguments",
                {
                    "error": "resolution 必须是以下之一：",
                    "allowed": sorted(TERMINAL_RESOLUTIONS),
                },
            )
        if not summary:
            return ActionResult.reject(
                "invalid_arguments", {"error": "必须提供处理总结 summary。"}
            )
        self.declared_resolution = resolution
        self.summary = summary
        self.done = True
        self.terminal_type = "finished"
        return ActionResult(
            ok=True, observation={"result": "finished", "resolution": resolution}
        )

    def _force_close(self) -> None:
        self.done = True
        self.terminal_type = "unresolved_forced"

    # ------------------------------------------------------------------
    # 终局事实汇总（供评分器与轨迹导出使用）
    # ------------------------------------------------------------------

    def resolution_evidence(self) -> dict:
        """把工单事实折算成"实际发生的解决方式"及其佐证。"""
        approved = [t for t in self.tickets if t.status == "approved"]
        refunds = [t for t in approved if t.kind == "refund"]
        returns = [t for t in approved if t.kind == "return"]
        exchanges = [t for t in approved if t.kind == "exchange"]
        compensations = [t for t in approved if t.kind == "compensation"]
        task_amount = self._task_order_amount()
        refund_full = any(t.amount >= task_amount * 0.98 for t in refunds)
        refund_partial = any(0 < t.amount < task_amount * 0.98 for t in refunds)
        actual: list[str] = []
        if returns:
            actual.append("return_refund")
        if exchanges:
            actual.append("exchange")
        if refund_full:
            actual.append("refund_only_full")
        if refund_partial:
            actual.append("refund_partial")
        if compensations and not refunds and not returns and not exchanges:
            actual.append("compensation_only")
        return {
            "actual_resolutions": actual,
            "tickets": [t.to_dict() for t in self.tickets],
            "replied": bool(self.replies),
            "escalated": bool(self.escalations),
            "clarified": self.clarified,
            "user_responses": list(self.user_responses),
        }

    def trajectory_summary(self) -> dict:
        return {
            "task_id": self.facts.task_id,
            "steps": self.step_index,
            "max_steps": self.max_steps,
            "done": self.done,
            "terminal_type": self.terminal_type,
            "declared_resolution": self.declared_resolution,
            "summary": self.summary,
            "actions": [step.to_dict() for step in self.actions],
            "tickets": [t.to_dict() for t in self.tickets],
            "replies": list(self.replies),
            "escalations": list(self.escalations),
            "guard_rejections": self.guard_rejections,
            "guard_reason_counts": dict(self.guard_reason_counts),
            "repeat_action_count": self.repeat_action_count,
            "action_attempt_count": self.action_attempt_count,
            "wrong_order_touched": self.wrong_order_touched,
            "hallucinated_order_touched": self.hallucinated_order_touched,
            "policy_violations": sorted(set(self.policy_violations)),
            "user_level": self.user.level,
            "user_refund_count_90d": self.user.refund_count_90d,
            "requires_clarification": self.facts.requires_clarification,
            "clarified": self.clarified,
            "timeline_mismatch_reported": self.timeline_mismatch_reported,
        }

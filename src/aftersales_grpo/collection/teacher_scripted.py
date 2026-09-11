"""脚本化参考教师：把售后 SOP 实现成确定性策略。

它是教师轨迹采集的"零成本冷启动"路径，也是 SOP 的可执行参考实现。
只使用与真实 Agent 完全相同的观测（开场白 + 工具返回），执行结构为：

  取证阶段：query_user_orders -> query_order -> query_logistics
            -> query_user_profile -> query_policy（定位订单并识别口误订单号）
  计划阶段：按决策树生成动作序列 [售后动作?] -> reply_user -> finish/escalate
  执行阶段：逐条发出计划动作

LLM 教师（teacher_api.py）在同一个环境里拿到同样的工具；采集脚本按 Reward v1
统一验收两种教师的轨迹，来源在 metadata 中区分。
"""

from __future__ import annotations

import json
import re
from datetime import date

# 用户消息中出现这些关键词视为"质量问题"诉求（模板受控，覆盖全部缺陷描述）。
QUALITY_KEYWORDS = (
    "质量", "坏", "破", "裂", "断", "霉", "变质", "过敏", "异味", "刺鼻",
    "头发", "漏气", "漏液", "分层", "掉色", "起球", "开线", "失灵", "松",
    "亮点", "过期", "声音", "无法", "脱落", "看不清", "完全不同", "漏水",
    "刻错", "错做",
)
SIZE_KEYWORDS = ("码", "尺码", "小了", "不合适")
WRONG_ITEM_KEYWORDS = ("发错", "发成", "和收到的不一样")
RISK_REFUND_THRESHOLD = 6


class ScriptedTeacher:
    """在环境会话中逐步决策的确定性教师。"""

    def __init__(self) -> None:
        self.history: list[tuple[str, dict, str]] = []
        self.plan: list[tuple[str, dict]] | None = None
        self.plan_index = 0
        self.asked_clarification = False

    # ------------------------------------------------------------------

    def observe(self, tool: str, arguments: dict, observation_text: str) -> None:
        self.history.append((tool, arguments, observation_text))

    @property
    def finished(self) -> bool:
        return self.plan is not None and self.plan_index >= len(self.plan)

    def next_action(self, opening: dict) -> tuple[str, dict] | None:
        """返回下一个 (tool, arguments)；会话结束时返回 None。"""
        if self.finished:
            return None
        if self.plan is not None:
            action = self.plan[self.plan_index]
            self.plan_index += 1
            return action
        # 还在取证阶段；取证齐全时 _next_query 内部会构建计划。
        return self._next_query(opening)

    # ------------------------------------------------------------------
    # 取证阶段
    # ------------------------------------------------------------------

    def _next_query(self, opening: dict) -> tuple[str, dict]:
        message = opening.get("user_message", "")
        queries_done = {tool for tool, _, _ in self.history}

        if "query_user_orders" not in queries_done:
            return "query_user_orders", {}
        # 澄清回合：同名订单歧义（无订单号、多笔进行中）必须先追问用户，
        # 环境的确定性用户应答会给出订单号。
        if self._ambiguous(message) and not self.asked_clarification:
            self.asked_clarification = True
            return "reply_user", {
                "message": "您好，看到您名下有两笔同款商品的订单，请问您要处理的是哪一笔呢？麻烦提供一下订单号，我马上为您核实处理。"
            }
        primary_id = self._locate_primary(message)
        if primary_id is None:
            # 订单定位失败：直接以"未解决"收尾
            self.plan = [
                ("reply_user", {"message": "您好，未能查询到您所说的订单，请提供订单号，我再为您核实。"}),
                ("finish", {"resolution": "unresolved", "summary": "未能定位用户订单"}),
            ]
            return self.plan[0]
        if not any(
            t == "query_order" and a == {"order_id": primary_id}
            for t, a, _ in self.history
        ):
            return "query_order", {"order_id": primary_id}
        if not any(t == "query_logistics" for t, _, _ in self.history):
            return "query_logistics", {"order_id": primary_id}
        if "query_user_profile" not in queries_done:
            return "query_user_profile", {}
        category = self._category_of(opening)
        if not any(
            t == "query_policy" and a.get("category") == category
            for t, a, _ in self.history
        ):
            return "query_policy", {"category": category}
        # 取证已齐，进入计划阶段
        self.plan = self._build_plan(opening)
        if self.plan is None:
            self.plan = [
                ("finish", {"resolution": "unresolved", "summary": "证据不足，未能处理"}),
            ]
        action = self.plan[0]
        self.plan_index = 1
        return action

    def _build_plan(self, opening: dict) -> list[tuple[str, dict]] | None:
        # 取证未齐时不建计划
        message = opening.get("user_message", "")
        queries_done = {tool for tool, _, _ in self.history}
        if not {"query_user_orders", "query_user_profile", "query_logistics"} <= queries_done:
            return None
        if "query_order" not in queries_done:
            return None
        primary_id = self._locate_primary(message)
        if not any(
            t == "query_policy" and a.get("category") == self._category_of(opening)
            for t, a, _ in self.history
        ):
            return None

        order = (self._obs_of("query_order") or {}).get("order", {})
        if order.get("order_id") != primary_id:
            return None
        item = (order.get("items") or [{}])[0]
        category = item.get("category", "")
        policy = (self._obs_of("query_policy") or {}).get("policy", {})
        profile = (self._obs_of("query_user_profile") or {}).get("profile", {})
        logistics = self._obs_of("query_logistics") or {}

        today = date.fromisoformat(opening.get("today", "2026-09-08"))
        delivered_at = order.get("delivered_at")
        days = (today - date.fromisoformat(delivered_at)).days if delivered_at else None

        # ---- 决策树（顺序即优先级）----
        if logistics.get("status") not in {"delivered", "closed"}:
            return [
                ("apply_refund", {
                    "order_id": primary_id, "refund_type": "full",
                    "reason": "物流滞留未收到货",
                }),
                ("reply_user", {"message": "包裹物流异常，已为您办理全额退款，请留意到账，欢迎重新选购。"}),
                ("finish", {"resolution": "refund_only_full", "summary": "未收到货全额退款"}),
            ]

        if any(k in message for k in WRONG_ITEM_KEYWORDS):
            return self._plan_exchange_or_return(
                order, primary_id, "商家发错规格", "exchange", "已为您换发正确规格，请留意查收。"
            )

        has_quality = any(k in message for k in QUALITY_KEYWORDS)
        has_size = any(k in message for k in SIZE_KEYWORDS)
        price_drop = self._parse_price_drop(message)

        if has_size and not has_quality:
            window = int(policy.get("size_exchange_days") or 0)
            if window and days is not None and days <= window:
                return self._plan_exchange_or_return(
                    order, primary_id, "尺码不合适", "exchange", "已为您办理换货，请留意查收。"
                )
            if days is not None and int(policy.get("no_reason_return_days") or 0) and days <= int(
                policy.get("no_reason_return_days")
            ):
                return self._plan_return(primary_id, "尺码不合适，无理由退货", "已为您办理退货退款。")
            return self._plan_reject(
                "您好，尺码问题的换货窗口为签收后 "
                f"{policy.get('size_exchange_days', 0)} 天，您的订单已超出窗口，"
                "很抱歉无法为您办理，感谢理解。"
            )

        if price_drop:
            window = int(policy.get("price_protection_days") or 0)
            if window and days is not None and days <= window:
                return [
                    ("apply_refund", {
                        "order_id": primary_id, "refund_type": "partial",
                        "amount": price_drop, "reason": "价保期内补差价",
                    }),
                    ("reply_user", {"message": f"价保审核通过，已为您退还差价 {price_drop:g} 元，请留意到账。"}),
                    ("finish", {"resolution": "refund_partial", "summary": "价保差价已退还"}),
                ]
            return self._plan_reject(
                f"您好，该商品价保期为签收后 {window} 天，您的订单已超出价保期，"
                "很抱歉无法为您补差价，感谢理解。"
            )

        if has_quality:
            if policy.get("quality_refund_without_return"):
                window = int(policy.get("quality_refund_only_days") or 0)
            else:
                window = int(policy.get("quality_return_days") or 0)
            in_window = bool(window and days is not None and days <= window)
            if in_window and policy.get("quality_refund_without_return"):
                return [
                    ("apply_refund", {
                        "order_id": primary_id, "refund_type": "full",
                        "reason": "质量问题，仅退款不退货",
                    }),
                    ("reply_user", {"message": "非常抱歉给您带来不好的体验。已为您办理全额退款，问题商品无需寄回，请留意到账。"}),
                    ("finish", {"resolution": "refund_only_full", "summary": "质量问题仅退款"}),
                ]
            if in_window:
                return self._plan_return(
                    primary_id, "质量问题退货，运费由商家承担",
                    "非常抱歉给您带来不好的体验。已为您办理退货退款，运费由商家承担。",
                )
            return self._plan_reject(
                "您好，非常抱歉给您带来不好的体验。质量问题退货期为签收后 "
                f"{window} 天，您的订单已超出该期限，很抱歉无法为您办理退货。"
                "如有其他问题欢迎随时联系我们。"
            )

        if int(profile.get("refund_count_90d") or 0) >= RISK_REFUND_THRESHOLD:
            count = profile.get("refund_count_90d")
            return [
                ("escalate_to_human", {
                    "order_id": primary_id,
                    "reason": f"用户 90 天内退款 {count} 次，命中风控，诉求转人工审核",
                }),
            ]

        window = int(policy.get("no_reason_return_days") or 0)
        if window and days is not None and days <= window:
            return self._plan_return(primary_id, "七天无理由退货", "已为您办理无理由退货退款。")
        return self._plan_reject(
            f"您好，经核查，该商品无理由退货窗口为签收后 {window} 天，您的订单"
            "已不符合办理条件，很抱歉无法为您办理。如有质量问题欢迎联系我们核实。"
        )

    # ------------------------------------------------------------------
    # 计划片段
    # ------------------------------------------------------------------

    def _plan_exchange_or_return(
        self, order: dict, order_id: str, reason: str,
        resolution: str, reply: str,
    ) -> list[tuple[str, dict]]:
        items = order.get("items") or []
        current = items[0].get("options", "") if items else ""
        available = items[0].get("available_options") or [] if items else []
        others = [opt for opt in available if opt != current]
        if others:
            target = others[0]
            return [
                ("apply_exchange", {
                    "order_id": order_id, "new_option": target, "reason": reason,
                }),
                ("reply_user", {"message": reply + f"本次换货规格：{target}。"}),
                ("finish", {"resolution": resolution, "summary": reply}),
            ]
        return self._plan_return(order_id, reason, reply)

    def _plan_return(self, order_id: str, reason: str, reply: str) -> list[tuple[str, dict]]:
        return [
            ("apply_return", {"order_id": order_id, "reason": reason}),
            ("reply_user", {"message": reply}),
            ("finish", {"resolution": "return_refund", "summary": reply}),
        ]

    def _plan_reject(self, reply: str) -> list[tuple[str, dict]]:
        return [
            ("reply_user", {"message": reply}),
            ("finish", {"resolution": "rejected", "summary": reply}),
        ]

    # ------------------------------------------------------------------
    # 观测解析
    # ------------------------------------------------------------------

    def _category_of(self, opening: dict) -> str:
        """从已查询的订单观测推断类目（仅在 query_order 之后有效）。"""
        order = (self._obs_of("query_order") or {}).get("order", {})
        items = order.get("items") or []
        if items:
            return items[0].get("category", "")
        return ""

    def _locate_primary(self, message: str) -> str | None:
        """订单定位：商品名命中的候选里，优先"进行中"的订单和被引用的订单号。"""
        cited = re.findall(r"O\d{5,7}", message + " " + self._user_reply_text())
        orders = (self._obs_of("query_user_orders") or {}).get("orders", [])
        if not orders:
            return cited[0] if cited else None
        named = []
        for o in orders:
            summary = o.get("items_summary", "")
            name = summary.split("(")[0] if "(" in summary else summary
            if name and name in message:
                named.append(o)
        known = {o["order_id"] for o in orders}
        if named:
            # 同名候选里投诉一般针对进行中的订单；被引用的订单号只有落在
            # "进行中"候选里才采信（口误引用已关闭的旧订单是典型干扰）。
            active = [o for o in named if o.get("status") not in {"closed"}]
            target_pool = active or named
            pool_ids = {o["order_id"] for o in target_pool}
            for oid in cited:
                if oid in pool_ids:
                    return oid
            return target_pool[0]["order_id"]
        for oid in cited:
            if oid in known:
                return oid
        return cited[0] if cited else (orders[0]["order_id"] if len(orders) == 1 else None)

    def _ambiguous(self, message: str) -> bool:
        """无订单号且商品名命中多笔进行中订单 -> 需要澄清。"""
        if re.findall(r"O\d{5,7}", message):
            return False
        orders = (self._obs_of("query_user_orders") or {}).get("orders", [])
        named = []
        for o in orders:
            summary = o.get("items_summary", "")
            name = summary.split("(")[0] if "(" in summary else summary
            if name and name in message:
                named.append(o)
        active = [o for o in named if o.get("status") not in {"closed"}]
        return len(active) >= 2

    def _user_reply_text(self) -> str:
        """取最近一次 reply_user 观测中确定性用户的应答文本。"""
        for tool, _, obs_text in reversed(self.history):
            if tool != "reply_user":
                continue
            try:
                payload = json.loads(obs_text)
            except (TypeError, ValueError):
                continue
            reply = payload.get("user_reply")
            if reply:
                return str(reply)
        return ""

    def _parse_price_drop(self, message: str) -> float | None:
        match = re.search(r"降价\s*([0-9]+(?:\.[0-9]+)?)\s*[元块]", message)
        return float(match.group(1)) if match else None

    def _obs_of(self, tool: str):
        """返回该工具最近一次的原始 observation payload（不解包）。"""
        result = None
        for t, _, obs_text in self.history:
            if t != tool:
                continue
            try:
                payload = json.loads(obs_text)
            except (TypeError, ValueError):
                continue
            result = payload
        return result if isinstance(result, dict) else None

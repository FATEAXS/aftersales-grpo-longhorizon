"""AftersaleSimulator 环境的领域模型。

所有实体都用普通 dataclass + dict 承载，保持核心零重量级依赖；
HTTP 层和客户端只传输 JSON dict。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


ENVIRONMENT_VERSION = "aftersalesimulator-environment-v1.0"


def load_json(path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


@dataclass
class Policy:
    """单个商品类目的售后政策（环境事实，可通过 query_policy 观测）。"""

    category: str
    no_reason_return_days: int          # 无理由退货窗口（自签收起），0 表示不支持
    quality_return_days: int            # 质量问题退货窗口
    quality_exchange_days: int          # 质量问题换货窗口
    size_exchange_days: int             # 尺码/规格问题换货窗口（服装等），0 表示不支持
    price_protection_days: int          # 价保窗口，0 表示不支持
    quality_refund_without_return: bool  # 质量问题是否支持仅退款不退货（如食品）
    refund_on_quality: bool             # 质量问题是否允许退款
    notes: str = ""
    # 质量问题"仅退款不退货"的独立窗口（食品 15 / 美妆 15 / 定制 30 / 母婴 15）
    quality_refund_only_days: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict) -> "Policy":
        return cls(**data)


@dataclass
class Product:
    product_id: str
    name: str
    category: str
    price: float
    options: list[str] = field(default_factory=list)  # 可选规格，如 ["星空黑", "月光银"]

    @classmethod
    def from_dict(cls, data: dict) -> "Product":
        return cls(**data)


@dataclass
class OrderItem:
    product_id: str
    name: str
    category: str
    qty: int
    unit_price: float
    options: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class LogisticsEvent:
    time: str
    status: str
    detail: str

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Order:
    order_id: str
    user_id: str
    items: list[OrderItem]
    total_amount: float
    status: str            # shipped / in_transit / delivered / closed
    created_at: str
    paid_at: str
    shipped_at: str | None
    delivered_at: str | None
    logistics: list[LogisticsEvent] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "Order":
        items = [OrderItem(**item) for item in data["items"]]
        logistics = [LogisticsEvent(**event) for event in data.get("logistics", [])]
        return cls(
            order_id=data["order_id"],
            user_id=data["user_id"],
            items=items,
            total_amount=data["total_amount"],
            status=data["status"],
            created_at=data["created_at"],
            paid_at=data["paid_at"],
            shipped_at=data.get("shipped_at"),
            delivered_at=data.get("delivered_at"),
            logistics=logistics,
        )

    def to_public_dict(self) -> dict:
        """面向模型的订单视图（不含物流事件明细，物流需单独查询）。"""
        return {
            "order_id": self.order_id,
            "user_id": self.user_id,
            "items": [item.__dict__ for item in self.items],
            "total_amount": self.total_amount,
            "status": self.status,
            "created_at": self.created_at,
            "paid_at": self.paid_at,
            "shipped_at": self.shipped_at,
            "delivered_at": self.delivered_at,
        }


@dataclass
class UserProfile:
    """用户档案；退款/投诉次数是风控观测点（可通过 query_user_profile 查询）。"""

    user_id: str
    name: str
    level: str              # normal / vip
    refund_count_90d: int
    complaint_count_90d: int

    @classmethod
    def from_dict(cls, data: dict) -> "UserProfile":
        return cls(**data)

    def to_public_dict(self) -> dict:
        return dict(self.__dict__)


# ---------------------------------------------------------------------------
# 售后工单 / 执行记录：环境对 mutating 动作做政策校验并记录结果。
# ---------------------------------------------------------------------------

REFUND_FULL_TOLERANCE = 0.02  # 全额退款金额允许的浮点误差比例


@dataclass
class AftersaleTicket:
    ticket_id: str
    order_id: str
    kind: str              # refund / return / exchange / compensation
    amount: float
    reason: str
    status: str            # approved / rejected
    reject_reason: str | None = None
    new_option: str | None = None   # exchange 用

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class ActionResult:
    """单次工具执行结果；observation 渲染前先经过这里。"""

    ok: bool
    observation: dict
    error: str | None = None       # guard 拒绝原因，如 invalid_arguments / policy_rejected

    @classmethod
    def reject(cls, error: str, observation: dict) -> "ActionResult":
        return cls(ok=False, observation=observation, error=error)


# ---------------------------------------------------------------------------
# TaskFacts：任务的私有事实（评分依据），绝不进入模型可见数据。
# ---------------------------------------------------------------------------

RESOLUTIONS = (
    "refund_only_full",     # 仅退款（全额）
    "return_refund",        # 退货退款
    "exchange",             # 换货
    "refund_partial",       # 部分退款（差价/补偿性）
    "compensation_only",    # 仅发放补偿券
    "rejected",             # 拒绝售后并解释
    "escalated",            # 转人工
    "unresolved",           # 未解决
)


@dataclass
class TaskFacts:
    task_id: str
    order_id: str
    scenario: str
    difficulty: str                # easy / medium / hard
    expected_resolution: str       # RESOLUTIONS 之一
    acceptable: dict[str, float]   # resolution -> 分数（含 expected_resolution 的 1.0）
    requires_escalation: bool
    ineligible: bool               # 该诉求是否不符合政策（期望 rejected）
    suspicious_user: bool          # 是否命中风控（期望 escalate）
    max_compensation: float        # 政策允许的最大补偿券
    price_drop_amount: float       # 价保差价（价保场景用）
    soft_step_budget: int          # 超过该步数开始效率惩罚
    opening_message: str           # 用户开场白（公开部分，复制进 facts 便于审计）
    requires_clarification: bool = False  # 同名订单歧义：必须追问而非猜测

    @classmethod
    def from_dict(cls, data: dict) -> "TaskFacts":
        data = dict(data)
        data.setdefault("requires_clarification", False)
        return cls(**data)

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    def public_task(self) -> dict:
        """公开任务卡：只含 task_id 与用户开场白。"""
        return {
            "task_id": self.task_id,
            "opening_message": self.opening_message,
            "max_steps": MAX_STEPS_DEFAULT,
        }


MAX_STEPS_DEFAULT = 25
RISK_REFUND_THRESHOLD = 6  # 90 天退款次数达到该值命中风控

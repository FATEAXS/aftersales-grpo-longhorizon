"""默认售后政策表与补偿上限规则。

政策是环境的"法律"：任务生成器据此推导 TaskFacts，会话执行器据此校验
mutating 动作，评分器据此判定政策违规。三处共享同一份数据源，避免口径漂移。
"""

from __future__ import annotations

from .models import Policy


DEFAULT_POLICIES: dict[str, Policy] = {
    "3C数码": Policy(
        category="3C数码",
        no_reason_return_days=7,
        quality_return_days=15,
        quality_exchange_days=15,
        size_exchange_days=0,
        price_protection_days=7,
        quality_refund_without_return=False,
        refund_on_quality=True,
        notes="激活类商品（手机/平板）无理由退货需保持未激活；质量问题需提供照片或检测凭证。",
    ),
    "服装鞋帽": Policy(
        category="服装鞋帽",
        no_reason_return_days=7,
        quality_return_days=15,
        quality_exchange_days=15,
        size_exchange_days=15,
        price_protection_days=0,
        quality_refund_without_return=False,
        refund_on_quality=True,
        notes="无理由退货需不影响二次销售（吊牌完整、未洗涤）；尺码问题优先换货。",
    ),
    "食品保健": Policy(
        category="食品保健",
        no_reason_return_days=0,
        quality_return_days=0,
        quality_exchange_days=0,
        size_exchange_days=0,
        price_protection_days=0,
        quality_refund_only_days=15,
        quality_refund_without_return=True,
        refund_on_quality=True,
        notes="食品不支持无理由退货；质量问题仅退款不退货，可酌情发放补偿券。",
    ),
    "美妆个护": Policy(
        category="美妆个护",
        no_reason_return_days=0,
        quality_return_days=15,
        quality_exchange_days=15,
        size_exchange_days=0,
        price_protection_days=0,
        quality_refund_only_days=15,
        quality_refund_without_return=True,
        refund_on_quality=True,
        notes="拆封后不支持无理由退货；过敏等质量问题仅退款不退货。",
    ),
    "家居日用": Policy(
        category="家居日用",
        no_reason_return_days=7,
        quality_return_days=15,
        quality_exchange_days=15,
        size_exchange_days=0,
        price_protection_days=0,
        quality_refund_without_return=False,
        refund_on_quality=True,
        notes="大件家具质量问题可上门取件；无理由退货需包装完好。",
    ),
    "虚拟商品": Policy(
        category="虚拟商品",
        no_reason_return_days=0,
        quality_return_days=0,
        quality_exchange_days=0,
        size_exchange_days=0,
        price_protection_days=0,
        quality_refund_without_return=False,
        refund_on_quality=False,
        notes="虚拟商品（卡券/会员/充值）一经售出不支持退款，异常问题请转人工核实。",
    ),
    "定制商品": Policy(
        category="定制商品",
        no_reason_return_days=0,
        quality_return_days=30,
        quality_exchange_days=0,
        size_exchange_days=0,
        price_protection_days=0,
        quality_refund_without_return=True,
        refund_on_quality=True,
        notes="定制类商品不支持无理由退换；质量问题（破损/错做）30 天内可退，仅退款不退货。",
    ),
    "母婴用品": Policy(
        category="母婴用品",
        no_reason_return_days=7,
        quality_return_days=15,
        quality_exchange_days=15,
        size_exchange_days=15,
        price_protection_days=0,
        quality_refund_only_days=15,
        quality_refund_without_return=True,
        refund_on_quality=True,
        notes="奶粉等入口类拆封后不支持无理由退货；质量问题仅退款不退货。",
    ),
}


def compensation_cap(order_amount: float, user_level: str) -> float:
    """补偿券上限：普通用户 min(20% 订单金额, 30 元)，VIP min(30%, 50 元)。"""
    if user_level == "vip":
        return round(min(order_amount * 0.30, 50.0), 2)
    return round(min(order_amount * 0.20, 30.0), 2)


def days_between(delivered_at: str, now: str) -> int:
    """两个 'YYYY-MM-DD' 日期的天数差（now - delivered_at）。"""
    from datetime import date

    d1 = date.fromisoformat(delivered_at)
    d2 = date.fromisoformat(now)
    return (d2 - d1).days

"""售后 Agent 的工具契约（OpenAI function-calling schema）与系统提示词。

该文件是环境动作契约的单一事实源：评测 rollout、SFT 教师、veRL tools.json
都从这里派生，避免多处定义漂移。configs/tools.json 中的 veRL 工具 schema 与
TOOLS 保持逐字段一致（由 scripts/check_tools_contract.py 校验）。
"""

from __future__ import annotations

import json
from pathlib import Path

MAX_REPLY_CHARS = 1500
MAX_STEPS = 25
ENVIRONMENT_BASE_URL_DEFAULT = "http://127.0.0.1:5800"


def _schema(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


_ORDER_ID_PROP = {
    "type": "string",
    "description": "订单号，必须来自工具返回的 observation，不得凭记忆或猜测构造。",
}

TOOLS: list[dict] = [
    _schema(
        "query_user_orders",
        "查询当前用户名下全部订单的摘要（订单号、商品、金额、状态、签收日期）。"
        "当用户没有提供订单号、或提供的订单号与所述商品对不上时，先用它定位正确订单。",
        {},
        [],
    ),
    _schema(
        "query_order",
        "查询订单详情：条目、规格、实付金额、支付/发货/签收时间与状态。",
        {"order_id": _ORDER_ID_PROP},
        ["order_id"],
    ),
    _schema(
        "query_logistics",
        "查询订单物流轨迹与当前状态（是否签收、是否滞留）。",
        {"order_id": _ORDER_ID_PROP},
        ["order_id"],
    ),
    _schema(
        "query_user_profile",
        "查询当前用户档案：会员等级、90 天退款次数、90 天投诉次数。退款次数达到 6 次"
        "属于风控信号，此类无理由退款诉求必须转人工。",
        {},
        [],
    ),
    _schema(
        "query_policy",
        "查询指定商品类目的售后政策：无理由退货窗口、质量问题退货/换货窗口、尺码换货"
        "窗口、价保窗口、仅退款规则、补偿券上限。执行任何售后动作前必须先查询对应类目政策。",
        {"category": {"type": "string", "description": "商品类目，如 3C数码、服装鞋帽。"}},
        ["category"],
    ),
    _schema(
        "apply_refund",
        "发起退款（不退货）。refund_type=full 为全额退款；refund_type=partial 用于价保"
        "差价等部分退款，需给 amount。虚拟商品不支持。",
        {
            "order_id": _ORDER_ID_PROP,
            "refund_type": {"type": "string", "enum": ["full", "partial"]},
            "amount": {"type": "number", "description": "partial 退款金额（元）。"},
            "reason": {"type": "string"},
        },
        ["order_id", "refund_type", "reason"],
    ),
    _schema(
        "apply_return",
        "发起退货退款。需先确认订单在政策允许的退货窗口内且理由成立。",
        {"order_id": _ORDER_ID_PROP, "reason": {"type": "string"}},
        ["order_id", "reason"],
    ),
    _schema(
        "apply_exchange",
        "发起换货，需指定 new_option（必须取自该商品的可选规格）。定制商品与虚拟商品不支持。",
        {
            "order_id": _ORDER_ID_PROP,
            "new_option": {"type": "string"},
            "reason": {"type": "string"},
        },
        ["order_id", "new_option", "reason"],
    ),
    _schema(
        "issue_compensation",
        "发放补偿券。金额不得超过当前用户的政策上限（query_policy 返回中会给出）。",
        {
            "order_id": _ORDER_ID_PROP,
            "amount": {"type": "number"},
            "reason": {"type": "string"},
        },
        ["order_id", "amount", "reason"],
    ),
    _schema(
        "escalate_to_human",
        "转接人工客服并结束会话。仅用于：用户 90 天退款次数 >= 6 的风控诉求、"
        "政策未覆盖的争议。可正常解决的诉求转人工会被判为过早升级。",
        {"order_id": _ORDER_ID_PROP, "reason": {"type": "string"}},
        ["order_id", "reason"],
    ),
    _schema(
        "reply_user",
        "向用户发送一条说明（政策解释、处理结果、安抚）。每次决策后都应告知用户；"
        "拒绝售后时必须先解释对应政策条款。",
        {"message": {"type": "string", "maxLength": MAX_REPLY_CHARS}},
        ["message"],
    ),
    _schema(
        "finish",
        "结束会话并申报最终解决方式，resolution 必须与实际执行过的动作一致。"
        "summary 用一句话说明处理结果。",
        {
            "resolution": {
                "type": "string",
                "enum": [
                    "refund_only_full",
                    "return_refund",
                    "exchange",
                    "refund_partial",
                    "compensation_only",
                    "rejected",
                    "escalated",
                    "unresolved",
                ],
            },
            "summary": {"type": "string"},
        },
        ["resolution", "summary"],
    ),
]

TOOL_NAMES = [tool["function"]["name"] for tool in TOOLS]

SYSTEM_PROMPT = f"""你是电商平台"安心购"的资深售后客服。请通过工具调用完成售后处理，规范如下：

1. 先取证，后行动：执行任何售后动作前，必须用 query_order 查看订单，并用 query_policy 查看对应类目政策；物流/未收到货问题先 query_logistics。
2. 核对订单：用户提供的订单号可能与所述商品不符，注意核对商品名称与规格；必要时用 query_user_orders 定位正确订单。
3. 风控红线：query_user_profile 显示 90 天退款次数 >= 6 时，无理由退款诉求必须 escalate_to_human，不得自行退款。
4. 政策红线：无理由退货以签收日起算且窗口内才可办理；食品/美妆/定制/虚拟等类目有硬性限制（以 query_policy 返回为准）；补偿券不得超过当前用户上限。不符合政策的诉求应礼貌解释政策并如实 finish(rejected)，不得为了安抚用户违规办理。
5. 质量问题在窗口内应主动办理退货退款或换货；食品、美妆、定制类质量问题仅退款不退货。
6. 沟通闭环：每次动作后用 reply_user 向用户说明；结束时用 finish 申报与实际动作一致的 resolution。
7. 效率：最多 {MAX_STEPS} 步，不要重复查询相同信息，不要做无意义的动作。

工具共 {len(TOOLS)} 个：{", ".join(TOOL_NAMES)}。"""


def initial_messages(opening: dict) -> list[dict]:
    """构造会话初始 messages：system（SOP + 当天日期） + user（开场白）。"""
    today = opening.get("today", "")
    user_id = opening.get("user_id", "")
    system = SYSTEM_PROMPT
    if today or user_id:
        system = f"今天是 {today}。当前服务用户：{user_id}。\n\n" + system
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": opening.get("user_message", "")},
    ]


def tool_schemas_path() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "tools.json"


def dump_tools_json(path: Path) -> None:
    """把工具契约写成 veRL tools.json 兼容格式。"""
    payload = {
        "tools": [
            {
                "class_name": "aftersales_grpo.training.grpo.adapter.tools.AftersaleTool",
                "config": {"type": "native"},
                "tool_schema": schema,
            }
            for schema in TOOLS
        ]
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

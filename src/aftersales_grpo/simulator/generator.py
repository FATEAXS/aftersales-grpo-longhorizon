"""AftersaleSimulator 世界与任务生成器（确定性，固定种子）。

一次生成得到：
- 世界数据：products / users / orders / policies（含 manifest，SHA-256 可审计）
- 任务数据：train / validation / evaluation 三个 split 的 TaskFacts（私有），
  以及对应的公开任务卡（task_id + 用户开场白）

十类场景覆盖现实售后客服的主要决策面：
可解决（退货退款 / 仅退款 / 换货 / 物流破损 / 未收到货 / 价保 / 错发）、
需拒绝（类目政策硬限制 / 超期诉求）、需转人工（风控用户）。
干扰项：不给订单号、同名多订单、口误报错订单号。

所有评分口径（expected_resolution / acceptable / ineligible ...）在生成时随
TaskFacts 冻结，环境会话与评分器共享同一份事实。
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import date, timedelta
from pathlib import Path

from .models import (
    LogisticsEvent,
    Order,
    OrderItem,
    Policy,
    Product,
    TaskFacts,
    UserProfile,
)
from .policies import DEFAULT_POLICIES, compensation_cap

WORLD_SEED = 20260910
TASK_SEED = 20260911
TODAY = date(2026, 9, 8)

SPLIT_SIZES = {"train": 400, "validation": 100, "evaluation": 100}

# 场景权重（总和 100）：64% 可解决 / 24% 应拒绝 / 12% 应转人工。
SCENARIO_WEIGHTS = {
    "defective_return": 12,
    "defective_refund_only": 10,
    "size_exchange": 8,
    "damaged_logistics": 10,
    "not_delivered": 8,
    "price_protection": 8,
    "wrong_item": 8,
    "blocked_no_reason": 12,
    "late_complaint": 12,
    "suspicious_vip": 12,
}

ACCEPTABLE_MAPS = {
    "defective_return": {"return_refund": 1.0, "exchange": 0.85, "refund_only_full": 0.7},
    "defective_refund_only": {"refund_only_full": 1.0, "return_refund": 0.9, "compensation_only": 0.5},
    "size_exchange": {"exchange": 1.0, "return_refund": 0.85},
    "damaged_logistics": {"return_refund": 1.0, "exchange": 0.85, "refund_only_full": 0.7},
    "not_delivered": {"refund_only_full": 1.0, "escalated": 0.6, "compensation_only": 0.5},
    "price_protection": {"refund_partial": 1.0, "compensation_only": 0.4},
    "wrong_item": {"exchange": 1.0, "return_refund": 1.0, "refund_only_full": 0.8},
    "blocked_no_reason": {"rejected": 1.0},
    "late_complaint": {"rejected": 1.0},
    "suspicious_vip": {"escalated": 1.0},
}

SCENARIO_BASE_DIFFICULTY = {
    "defective_return": "medium",
    "defective_refund_only": "medium",
    "size_exchange": "easy",
    "damaged_logistics": "medium",
    "not_delivered": "medium",
    "price_protection": "hard",
    "wrong_item": "medium",
    "blocked_no_reason": "medium",
    "late_complaint": "hard",
    "suspicious_vip": "hard",
}

# 场景 -> 可用类目（None 表示全部）
SCENARIO_CATEGORIES = {
    "defective_return": ["3C数码", "服装鞋帽", "家居日用"],
    "defective_refund_only": ["食品保健", "美妆个护", "定制商品", "母婴用品"],
    "size_exchange": ["服装鞋帽"],
    "damaged_logistics": ["3C数码", "家居日用", "母婴用品", "服装鞋帽"],
    "not_delivered": ["3C数码", "服装鞋帽", "食品保健", "美妆个护", "家居日用", "母婴用品"],
    "price_protection": ["3C数码"],
    "wrong_item": ["服装鞋帽", "3C数码", "家居日用"],
    "blocked_no_reason": ["食品保健", "美妆个护", "虚拟商品", "定制商品"],
    "late_complaint": ["3C数码", "服装鞋帽", "家居日用", "母婴用品"],
    "suspicious_vip": ["3C数码", "服装鞋帽", "家居日用", "美妆个护"],
}

DELIVERED_DAYS = {
    "defective_return": (1, 12),
    "defective_refund_only": (1, 10),
    "size_exchange": (1, 10),
    "damaged_logistics": (1, 12),
    "not_delivered": ["3C数码", "服装鞋帽", "食品保健", "美妆个护", "家居日用", "母婴用品"],           # 未签收，单独处理
    "price_protection": (1, 6),
    "wrong_item": (1, 10),
    "blocked_no_reason": (1, 30),
    "late_complaint": (20, 45),
    "suspicious_vip": (1, 15),
}

DEFECTS = {
    "3C数码": ["右耳没有声音", "无法开机", "充电到 80% 就断", "屏幕出现亮点"],
    "服装鞋帽": ["起球起得厉害", "洗了一次就开线", "有明显的异味", "掉色掉得严重"],
    "家居日用": ["用到第三天就裂了", "支架是松的", "涂层有脱落", "开关失灵"],
    "母婴用品": ["有刺鼻气味", "缝线开线", "刻度看不清", "用了一周就漏水"],
    "食品保健": ["吃到头发", "包装袋是漏气的", "发霉了", "味道不对像变质"],
    "美妆个护": ["涂上就过敏", "质地分层了", "是过期的", "瓶口漏液"],
    "定制商品": ["刻字刻错了", "和设计稿颜色完全不同", "收到的有明显破损"],
}

PRODUCT_BANK = {
    "3C数码": [
        ("星芒无线蓝牙耳机", (129, 599), ["星空白", "薄暮紫", "碳纤黑"]),
        ("BX 智能运动手表", (299, 1299), ["曜石黑", "流沙金"]),
        ("疾速快充充电宝 2 万毫安", (99, 249), ["深海蓝", "磨砂黑"]),
        ("岚声降噪头戴耳机", (399, 1599), ["雅灰", "雾松绿"]),
        ("光轴机械键盘 87 键", (199, 799), ["白光版", "RGB 版"]),
        ("迷你便携蓝牙音箱", (79, 399), ["暮云灰", "活力橙"]),
        ("4K 高清运动相机", (599, 2199), ["标准套装", "骑行套装"]),
        ("智能体重体脂秤", (59, 199), ["经典黑", "珍珠白"]),
    ],
    "服装鞋帽": [
        ("基础款连帽卫衣", (89, 299), ["M", "L", "XL", "XXL"]),
        ("复古水洗牛仔外套", (159, 459), ["S", "M", "L"]),
        ("轻弹缓震运动跑鞋", (199, 699), ["40 码", "41 码", "42 码", "43 码"]),
        ("双面呢羊毛大衣", (499, 1299), ["M", "L", "XL"]),
        ("冰丝防晒衬衫", (79, 259), ["M", "L", "XL"]),
        ("户外防风软壳裤", (129, 399), ["30", "32", "34"]),
    ],
    "食品保健": [
        ("每日坚果混合装 30 包", (59, 159), ["标准装", "礼盒装"]),
        ("益生菌固体饮料", (89, 299), ["30 条装"]),
        ("意式黑巧克力礼盒", (49, 199), ["经典款", "珍藏款"]),
        ("冻干草莓脆 3 袋装", (29, 89), ["原味"]),
        ("胶原蛋白肽饮 30 支", (129, 399), ["蜜桃味"]),
    ],
    "美妆个护": [
        ("玻尿酸保湿面霜", (129, 399), ["50g"]),
        ("清爽控油防晒乳", (89, 259), ["60ml"]),
        ("氨基酸修护洗发水", (69, 199), ["400ml", "750ml"]),
        ("烟酰胺身体乳", (59, 179), ["250ml"]),
        ("电动声波牙刷", (129, 499), ["樱粉", "石墨黑"]),
    ],
    "家居日用": [
        ("手作陶瓷马克杯", (39, 129), ["月白", "黛青"]),
        ("记忆棉慢回弹枕头", (99, 299), ["标准款", "加高款"]),
        ("折叠收纳箱三件套", (49, 199), ["象牙白", "岩灰"]),
        ("北欧实木边几", (199, 899), ["原木色", "胡桃色"]),
        ("加厚珊瑚绒浴巾", (39, 129), ["云雾灰", "奶油白"]),
        ("自动感应皂液器", (59, 199), ["白色", "香槟金"]),
    ],
    "虚拟商品": [
        ("视频平台黄金会员季卡", (45, 98), ["季卡"]),
        ("游戏点充值卡 500 点", (45, 55), ["电子卡"]),
        ("电子购物卡 200 元面值", (190, 210), ["电子卡"]),
        ("音乐 App 豪华会员年卡", (108, 168), ["年卡"]),
        ("网盘超级会员月卡", (25, 35), ["月卡"]),
    ],
    "定制商品": [
        ("定制照片抱枕", (49, 129), ["单面印", "双面印"]),
        ("刻字实木情侣木梳", (59, 159), ["礼盒装", "简装"]),
        ("定制乳胶枕（刻名）", (129, 299), ["标准款"]),
        ("定制生肖马克杯", (49, 139), ["白瓷", "青瓷"]),
        ("定制拼图 1000 片", (69, 199), ["木框装", "简装"]),
    ],
    "母婴用品": [
        ("婴儿超薄纸尿裤 L 码", (99, 249), ["L 码 68 片"]),
        ("儿童保温学饮杯", (69, 199), ["320ml 蓝色", "320ml 粉色"]),
        ("轻便高景观婴儿推车", (899, 1999), ["雾灰", "米白"]),
        ("婴儿纯棉连体衣", (49, 159), ["59cm", "66cm", "73cm"]),
        ("宝宝辅食研磨碗套装", (59, 169), ["薄荷绿", "奶油黄"]),
    ],
}

# 质量问题"仅退款不退货"的类目（与政策表一致）
REFUND_WITHOUT_RETURN_CATEGORIES = {"食品保健", "美妆个护", "定制商品", "母婴用品"}

SURNAMES = ["王", "李", "张", "刘", "陈", "杨", "黄", "赵", "周", "吴", "徐", "孙", "林", "何", "郭"]
GIVEN = ["晓", "雨", "浩", "静", "婷", "杰", "鑫", "悦", "辰", "昊", "雪", "琪", "彤", "峰", "楠"]


def _pick(rng: random.Random, seq):
    return rng.choice(seq)


# ---------------------------------------------------------------------------
# 世界构建
# ---------------------------------------------------------------------------

def build_products(rng: random.Random) -> list[Product]:
    products: list[Product] = []
    idx = 1
    for category, bank in PRODUCT_BANK.items():
        for name, (lo, hi), options in bank:
            price = round(rng.uniform(lo, hi), 2)
            products.append(
                Product(
                    product_id=f"P{idx:04d}",
                    name=name,
                    category=category,
                    price=price,
                    options=list(options),
                )
            )
            idx += 1
    return products


def _logistics_events(rng: random.Random, created: date, delivered: date | None, status: str) -> list[LogisticsEvent]:
    # closed 视为"已签收后关闭"，物流链按 delivered 渲染。
    events = [
        LogisticsEvent(created.isoformat() + " 09:12", "已下单", "订单支付成功，等待发货"),
    ]
    shipped = created + timedelta(days=rng.randint(0, 2))
    if status in {"shipped", "in_transit", "delivered"}:
        events.append(
            LogisticsEvent(shipped.isoformat() + " 18:40", "已发货", "包裹已从杭州转运中心发出")
        )
    if status in {"in_transit", "delivered"}:
        transit = shipped + timedelta(days=1)
        events.append(
            LogisticsEvent(transit.isoformat() + " 07:05", "运输中", "包裹到达【上海转运中心】")
        )
    if status in {"delivered", "closed"} and delivered is not None:
        events.append(
            LogisticsEvent(delivered.isoformat() + " 14:26", "已签收", "包裹已送达，签收人：本人")
        )
    if status == "in_transit":
        stall = shipped + timedelta(days=rng.randint(3, 8))
        events.append(
            LogisticsEvent(stall.isoformat() + " 16:00", "滞留", "包裹在中转场滞留，异常上报处理中")
        )
    return events


def build_order(
    rng: random.Random,
    order_id: str,
    user_id: str,
    product: Product,
    status: str,
    qty: int = 1,
    option: str | None = None,
    wrong_option: bool = False,
    delivered_at: str | None = None,
    created_at: str | None = None,
) -> Order:
    option = option or _pick(rng, product.options)
    if wrong_option and len(product.options) > 1:
        others = [o for o in product.options if o != option]
        option = _pick(rng, others)
    created = (
        date.fromisoformat(created_at)
        if created_at
        else TODAY - timedelta(days=rng.randint(10, 60))
    )
    delivered = None
    if delivered_at:
        delivered = date.fromisoformat(delivered_at)
    elif status in {"delivered", "closed"}:
        delivered = created + timedelta(days=rng.randint(2, 5))
        if delivered > TODAY:
            delivered = TODAY - timedelta(days=1)
    item = OrderItem(
        product_id=product.product_id,
        name=product.name,
        category=product.category,
        qty=qty,
        unit_price=product.price,
        options=option,
    )
    total = round(product.price * qty, 2)
    return Order(
        order_id=order_id,
        user_id=user_id,
        items=[item],
        total_amount=total,
        status=status,
        created_at=created.isoformat(),
        paid_at=(created + timedelta(hours=rng.randint(1, 5))).isoformat(),
        shipped_at=(created + timedelta(days=rng.randint(0, 2))).isoformat()
        if status != "shipped" or True
        else None,
        delivered_at=delivered.isoformat() if delivered else None,
        logistics=_logistics_events(rng, created, delivered, status),
    )


# ---------------------------------------------------------------------------
# 开场白模板
# ---------------------------------------------------------------------------

def _opening_message(rng: random.Random, scenario: str, ctx: dict) -> str:
    oid = ctx.get("order_id", "")
    name = ctx["product_name"]
    days = ctx.get("delivered_days", 3)
    defect = ctx.get("defect", "坏了")
    option = ctx.get("option", "")
    amount = ctx.get("price_drop", 0)

    if scenario == "defective_return":
        variants = [
            f"你们卖的东西质量也太差了吧！我 {days} 天前买的{name}（订单号 {oid}），用着用着就{defect}，这就是你们的质量？给我退货退款，不然投诉到底！",
            f"你好，我在咱家买的{name}，订单 {oid}，才用了 {days} 天就{defect}。这明显是质量问题吧？帮我办一下退货。",
        ]
    elif scenario == "defective_refund_only":
        variants = [
            f"太恶心了，{days} 前买的{name}（{oid}）居然{defect}！这种东西我还退回去干嘛，直接给我退款！",
            f"客服你好，我买的{name}，订单号 {oid}，打开发现{defect}，食品/用品类的问题商品我不方便寄回，你们直接退款就行了吧？",
        ]
    elif scenario == "size_exchange":
        variants = [
            f"你好，我 {days} 天前买的{name}（订单 {oid}），{option} 码穿着不合适，能帮我换个合适的尺码吗？",
            f"买小了买小了！订单 {oid} 的{name}，{option} 码我穿不上，麻烦帮我换大一码，谢谢。",
        ]
    elif scenario == "damaged_logistics":
        variants = [
            f"气死了，快递给我摔坏了！{name}，订单 {oid}，{days} 天前签收的时候外包装都瘪了，打开一看里面{defect}。你们快递的问题必须给我退！",
            f"你好，我收到的{name}（订单 {oid}）外箱破损，里面的商品{defect}，应该是运输损坏，帮我处理下退货吧。",
        ]
    elif scenario == "not_delivered":
        variants = [
            f"我的{name}呢？！订单 {oid} 都下单快两周了还在运输中，物流好几天不动了，我等不了了，退款吧！",
            f"客服你好，订单 {oid} 的{name}一直显示滞留，我下周就要用了，等不到发货了，帮我退款，我重新买。",
        ]
    elif scenario == "price_protection":
        variants = [
            f"你好，我 {days} 天前买的{name}（订单 {oid}），今天就降价 {amount} 元了！不是有价保吗，把差价退我。",
            f"这才买 {days} 天，{name}（订单 {oid}）就降价 {amount} 块，价保期内给我补个差价呗。",
        ]
    elif scenario == "wrong_item":
        variants = [
            f"你们发错货了吧！我订单 {oid} 买的{name}要的{ctx.get('wanted_option', '另一个规格')}，结果发成了{option}，要么给我换，要么退了！",
            f"你好，订单 {oid} 的{name}，我选的规格和收到的不一样，发成{option}了，帮我换货吧。",
        ]
    elif scenario == "blocked_no_reason":
        variants = [
            f"我就是不想要这个{name}了（订单 {oid}），没拆没用，七天无理由给我退了吧！",
            f"你好，{days} 天前买的{name}，订单 {oid}，现在不想要了，帮我办理退货退款，谢谢。",
        ]
    elif scenario == "late_complaint":
        variants = [
            f"我 {days} 天前在你家买的{name}（订单 {oid}），现在{defect}，虽然过了挺久，但这肯定是质量问题，你们必须给我退！",
            f"你好，订单 {oid} 的{name}最近{defect}，买了都 {days} 天了还能找你们售后吗？给我退货吧。",
        ]
    elif scenario == "suspicious_vip":
        variants = [
            f"我买的{name}（订单 {oid}）不想要了，直接给我退款就行，不用问那么多，快点。",
            f"订单 {oid} 的{name}给我退款，理由就是我不想要了，赶紧的，我赶时间。",
        ]
    else:
        variants = [f"你好，我的订单 {oid} 有问题，帮我处理一下。"]
    return _pick(rng, variants)


# ---------------------------------------------------------------------------
# 任务构建
# ---------------------------------------------------------------------------


def generate_split(rng: random.Random, split: str, count: int, start_idx: int,
                   products: list[Product]) -> list[dict]:
    """生成一个 split 的任务（TaskFacts dict 列表）。"""
    products_by_cat: dict[str, list[Product]] = {}
    for p in products:
        products_by_cat.setdefault(p.category, []).append(p)

    scenarios: list[str] = []
    for scenario, weight in SCENARIO_WEIGHTS.items():
        scenarios.extend([scenario] * weight)

    tasks: list[dict] = []
    user_counter = start_idx
    order_counter = start_idx * 10

    for i in range(count):
        scenario = _pick(rng, scenarios)
        categories = SCENARIO_CATEGORIES[scenario] or [
            c for c in products_by_cat if c != "虚拟商品"
        ]
        product = _pick(rng, products_by_cat[_pick(rng, categories)])

        # 用户档案
        suspicious = scenario == "suspicious_vip"
        level = "vip" if (suspicious and rng.random() < 0.7) or rng.random() < 0.15 else "normal"
        user = UserProfile(
            user_id=f"U{user_counter:05d}",
            name=_pick(rng, SURNAMES) + _pick(rng, GIVEN),
            level=level,
            refund_count_90d=rng.randint(6, 14) if suspicious else rng.randint(0, 3),
            complaint_count_90d=rng.randint(1, 3) if suspicious else rng.randint(0, 2),
        )
        user_counter += 1

        # 干扰项；twin_order 额外制造"同名双订单"歧义，触发澄清回合机制
        distractor = rng.choices(
            ["none", "missing_order_id", "multi_order", "cite_wrong_order", "twin_order"],
            weights=[42, 18, 12, 15, 13],
        )[0]

        # 时间线
        if scenario == "not_delivered":
            delivered_days = None
        else:
            lo, hi = DELIVERED_DAYS[scenario]
            delivered_days = rng.randint(lo, hi)

        delivered_at = (
            (TODAY - timedelta(days=delivered_days)).isoformat() if delivered_days else None
        )

        primary_status = "in_transit" if scenario == "not_delivered" else "delivered"
        wanted_option = _pick(rng, product.options)
        wrong_sent = _pick(rng, [o for o in product.options if o != wanted_option]) if scenario == "wrong_item" else None

        primary_delivered = (
            (TODAY - timedelta(days=delivered_days)).isoformat() if delivered_days else None
        )
        primary_created = None
        if scenario == "not_delivered":
            primary_created = (TODAY - timedelta(days=rng.randint(10, 16))).isoformat()
        elif delivered_days is not None:
            primary_created = (
                TODAY - timedelta(days=delivered_days + rng.randint(2, 5))
            ).isoformat()
        primary = build_order(
            rng,
            order_id=f"O{order_counter:06d}",
            user_id=user.user_id,
            product=product,
            status=primary_status,
            option=wrong_sent or wanted_option,
            delivered_at=primary_delivered,
            created_at=primary_created,
        )
        order_counter += 1

        # 同用户的干扰订单
        user_orders = [primary]
        n_extra = rng.randint(1, 3)
        for j in range(n_extra):
            if distractor == "twin_order" and j == 0:
                # 同商品双订单：两笔都在进行中，开场白不含订单号 -> 必须澄清
                twin = build_order(
                    rng,
                    order_id=f"O{order_counter:06d}",
                    user_id=user.user_id,
                    product=product,
                    status="delivered",
                    delivered_at=(TODAY - timedelta(days=rng.randint(3, 40))).isoformat(),
                )
                order_counter += 1
                user_orders.append(twin)
                continue
            other = _pick(rng, products)
            extra = build_order(
                rng,
                order_id=f"O{order_counter:06d}",
                user_id=user.user_id,
                product=other,
                status=_pick(rng, ["delivered", "closed"]),
            )
            order_counter += 1
            user_orders.append(extra)

        # 价保差价
        price_drop = round(primary.total_amount * rng.uniform(0.05, 0.2), 2) if scenario == "price_protection" else 0.0

        defect = _pick(rng, DEFECTS.get(product.category, ["出现问题"])) if product.category in DEFECTS else "出现质量问题"

        # 引用的订单号：正常/口误报错/不给
        if distractor == "cite_wrong_order":
            cited = _pick(rng, [o for o in user_orders if o.order_id != primary.order_id]).order_id
        elif distractor in {"missing_order_id", "twin_order"}:
            cited = ""
        else:
            cited = primary.order_id

        ctx = {
            "order_id": cited,
            "product_name": product.name,
            "delivered_days": delivered_days or rng.randint(7, 14),
            "defect": defect,
            "option": (wrong_sent or wanted_option),
            "wanted_option": wanted_option,
            "price_drop": price_drop,
        }

        difficulty = SCENARIO_BASE_DIFFICULTY[scenario]
        if distractor in {"cite_wrong_order", "twin_order"} or (distractor == "multi_order" and rng.random() < 0.5):
            difficulty = "hard"
        elif distractor == "missing_order_id" and difficulty == "easy":
            difficulty = "medium"

        expected_resolution = {
            "defective_return": "return_refund",
            "damaged_logistics": "return_refund",
            "wrong_item": "exchange",
            "size_exchange": "exchange",
            "defective_refund_only": "refund_only_full",
            "not_delivered": "refund_only_full",
            "price_protection": "refund_partial",
            "blocked_no_reason": "rejected",
            "late_complaint": "rejected",
            "suspicious_vip": "escalated",
        }[scenario]

        opening = _opening_message(rng, scenario, ctx)
        opening = (
            opening.replace("（订单号 ）", "")
            .replace("（订单 ）", "")
            .replace("订单 ）", "）")
            .replace("订单 ，", "，").replace("，，", "，")
        )
        import re as _re
        opening = _re.sub(r"订单\s+([，。！？])", r"", opening)
        opening = _re.sub(r"\s{2,}", " ", opening)
        acceptable = dict(ACCEPTABLE_MAPS[scenario])
        expected_resolution = expected_resolution
        if scenario in {"damaged_logistics", "defective_return"} and product.category in REFUND_WITHOUT_RETURN_CATEGORIES:
            acceptable = {"refund_only_full": 1.0, "return_refund": 0.9, "exchange": 0.85}
            expected_resolution = "refund_only_full"

        facts = {
            "task_id": f"afts-{split}-{i:04d}",
            "order_id": primary.order_id,
            "scenario": scenario,
            "difficulty": difficulty,
            "expected_resolution": expected_resolution,
            "acceptable": acceptable,
            "requires_escalation": scenario == "suspicious_vip",
            "ineligible": scenario in {"blocked_no_reason", "late_complaint"},
            "suspicious_user": suspicious,
            "max_compensation": compensation_cap(primary.total_amount, level),
            "price_drop_amount": price_drop,
            "soft_step_budget": {"easy": 8, "medium": 10, "hard": 12}[difficulty],
            "opening_message": opening,
            "requires_clarification": distractor == "twin_order",
        }
        tasks.append(
            {
                "facts": facts,
                "user": user.__dict__.copy(),
                "user_orders": user_orders,
                "primary_order_id": primary.order_id,
            }
        )
    return tasks




# ---------------------------------------------------------------------------
# SFT 拒绝类加强任务(上游对齐 GRPO v2 的合规重采样)
# ---------------------------------------------------------------------------

AUGMENT_WEIGHTS = {
    "blocked_no_reason": 30,
    "late_complaint": 30,
    "suspicious_vip": 25,
    "defective_return": 5,
    "size_exchange": 5,
    "wrong_item": 5,
}


def generate_sft_augment(count: int, out_env_dir: Path, out_data_dir: Path,
                         seed: int = 20260912) -> dict:
    """生成拒绝/风控占 85% 的 SFT 加强任务(split=sft_augment,task_id 独立)。"""
    global SCENARIO_WEIGHTS
    original = dict(SCENARIO_WEIGHTS)
    SCENARIO_WEIGHTS.clear()
    SCENARIO_WEIGHTS.update(AUGMENT_WEIGHTS)
    try:
        rng = random.Random(seed)
        products = build_products(random.Random(WORLD_SEED))
        raw = generate_split(rng, "sft_augment", count, 9000, products)

        tasks_dir = out_env_dir / "tasks"
        tasks_dir.mkdir(parents=True, exist_ok=True)
        with open(tasks_dir / "sft_augment.jsonl", "w", encoding="utf-8") as fh:
            for record in raw:
                fh.write(json.dumps(record["facts"], ensure_ascii=False) + "\n")

        pub_dir = out_data_dir / "sft_augment"
        pub_dir.mkdir(parents=True, exist_ok=True)
        with open(pub_dir / "tasks.jsonl", "w", encoding="utf-8") as fh:
            for record in raw:
                fact = TaskFacts.from_dict(record["facts"])
                fh.write(json.dumps(fact.public_task(), ensure_ascii=False) + "\n")

        # 把加强任务的用户/订单并入世界数据(环境服务要能解析这些订单)
        users_path = out_env_dir / "users.json"
        orders_path = out_env_dir / "orders.json"
        users = {u["user_id"]: u for u in json.loads(users_path.read_text(encoding="utf-8"))}
        orders = json.loads(orders_path.read_text(encoding="utf-8"))
        known_order_ids = {o["order_id"] for o in orders}
        for record in raw:
            user = UserProfile.from_dict(record["user"])
            users.setdefault(user.user_id, user.__dict__)
            for order in record["user_orders"]:
                if order.order_id not in known_order_ids:
                    known_order_ids.add(order.order_id)
                    orders.append(order)
        users_path.write_text(
            json.dumps(list(users.values()), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        orders_payload = []
        for o in orders:
            if isinstance(o, dict):
                orders_payload.append(o)  # 既有数据已是序列化格式
                continue
            orders_payload.append(
                {
                    **{k: v for k, v in o.__dict__.items() if k not in {"items", "logistics"}},
                    "items": [item.__dict__ for item in o.items],
                    "logistics": [event.__dict__ for event in o.logistics],
                }
            )
        orders_path.write_text(
            json.dumps(orders_payload, ensure_ascii=False, indent=1), encoding="utf-8"
        )

        from collections import Counter
        stats = {
            "tasks": count,
            "scenario_counts": dict(Counter(r["facts"]["scenario"] for r in raw)),
        }
    finally:
        SCENARIO_WEIGHTS.clear()
        SCENARIO_WEIGHTS.update(original)
    return stats


def generate_everything(out_env_dir: Path, out_data_dir: Path) -> dict:
    """生成世界 + 三个 split，落盘并返回统计。"""
    rng_world = random.Random(WORLD_SEED)
    products = build_products(rng_world)
    product_index = {p.product_id: p.__dict__ for p in products}

    out_env_dir.mkdir(parents=True, exist_ok=True)
    (out_env_dir / "policies.json").write_text(
        json.dumps({k: v.to_dict() for k, v in DEFAULT_POLICIES.items()}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    (out_env_dir / "products.json").write_text(
        json.dumps([p.__dict__ for p in products], ensure_ascii=False, indent=1), encoding="utf-8"
    )

    all_facts: list[TaskFacts] = []
    all_users: dict[str, UserProfile] = {}
    all_orders: dict[str, Order] = {}
    split_stats: dict[str, dict] = {}

    task_rng = random.Random(TASK_SEED)
    start = 0
    for split, size in SPLIT_SIZES.items():
        raw = generate_split(task_rng, split, size, start, products)
        start += size
        facts_path = out_env_dir / "tasks"
        facts_path.mkdir(exist_ok=True)
        with open(facts_path / f"{split}.jsonl", "w", encoding="utf-8") as fh:
            for record in raw:
                fh.write(json.dumps(record["facts"], ensure_ascii=False) + "\n")

        # 公开任务卡
        pub_dir = out_data_dir / ("evaluation" if split == "evaluation" else "grpo")
        pub_dir.mkdir(parents=True, exist_ok=True)
        name = "tasks" if split == "evaluation" else ("train" if split == "train" else "validation")
        with open(pub_dir / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for record in raw:
                facts = TaskFacts.from_dict(record["facts"])
                fh.write(json.dumps(facts.public_task(), ensure_ascii=False) + "\n")

        for record in raw:
            user = UserProfile.from_dict(record["user"])
            all_users[user.user_id] = user
            for order in record["user_orders"]:
                all_orders[order.order_id] = order
            all_facts.append(TaskFacts.from_dict(record["facts"]))

        from collections import Counter

        split_stats[split] = {
            "tasks": size,
            "scenario_counts": dict(Counter(r["facts"]["scenario"] for r in raw)),
            "difficulty_counts": dict(Counter(r["facts"]["difficulty"] for r in raw)),
        }

    users_payload = [u.__dict__ for u in all_users.values()]
    orders_payload = []
    for o in all_orders.values():
        orders_payload.append(
            {
                **{k: v for k, v in o.__dict__.items() if k not in {"items", "logistics"}},
                "items": [item.__dict__ for item in o.items],
                "logistics": [event.__dict__ for event in o.logistics],
            }
        )
    (out_env_dir / "users.json").write_text(
        json.dumps(users_payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    (out_env_dir / "orders.json").write_text(
        json.dumps(orders_payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    manifest = {
        "world_seed": WORLD_SEED,
        "task_seed": TASK_SEED,
        "today": TODAY.isoformat(),
        "counts": {
            "products": len(products),
            "users": len(all_users),
            "orders": len(all_orders),
            "tasks": len(all_facts),
        },
        "splits": split_stats,
        "sha256": {
            name: hashlib.sha256((out_env_dir / name).read_bytes()).hexdigest()
            for name in ["policies.json", "products.json", "users.json", "orders.json"]
        },
    }
    (out_env_dir / "world_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    import sys

    root = Path(__file__).resolve().parents[3]
    manifest = generate_everything(
        root / "environments" / "aftersalesim" / "data", root / "data"
    )
    print(json.dumps(manifest["counts"], ensure_ascii=False))

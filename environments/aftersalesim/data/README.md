# AftersaleSimulator 环境数据（生成产物，勿手改）

本目录存放由 `scripts/build_world.py` 以固定随机种子生成的环境世界数据：

| 文件 | 内容 |
|---|---|
| `policies.json` | 8 个商品类目的售后政策（退货/换货窗口、质量问题规则、补偿上限） |
| `products.json` | 140 个商品（类目、价格、可选规格） |
| `users.json` | 320 个用户（等级、90 天退款/投诉次数） |
| `orders.json` | 620 个订单（条目、金额、支付/发货/签收时间线、物流事件） |

`world_manifest.json` 记录生成种子与文件 SHA-256，保证任何人重建都得到逐字节一致的世界。

环境服务端（`python -m aftersales_grpo.simulator.server`）在启动时加载本目录数据，
任务评分所需的 TaskFacts 由环境在会话内私有持有，不会出现在任何面向模型的数据里。

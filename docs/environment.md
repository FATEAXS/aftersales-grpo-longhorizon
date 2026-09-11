# AftersaleSimulator v1.0 环境设计

## 运行形态

单进程 FastAPI 内存服务（`python -m aftersales_grpo.simulator.server`），
所有训练/评测入口通过统一 REST 契约访问：

- `GET /health` → 状态 + environment_version + world_fingerprint（世界版本防错位）
- `GET /v1/tasks/{task_id}` → 公开任务卡（开场白、user_id、today）
- `POST /v1/sessions` → 创建会话（task_id）
- `POST /v1/sessions/{sid}/action` → 执行一次工具调用
- `GET /v1/sessions/{sid}/result` → 轨迹摘要 + Reward v1/SOP-PR 明细
- `POST /v1/sessions/{sid}/release` → 释放会话

## 两层校验模型

- **客观层（会话状态机直接拒绝）**：订单不存在/不属于当前用户、退款超实付、
  补偿超政策上限、虚拟商品不可退、定制商品不可换、未查询订单即动作
  （证据先行 guard）。
- **事实层（评分器依据 TaskFacts 审计）**：用户真实诉求是否符合政策窗口、
  是否命中风控、解决方式是否最优。工单"先受理、后审核"，模拟真实平台。

会话内订单状态变更是**深拷贝隔离**的，同一任务可重复评测而不污染世界。

## 政策表（8 类目）

无理由退货窗口、质量问题退货/仅退款不退货窗口、尺码换货窗口、价保窗口、
补偿券上限（普通 min(20%,30 元)，VIP min(30%,50 元)）。政策通过
`query_policy` 工具完全可观测，是 Agent 决策依据而非隐藏信息。

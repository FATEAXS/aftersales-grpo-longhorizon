# 数据采集与 SFT 数据构造

## 双教师

- **scripted（默认）**：确定性参考教师，是售后 SOP 的可执行实现。零成本、
  完全可复现，用于冷启动数据与流水线验证。Final-100 上限 100%。
- **api**：任意 OpenAI 兼容接口的 LLM 教师。与 scripted 教师共用同一环境契约
  与验收口径，来源在 metadata 中区分。

## 验收与审计

每条轨迹真实执行环境动作，按 Reward v1 终局结果验收（默认 total ≥ 1.0）。
`data/sft/metadata.json` 记录原始轨迹数、接受率、丢弃原因与奖励分布。

当前 scripted 教师：400 条原始轨迹，验收 384 条（96.0%），8:1 划分训练/验证。

## Loss Mask

SFT 只在 assistant 回合（文本 + tool_call）计算 loss；system、用户消息、
环境 observation 全部 mask。assistant 边界由目标模型 chat template 前缀差分
确定，超长或边界异常样本丢弃，绝不截断工具调用。

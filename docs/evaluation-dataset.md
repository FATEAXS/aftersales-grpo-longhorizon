# Final-100 Clean 留出集

- 100 个任务（`data/evaluation/tasks.jsonl` 公开任务卡；
  `environments/aftersalesim/data/tasks/evaluation.jsonl` 为私有 TaskFacts，
  仅环境服务与评分阶段使用）；
- 与 train(400)/validation(100) task_id 零重叠，同一世界种子生成；
- 场景分布与训练集同口径：约 64% 可解决 / 24% 应拒绝 / 12% 风控转人工，
  难度 easy/medium/hard 混合，覆盖三类干扰（口误订单号/同名多订单/
  缺失订单号）与歧义澄清任务；
- 脚本化参考策略上限 100%（教师上界），用于校准任务可解性。

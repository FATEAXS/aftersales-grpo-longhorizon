# 数据目录

| 目录 | 内容 |
|---|---|
| `sft/` | 教师轨迹（OpenAI 消息格式）+ 采集审计 metadata |
| `grpo/` | 任务卡 JSONL + veRL Parquet（train 含红线过采样） |
| `evaluation/` | Final-100 Clean 公开任务卡 |

重建世界与任务：`bash scripts/build_world.sh`（固定种子，可逐字节复现）。
私有 TaskFacts 位于 `environments/aftersalesim/data/tasks/`，仅环境服务使用。

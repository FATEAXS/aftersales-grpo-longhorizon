# veRL 在线 GRPO

固定 `verl==0.8.0`，不复制 veRL 源码；项目只保留 AgentLoop 适配层
（`src/aftersales_grpo/training/grpo/adapter/`）。

## 与环境的对接

- 每条 trajectory 由 `AftersaleSession` 绑定一个环境会话（contextvar 隔离并发）；
- 工具经 `AftersaleTool`（veRL BaseTool）在线程中调用同步 HTTP 客户端；
- 环境 finish/escalate 即终局，`_handle_processing_tools_state` 立即终止采样；
- 结束时从环境结算 `reward = Reward v1 total + λ × SOP-PR`，写入
  `extra_fields["aftersales"]`（含四面板所需全部诊断字段）。

## 训练配置（configs/grpo.yaml，24GB 单卡口径）

- LoRA r=16 + optimizer/param offload，vLLM colocate（gpu_memory_utilization 0.40）
- n=4 在线轨迹/prompt，temperature 0.7，max 25 工具步
- 红线协议：风控任务在 train.parquet 中按倍数过采样

## 运行

```bash
AFTERSALE_MODEL_PATH=outputs/models/sft-merged bash scripts/grpo.sh
bash scripts/export_grpo.sh outputs/models/grpo/global_step_120/actor outputs/models/grpo-merged
```

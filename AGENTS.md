# AfterSales GRPO 开发约定

- 环境工具契约的单一事实源在 `src/aftersales_grpo/environment/tools.py`；
  `configs/tools.json` 由 `dump_tools_json` 生成，
  `python scripts/check_tools_contract.py` 校验一致性。
- 世界与任务数据由固定种子生成，勿手改 `environments/aftersalesim/data/`；
  修改生成器后运行 `bash scripts/build_world.sh` 重建，
  并重启环境服务（`/health` 的 world_fingerprint 用于核对版本）。
- 评分口径（Reward v1 / SOP-PR）只存在于 `simulator/grader.py` 与
  `simulator/process_reward.py`；评测与训练侧不得自行猜分数。
- 轨迹产物必须通过盲评守卫（`evaluation/artifacts.py`）：
  任何 TaskFacts 私有字段不得进入 `trajectories.jsonl`。
- 提交前运行 `python -m pytest tests/`。

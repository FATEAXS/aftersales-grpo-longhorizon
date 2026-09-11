# LoRA SFT

```bash
BASE_MODEL=Qwen/Qwen3-1.7B bash scripts/sft.sh
```

- 技术栈：Transformers + PEFT（`scripts/train_lora_sft.py`）
- 默认超参：LoRA r=16 / α=32 / all-linear，bf16，梯度检查点，
  batch=1 × 累积 8，lr 1e-4 cosine，3 epochs（24GB 单卡口径）
- 产物：`outputs/models/sft-lora`（adapter）→ `scripts/merge_lora.py`
  合并为 `outputs/models/sft-merged`（可被 vLLM 直接服务）

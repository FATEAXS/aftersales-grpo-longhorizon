#!/usr/bin/env python3
"""合并 LoRA adapter 到基座权重，导出可被 vLLM 直接服务的完整模型。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="合并 LoRA adapter")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(model, str(args.adapter))
    merged = model.merge_and_unload()
    merged.save_pretrained(str(args.output))
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.save_pretrained(str(args.output))
    print(f"merged model saved -> {args.output}")


if __name__ == "__main__":
    main()

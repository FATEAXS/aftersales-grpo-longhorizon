#!/usr/bin/env python3
"""对验收后的售后 tool-calling 数据执行 LoRA SFT（Transformers + PEFT）。

默认按 24GB 单卡（RTX 3090）配置：bf16、梯度检查点、batch=1 + 梯度累积。
Loss 只作用于 assistant 回合（见 training/sft/dataset.py）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys_path = str(ROOT / "src")
if sys_path not in __import__("sys").path:
    __import__("sys").path.insert(0, sys_path)

import torch  # noqa: E402
from torch.utils.data import Dataset  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="售后 LoRA SFT")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", type=Path, default=ROOT / "data/sft/train.jsonl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/sft/validation.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/models/sft-lora")
    parser.add_argument("--max-length", type=int, default=12288)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--logging-steps", type=int, default=5)
    return parser.parse_args()


class SupervisedDataset(Dataset):
    def __init__(self, path: Path, tokenizer, max_length: int, tools: list[dict]):
        from aftersales_grpo.training.sft.dataset import load_supervised_examples

        self.examples, self.stats = load_supervised_examples(
            path, tokenizer, max_length=max_length, tools=tools
        )
        print(f"loaded {path.name}: kept={self.stats['kept']} dropped={self.stats['dropped']}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class Collator:
    """右填充 + labels 对齐；input_ids 已含全部所需 token。"""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, attention_mask, labels = [], [], []
        for item in batch:
            padding = max_len - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main() -> None:
    args = parse_args()
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from aftersales_grpo.environment.tools import TOOLS

    train_set = SupervisedDataset(args.train, tokenizer, args.max_length, TOOLS)
    eval_set = (
        SupervisedDataset(args.validation, tokenizer, args.max_length, TOOLS)
        if args.validation and args.validation.exists()
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        attn_implementation="sdpa",
    )
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_set else "no",
        eval_steps=50,
        save_strategy="no",
        bf16=torch.cuda.is_available(),
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        seed=42,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=Collator(tokenizer.pad_token_id),
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output / "train_stats.json").write_text(
        json.dumps({"train": train_set.stats, "eval": eval_set.stats if eval_set else None}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"LoRA adapter saved -> {args.output}")


if __name__ == "__main__":
    main()

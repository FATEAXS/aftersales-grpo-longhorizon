"""把 OpenAI tool-calling 消息轨迹渲染成带 Loss Mask 的训练样本。

只有 assistant 回合（文本 + tool_call）参与 loss；system、用户消息与环境
observation 一律 mask 为 ``IGNORE_INDEX``。assistant 边界由目标模型的 chat
template 逐回合渲染前缀差分确定，不手写任何特殊 token，换底座模型即换模板。
超长或模板边界对不齐的样本直接丢弃，绝不做可能截断工具调用的截断。
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

IGNORE_INDEX = -100

# Qwen3 类模板对"最后一条 assistant"会注入 <think></think> 空块，而同一消息
# 位于对话中间时不会注入——position 不变不成立。用两种渲染变体探测边界：
# A) 原样（适配"最后一条带 think"的口径）；B) 追加哑 user 消息把 assistant
# 变成中间消息（适配"中间不带 think"的口径）。哪个能作为完整渲染的前缀就用哪个。
BOUNDARY_DUMMY = {"role": "user", "content": "__BOUNDARY_DUMMY__"}


def _token_ids(tokenizer, text) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def _common_prefix_length(left: list[int], right: list[int]) -> int:
    length = 0
    for a, b in zip(left, right):
        if a != b:
            break
        length += 1
    return length


def normalize_messages_for_chat_template(messages: list[dict]) -> list[dict] | None:
    """OpenAI 格式的 arguments 是 JSON 字符串；多数官方模板要求 dict。"""
    normalized = deepcopy(messages)
    for message in normalized:
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            arguments = function.get("arguments")
            if not isinstance(arguments, str):
                continue
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                return None
            if not isinstance(parsed, dict):
                return None
            function["arguments"] = parsed
    return normalized


def build_supervised_example(
    messages: list[dict],
    tools: list[dict],
    tokenizer,
    max_length: int = 8192,
    chat_template=None,
) -> dict | None:
    """渲染一条轨迹并只打开 assistant 回合的标签；不可训练时返回 None。"""
    template = chat_template or tokenizer
    rendered = normalize_messages_for_chat_template(messages)
    if rendered is None:
        return None
    assistant_indices = [
        i for i, m in enumerate(rendered) if m.get("role") == "assistant"
    ]
    if not assistant_indices:
        return None

    try:
        full_text = template.apply_chat_template(
            rendered, tools=tools, tokenize=False, add_generation_prompt=False
        )
        input_ids = _token_ids(tokenizer, full_text)
    except Exception:
        return None
    if len(input_ids) > int(max_length):
        return None

    labels = [IGNORE_INDEX] * len(input_ids)
    for index in assistant_indices:
        try:
            prefix_ids = _token_ids(
                tokenizer,
                template.apply_chat_template(
                    rendered[:index],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=True,
                ),
            )
        except Exception:
            return None

        # 变体 A：assistant 作为渲染的最后一条消息
        try:
            through_a = _token_ids(
                tokenizer,
                template.apply_chat_template(
                    rendered[: index + 1],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                ),
            )
        except Exception:
            through_a = None
        # 变体 B：追加哑 user，让 assistant 处于中间位置
        through_b = None
        try:
            through_b = _token_ids(
                tokenizer,
                template.apply_chat_template(
                    rendered[: index + 1] + [dict(BOUNDARY_DUMMY)],
                    tools=tools,
                    tokenize=False,
                    add_generation_prompt=False,
                ),
            )
            dummy_ids = _token_ids(
                tokenizer,
                "<|im_start|>user\n__BOUNDARY_DUMMY__<|im_end|>\n",
            )
        except Exception:
            through_b = None

        matched = False
        for through_ids, trim in ((through_a, 0), (through_b, len(dummy_ids) if through_b else 0)):
            if not through_ids:
                continue
            start = _common_prefix_length(prefix_ids, through_ids)
            end = len(through_ids) - trim
            if start >= end or end > len(input_ids):
                continue
            if input_ids[:end] != through_ids[:end]:
                continue
            labels[start:end] = input_ids[start:end]
            matched = True
            break
        if not matched:
            return None

    if not any(label != IGNORE_INDEX for label in labels):
        return None
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def load_supervised_examples(path, tokenizer, max_length: int = 8192, tools=None):
    """读取 SFT JSONL 并渲染；返回 (examples, stats)。"""
    examples: list[dict] = []
    stats = {"total": 0, "kept": 0, "dropped": 0, "dropped_overlong": 0}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            stats["total"] += 1
            row = json.loads(line)
            example = build_supervised_example(
                messages=row["messages"],
                tools=tools if tools is not None else row.get("tools") or [],
                tokenizer=tokenizer,
                max_length=max_length,
            )
            if example is None:
                stats["dropped"] += 1
                continue
            example["task_id"] = row.get("task_id")
            examples.append(example)
            stats["kept"] += 1
    return examples, stats


def stable_subset(rows: list[dict], seed: int, tag: str) -> set:
    """按 task_id 稳定抽样工具（用于可复现的验证子集）。"""
    def key(task_id):
        return hashlib.sha256(f"{seed}:{tag}:{task_id}".encode()).hexdigest()

    return {row.get("task_id") for row in sorted(rows, key=lambda r: key(str(r.get("task_id"))))}


def write_tokenized_jsonl(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for example in examples:
            fh.write(json.dumps(example, ensure_ascii=False) + "\n")

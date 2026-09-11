"""轨迹处理纯函数(无 torch 依赖,便于轻量环境测试)。"""

from __future__ import annotations

import json
import re

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_assistant_turn(text: str) -> dict:
    """从一次生成文本中解析 <think> 剥离后的回复与 <tool_call> 调用。"""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
            name = payload.get("name")
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if name:
                calls.append(
                    {
                        "id": f"call_{len(calls) + 1}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                    }
                )
        except json.JSONDecodeError:
            continue
    reply = TOOL_CALL_RE.sub("", text).strip()
    return {"content": reply or None, "tool_calls": calls}


def assistant_loss_mask(generated_ids: list[int], segments: list[dict]) -> list[int]:
    """仅 assistant 分段参与 loss;工具响应等注入 token 一律为 0。"""
    mask = [0] * len(generated_ids)
    for seg in segments:
        for i in range(seg["start"], seg["end"]):
            mask[i] = 1
    return mask

#!/usr/bin/env python3
"""HF transformers 的 OpenAI 兼容服务(微批处理版,/v1/chat/completions + tools)。

与单请求版相比:工作线程把并发请求聚成最多 --batch-size 条,left-pad 联合
生成,显著提高 GPU 利用率与评测吞吐(实测 3~6 倍)。生成的 <tool_call> 块
解析为标准 tool_calls 字段,评测端无感。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import threading
import time
import uuid
from pathlib import Path

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

app = FastAPI(title="hf-openai-server")
_STATE: dict = {}
_QUEUE: list[dict] = []
_QUEUE_LOCK = threading.Lock()
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[dict]
    tools: list[dict] | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 768


def load(model_path: str, adapter: str | None, enable_thinking: bool = True):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # 批量生成必须左填充
    model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).cuda()
    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()
    model.eval()
    _STATE["tokenizer"] = tokenizer
    _STATE["model"] = model
    _STATE["enable_thinking"] = enable_thinking


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    calls = []
    for match in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
            arguments = payload.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:8]}",
                    "type": "function",
                    "function": {
                        "name": payload.get("name"),
                        "arguments": json.dumps(arguments, ensure_ascii=False),
                    },
                }
            )
        except json.JSONDecodeError:
            continue
    reply = TOOL_CALL_RE.sub("", text).strip()
    return reply, calls


def render_prompt(tokenizer, request: ChatRequest) -> str:
    render_kwargs = {}
    if request.tools:
        render_kwargs["tools"] = request.tools
    if not _STATE.get("enable_thinking", True):
        render_kwargs["enable_thinking"] = False
    try:
        return tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True, **render_kwargs
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            request.messages, tokenize=False, add_generation_prompt=True, **render_kwargs
        )


def run_batch(batch: list[dict]) -> None:
    tokenizer = _STATE["tokenizer"]
    model = _STATE["model"]
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    encoded = []
    for item in batch:
        text = render_prompt(tokenizer, item["request"])
        ids = tokenizer(text, add_special_tokens=False).input_ids[-(_STATE["max_input"] - 16):]
        encoded.append(ids)
    max_len = max(len(ids) for ids in encoded)
    input_ids = torch.tensor(
        [[pad_id] * (max_len - len(ids)) + ids for ids in encoded], device=model.device
    )
    attention = torch.tensor(
        [[0] * (max_len - len(ids)) + [1] * len(ids) for ids in encoded], device=model.device
    )
    with torch.no_grad():
        out = model.generate(
            input_ids,
            attention_mask=attention,
            max_new_tokens=_STATE["max_new_tokens"],
            do_sample=_STATE["do_sample"],
            temperature=_STATE["temperature"],
            top_p=0.95 if _STATE["do_sample"] else 1.0,
            pad_token_id=pad_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    for i, item in enumerate(batch):
        generated = out[i][input_ids.shape[1]:]
        content = tokenizer.decode(generated, skip_special_tokens=False)
        content = content.split("<|im_end|>")[0]
        reply, tool_calls = parse_tool_calls(content)
        message = {"role": "assistant", "content": reply or None}
        if tool_calls:
            message["tool_calls"] = tool_calls
        item["result"] = {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "choices": [{"index": i, "message": message, "finish_reason": "stop"}],
            "model": item["request"].model,
        }
        item["event"].set()


def batch_worker(batch_size: int, batch_wait: float, do_sample: bool, temperature: float) -> None:
    while True:
        batch: list[dict] = []
        with _QUEUE_LOCK:
            while _QUEUE and len(batch) < batch_size:
                batch.append(_QUEUE.pop(0))
        if not batch:
            time.sleep(0.02)
            continue
        if len(batch) < batch_size:
            deadline = time.time() + batch_wait
            while time.time() < deadline and len(batch) < batch_size:
                with _QUEUE_LOCK:
                    while _QUEUE and len(batch) < batch_size:
                        batch.append(_QUEUE.pop(0))
                if len(batch) < batch_size:
                    time.sleep(0.02)
        try:
            run_batch(batch)
        except Exception as exc:  # 单批失败不影响服务存活
            for item in batch:
                item["error"] = str(exc)[:200]
                item["event"].set()


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    item = {"request": request, "event": threading.Event(), "result": None, "error": None}
    with _QUEUE_LOCK:
        _QUEUE.append(item)
    if not item["event"].wait(timeout=900):
        raise HTTPException(status_code=504, detail="generation timeout")
    if item["error"]:
        raise HTTPException(status_code=500, detail=item["error"])
    return item["result"]


@app.get("/health")
def health():
    return {"status": "ok" if _STATE.get("model") else "loading"}


def main() -> None:
    parser = argparse.ArgumentParser(description="HF OpenAI 兼容服务(微批处理)")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None, help="可选 LoRA adapter(服务时合并)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--batch-wait", type=float, default=0.6)
    parser.add_argument("--max-input", type=int, default=12000)
    parser.add_argument("--max-new-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="服务端采样温度;0 为贪心(评测默认)")
    parser.add_argument("--no-thinking", action="store_true",
                        help="关闭 Qwen3 思考注入(基座评测用;SFT/GRPO 模型用默认)")
    args = parser.parse_args()
    load(args.model, args.adapter, enable_thinking=not args.no_thinking)
    _STATE["max_input"] = args.max_input
    _STATE["max_new_tokens"] = args.max_new_tokens
    _STATE["do_sample"] = args.temperature > 0
    _STATE["temperature"] = max(args.temperature, 1e-4)
    worker = threading.Thread(
        target=batch_worker,
        args=(args.batch_size, args.batch_wait, _STATE["do_sample"], _STATE["temperature"]),
        daemon=True,
    )
    worker.start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

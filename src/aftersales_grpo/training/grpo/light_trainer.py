"""轻量 GRPO 训练器 v2（单卡 24GB、HF transformers、无 vLLM 依赖）。

在 v1 基础上引入四个训练机制（与 docs/grpo.md 口径一致）：

1. 场景分层批次采样（P0）：每步按"敏感任务(拒绝/风控) : 常规任务"固定配额
   抽样，消除批次场景构成导致的组内基线漂移；
2. 风险加权优势：敏感任务（风控/应拒绝）上错误动作的 advantage
   按 --risk-weight 加权，把合规从事采样推进到梯度权重；
3. 组内最优基线：敏感任务组内若存在成功轨迹，以其中最优成功轨迹的奖励为基线计算优势（而非均值）——"大多数都错"的组不再把错误当正常；
4. 多内层 epoch + 比率裁剪（P0）：每批数据复用 inner_epochs 次，存储
   old log-prob，clip(π/π_old, 1-ε, 1+ε)，获得真正的 PPO 语义
   （v1 等价于单步 REINFORCE）。

另支持 --dump-pairs 导出组内 chosen/rejected 轨迹对（供离线 DPO）。

veRL 0.8 路径（configs/grpo.yaml + adapter/）面向多卡生产环境保留；
两条路径共享同一环境、同一奖励函数、同一 AgentLoop 语义。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
while not (ROOT / "pyproject.toml").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent  # 从 src 深处向上定位仓库根(默认 --facts/--output 依赖它)
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from aftersales_grpo.environment.client import AftersaleEnv, render_observation  # noqa: E402
from aftersales_grpo.environment.tools import TOOLS, initial_messages  # noqa: E402
from aftersales_grpo.training.grpo.trajectory_utils import (  # noqa: E402
    assistant_loss_mask,
    parse_assistant_turn,
)


def parse_args():
    parser = argparse.ArgumentParser(description="轻量 GRPO 训练器 v2")
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--tasks", type=Path, default=ROOT / "data/grpo/train.jsonl")
    parser.add_argument("--facts", type=Path,
                        default=ROOT / "environments/aftersalesim/data/tasks/train.jsonl")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/models/grpo-light-v2")
    parser.add_argument("--env-url", default=os.environ.get("AFTERSALE_BASE_URL", "http://127.0.0.1:5800"))
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--prompts-per-step", type=int, default=2)
    parser.add_argument("--group", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max-turns", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--process-lambda", type=float, default=0.5)
    parser.add_argument("--risk-weight", type=float, default=1.5,
                        help="敏感任务(拒绝/风控)的 advantage 权重")
    parser.add_argument("--inner-epochs", type=int, default=2,
                        help="每批数据的内层更新次数（>1 时启用 PPO 比率裁剪）")
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--dump-pairs", type=Path, default=None,
                        help="导出组内 chosen/rejected 轨迹对（离线 DPO 用）")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_prompt_text(tokenizer, opening: dict, tools: list[dict]) -> str:
    # 与 SFT 渲染一致：不注入空 think 块，由模型自行生成（SFT 已学到该格式）
    return tokenizer.apply_chat_template(
        initial_messages(opening),
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
    )


@torch.no_grad()
def rollout_once(args, model, tokenizer, env, task_id) -> dict:
    """跑一条轨迹：返回 token 序列、assistant 分段与奖励。"""
    reset = env.reset(task_id)
    opening = reset["observation"]
    rendered = build_prompt_text(tokenizer, opening, TOOLS)
    prompt_ids = tokenizer(rendered, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    generated_ids: list[int] = []
    segments: list[dict] = []
    done = False
    steps = 0

    while not done and steps < args.max_turns:
        new_ids = torch.tensor([generated_ids], dtype=torch.long, device=model.device)
        input_ids = torch.cat([prompt_ids, new_ids], dim=1) if new_ids.numel() else prompt_ids
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=True,
            temperature=args.temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )[0][input_ids.shape[1]:].tolist()
        start = len(generated_ids)
        generated_ids.extend(out)
        segments.append({"start": start, "end": len(generated_ids)})

        text = tokenizer.decode(out, skip_special_tokens=False)
        text = text.replace("<|im_end|>", "").strip()
        turn = parse_assistant_turn(text)
        if not turn["tool_calls"]:
            env.step("finish", {"resolution": "unresolved", "summary": (turn["content"] or "")[:200]})
            break

        all_done = False
        response_text = ""
        for call in turn["tool_calls"]:
            arguments = json.loads(call["function"]["arguments"])
            result = env.step(call["function"]["name"], arguments)
            steps += 1
            obs_text = render_observation(result)
            response_text += (
                f"<|im_start|>user\n<tool_response>\n{obs_text}\n</tool_response><|im_end|>\n"
            )
            user_reply = (result.get("observation") or {}).get("user_reply")
            if user_reply:
                response_text += f"<|im_start|>user\n{user_reply}<|im_end|>\n"
            all_done = all_done or bool(result.get("done"))
        # 工具响应注入上下文（assistant_loss_mask 只标 assistant 段，响应不参与 loss）
        generated_ids.extend(tokenizer(response_text, add_special_tokens=False).input_ids)
        if all_done:
            done = True

    outcome = env.result()
    env.release()
    breakdown = outcome["reward"]
    process = breakdown.get("process", {})
    reward = float(breakdown.get("total", 0.0)) + args.process_lambda * float(process.get("total", 0.0))
    return {
        "task_id": task_id,
        "prompt_len": prompt_len,
        "prompt_ids": prompt_ids.squeeze(0).tolist(),
        "generated_ids": generated_ids,
        "segments": segments,
        "reward": reward,
        "success": breakdown.get("base") is not None and breakdown["base"] >= 1.0,
        "steps": steps,
    }


def assistant_loss_mask(generated_ids: list[int], segments: list[dict]) -> list[int]:
    mask = [0] * len(generated_ids)
    for seg in segments:
        for i in range(seg["start"], seg["end"]):
            mask[i] = 1
    return mask


def batched_token_logprobs(model, ids: torch.Tensor, attention: torch.Tensor,
                           token_mask: torch.Tensor) -> torch.Tensor:
    """返回 mask 位置上的 token log-prob（float32, 已按行选取, 省显存）。"""
    logits = model(input_ids=ids, attention_mask=attention).logits[:, :-1, :]
    targets = ids[:, 1:]
    sel_mask = token_mask[:, 1:].squeeze(0).bool()
    selected_logits = logits.squeeze(0)[sel_mask].float()
    selected_targets = targets.squeeze(0)[sel_mask]
    log_probs = torch.log_softmax(selected_logits, dim=-1)
    return log_probs.gather(-1, selected_targets.unsqueeze(-1)).squeeze(-1)


def sequence_inputs(rollout: dict, device):
    """拼 prompt+generated, 超长时右截断并同步 loss_mask。"""
    sequence = rollout["prompt_ids"] + rollout["generated_ids"]
    loss_mask = [0] * rollout["prompt_len"] + rollout["loss_mask"]
    if len(sequence) > 12288:
        sequence = sequence[-12288:]
        loss_mask = loss_mask[-12288:]
    return (
        torch.tensor([sequence], dtype=torch.long, device=device),
        torch.tensor([loss_mask], dtype=torch.long, device=device),
        torch.tensor([[1] * len(sequence)], dtype=torch.long, device=device),
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).cuda()
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter, is_trainable=True)
        print(f"[grpo-light] 继承 adapter: {args.adapter}", flush=True)
    else:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                       task_type="CAUSAL_LM", target_modules="all-linear"),
        )
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()

    task_ids = [
        json.loads(line)["task_id"]
        for line in open(args.tasks, encoding="utf-8")
        if line.strip()
    ]
    # 私有事实仅用于训练侧的分层抽样与风险加权（与评分同源，不进入轨迹产物）
    facts_map: dict[str, dict] = {}
    if args.facts and Path(args.facts).exists():
        for line in open(args.facts, encoding="utf-8"):
            if line.strip():
                fact = json.loads(line)
                facts_map[fact["task_id"]] = fact

    def is_sensitive(task_id: str) -> bool:
        fact = facts_map.get(task_id, {})
        return bool(fact.get("requires_escalation") or fact.get("ineligible"))

    sensitive_ids = [t for t in task_ids if is_sensitive(t)]
    standard_ids = [t for t in task_ids if not is_sensitive(t)]
    print(f"[grpo-light] 分层抽样池: 敏感 {len(sensitive_ids)} / 常规 {len(standard_ids)}", flush=True)

    env = AftersaleEnv(base_url=args.env_url)
    if env.health()["status"] != "ok":
        raise SystemExit("环境服务未启动")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    metrics_path = output / "metrics.jsonl"
    pairs_path = args.dump_pairs
    if pairs_path:
        pairs_path.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    for step in range(1, args.steps + 1):
        started = time.time()
        # ---- P0-2: 场景分层批次采样（敏感:常规 = 1:1, 池不足时回退） ----
        half = max(1, args.prompts_per_step // 2)
        batch_tasks = []
        if sensitive_ids:
            batch_tasks.extend(rng.sample(sensitive_ids, min(half, len(sensitive_ids))))
        remaining = args.prompts_per_step - len(batch_tasks)
        batch_tasks.extend(rng.sample(standard_ids, min(remaining, len(standard_ids))))

        rollouts = []
        for task_id in batch_tasks:
            group = [rollout_once(args, model, tokenizer, env, task_id) for _ in range(args.group)]
            rewards = [r["reward"] for r in group]
            mean = sum(rewards) / len(rewards)
            std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
            sensitive = is_sensitive(task_id)
            successes = [r for r in group if r["success"]]

            for r in group:
                if sensitive and successes:
                    # ---- 创新 B: 组内最优基线 ----
                    anchor = max(r_["reward"] for r_ in successes)
                    r["advantage"] = (r["reward"] - anchor) / (std + 1e-4)
                    if r["success"] and r["reward"] == anchor:
                        r["advantage"] += 0.5  # 组内最优轨迹额外加 0.5
                else:
                    r["advantage"] = (r["reward"] - mean) / (std + 1e-4)
                # ---- 创新 A: 风险加权优势 ----
                if sensitive:
                    r["advantage"] *= args.risk_weight
                r["group_mean"] = mean
                r["sensitive"] = sensitive
            rollouts.extend(group)

        # ---- P0-3: 多内层 epoch + 比率裁剪 ----
        for rollout in rollouts:
            # loss_mask 仅含 assistant 段;prompt 前缀由 sequence_inputs 统一添加
            rollout["loss_mask"] = assistant_loss_mask(
                rollout["generated_ids"], rollout["segments"]
            )
            ids, mask, attention = sequence_inputs(rollout, model.device)
            rollout["ids"], rollout["mask"], rollout["attention"] = ids, mask, attention

        optimizer.zero_grad(set_to_none=True)
        loss_total = 0.0
        with torch.no_grad():
            for rollout in rollouts:
                rollout["old_logprobs"] = batched_token_logprobs(
                    model, rollout["ids"], rollout["attention"], rollout["mask"]
                ).detach()
        for _inner in range(max(1, args.inner_epochs)):
            inner_loss = 0.0
            for rollout in rollouts:
                new_logprobs = batched_token_logprobs(
                    model, rollout["ids"], rollout["attention"], rollout["mask"]
                )
                ratio = torch.exp(new_logprobs - rollout["old_logprobs"])
                clipped = torch.clamp(ratio, 1 - args.clip_eps, 1 + args.clip_eps)
                pg = -torch.min(ratio, clipped) * rollout["advantage"]
                inner_loss += pg.sum() / rollout["mask"].sum().clamp(min=1)
            (inner_loss / len(rollouts)).backward()
            loss_total += inner_loss.item() / (len(rollouts) * max(1, args.inner_epochs))
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        # ---- P1-8: 导出组内 chosen/rejected 轨迹对（离线 DPO） ----
        if pairs_path:
            by_task: dict[str, list] = {}
            for r in rollouts:
                by_task.setdefault(r["task_id"], []).append(r)
            with open(pairs_path, "a", encoding="utf-8") as fh:
                for task_id, group in by_task.items():
                    group_rewards = [r["reward"] for r in group]
                    if max(group_rewards) - min(group_rewards) < 0.5:
                        continue
                    chosen = max(group, key=lambda r: r["reward"])
                    rejected = min(group, key=lambda r: r["reward"])
                    fh.write(json.dumps({
                        "task_id": task_id,
                        "prompt_ids": chosen["prompt_ids"],
                        "chosen_ids": chosen["generated_ids"],
                        "chosen_mask": assistant_loss_mask(chosen["generated_ids"], chosen["segments"]),
                        "rejected_ids": rejected["generated_ids"],
                        "rejected_mask": assistant_loss_mask(rejected["generated_ids"], rejected["segments"]),
                        "chosen_reward": chosen["reward"],
                        "rejected_reward": rejected["reward"],
                    }, ensure_ascii=False) + "\n")

        rewards = [r["reward"] for r in rollouts]
        successes = sum(1 for r in rollouts if r["success"])
        record = {
            "step": step,
            "mean_reward": round(sum(rewards) / len(rewards), 4),
            "success_rate": round(successes / len(rollouts), 4),
            "loss": round(loss_total, 4),
            "sensitive_share": round(sum(1 for r in rollouts if r["sensitive"]) / len(rollouts), 2),
            "seconds": round(time.time() - started, 1),
        }
        with open(metrics_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[step {step}/{args.steps}] {record}", flush=True)

        if step % args.save_every == 0 or step == args.steps:
            model.save_pretrained(str(output / f"step_{step}"))
            print(f"[checkpoint] -> {output / f'step_{step}'}", flush=True)

    model.save_pretrained(str(output / "final"))
    print(f"[done] final adapter -> {output / 'final'}")


if __name__ == "__main__":
    main()

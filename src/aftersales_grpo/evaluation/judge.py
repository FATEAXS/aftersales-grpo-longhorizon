"""P2-12: 通信质量 LLM Judge 面板。

用 OpenAI 兼容的 Judge 模型对轨迹中的客服回复逐条打 0/1/2 分
（0=无人味/答非所问, 1=合格, 2=有安抚与清晰政策解释），聚合进 summary.json
的 panel_f_judge。与原购物项目的 Pro Judge 思路同源,但只评"沟通文本"
这一确定性面板覆盖不到的维度,且不接触任何私有事实。
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

JUDGE_SYSTEM = """你是客服质检专家。对给定的客服回复做质量评分(只看文本本身):
0 分 = 机械模板、答非所问、或没有回应处用户情绪;
1 分 = 合格,说明了处理结果;
2 分 = 合格之外还有清晰的政策依据/时间线说明,或有效的情绪安抚。
只输出一个 JSON: {"score": 0|1|2, "reason": "<=20字"}"""


def judge_replies(
    trajectories_path: Path,
    summary_path: Path,
    judge_base_url: str,
    judge_model: str,
    api_key: str = "EMPTY",
    max_tasks: int | None = 50,
) -> dict:
    client = httpx.Client(timeout=120)
    scores: dict[str, list] = {}
    with open(trajectories_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            replies = [
                m.get("content")
                for m in record.get("messages", [])
                if m.get("role") == "assistant" and m.get("content")
            ]
            if not replies:
                continue
            scores[record["task_id"]] = replies
            if max_tasks and len(scores) >= max_tasks:
                break

    per_task = {}
    total, count = 0, 0
    for task_id, replies in scores.items():
        reply_block = "\n".join(f"- {r[:300]}" for r in replies[:5])
        try:
            response = client.post(
                f"{judge_base_url.rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": judge_model,
                    "messages": [
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": f"客服回复:\n{reply_block}\n\n请评分。"},
                    ],
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
            payload = json.loads(content[content.find("{"): content.rfind("}") + 1])
            score = int(payload.get("score", 0))
        except Exception:
            score = None
        if score is not None:
            per_task[task_id] = score
            total += score
            count += 1

    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    summary["panel_f_judge"] = {
        "judge_model": judge_model,
        "judged_tasks": count,
        "mean_score": round(total / count, 3) if count else None,
        "score_two_rate": round(sum(1 for s in per_task.values() if s == 2) / count, 4) if count else None,
        "score_zero_rate": round(sum(1 for s in per_task.values() if s == 0) / count, 4) if count else None,
        "per_task": per_task,
    }
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    client.close()
    return summary["panel_f_judge"]

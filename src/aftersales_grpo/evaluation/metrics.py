"""四面板指标聚合：终局/合规/沟通澄清/效率稳健。"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .artifacts import guard_blind_trajectory


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def summarize(records: list[dict], actor_name: str) -> dict:
    total = len(records)
    strict_success = sum(1 for r in records if r.get("success"))
    valid_records = [r for r in records if r.get("terminal") not in (None, "infrastructure_error")]

    terminal_types = Counter(r.get("terminal_type") or "none" for r in records)
    base_reasons = Counter(r.get("base_reason") or "unknown" for r in records)

    # ---- Panel B: 合规 -------------------------------------------------
    redline_tasks = [r for r in records if r.get("task_flags", {}).get("escalation_expected")]
    ineligible_tasks = [r for r in records if r.get("task_flags", {}).get("ineligible_expected")]
    redline_violations = sum(1 for r in redline_tasks if r.get("redline_violation"))
    redline_success = sum(1 for r in redline_tasks if r.get("success"))
    ineligible_violations = sum(1 for r in ineligible_tasks if r.get("ineligible_action_taken"))
    ineligible_success = sum(1 for r in ineligible_tasks if r.get("success"))
    premature = sum(1 for r in records if r.get("base_reason") == "premature_escalation")
    resolvable = [
        r
        for r in records
        if not r.get("task_flags", {}).get("escalation_expected")
        and not r.get("task_flags", {}).get("ineligible_expected")
    ]
    policy_attempts = sum(1 for r in records if r.get("violations"))

    # ---- Panel C: 沟通与澄清 -------------------------------------------
    clarification_tasks = [r for r in records if r.get("task_flags", {}).get("clarification_expected")]
    clarification_asks = sum(1 for r in clarification_tasks if r.get("clarified"))
    clarification_success = sum(1 for r in records if r.get("clarification_success"))
    misreports = sum(1 for r in records if r.get("misreport"))

    # ---- Panel D: 效率与稳健 -------------------------------------------
    steps = [r.get("steps", 0) for r in records]
    guard = [r.get("guard_rejections", 0) for r in records]
    repeats = sum(1 for r in records if r.get("repeat_action_count", 0) >= 2)
    by_difficulty = {
        difficulty: {
            "total": len(bucket),
            "success_rate": _rate(sum(1 for r in bucket if r.get("success")), len(bucket)),
            "mean_reward": round(sum(r.get("reward", 0.0) for r in bucket) / len(bucket), 4) if bucket else None,
        }
        for difficulty, bucket in _group(records, "difficulty").items()
    }
    by_scenario = {
        scenario: {
            "total": len(bucket),
            "success_rate": _rate(sum(1 for r in bucket if r.get("success")), len(bucket)),
        }
        for scenario, bucket in sorted(_group(records, "scenario").items())
    }

    summary = {
        "actor": actor_name,
        "total_tasks": total,
        "not_judged": total - len(valid_records),
        "panel_a_terminal": {
            "strict_success_rate": _rate(strict_success, total),
            "mean_reward": round(sum(r.get("reward", 0.0) for r in records) / total, 4) if total else None,
            "terminal_types": dict(terminal_types),
            "base_reasons": dict(base_reasons),
        },
        "panel_b_compliance": {
            "redline_tasks": len(redline_tasks),
            "redline_violation_rate": _rate(redline_violations, len(redline_tasks)),
            "redline_success_rate": _rate(redline_success, len(redline_tasks)),
            "ineligible_action_rate": _rate(ineligible_violations, len(ineligible_tasks)),
            "ineligible_success_rate": _rate(ineligible_success, len(ineligible_tasks)),
            "premature_escalation_count": premature,
            "premature_escalation_rate": _rate(premature, len(resolvable)),
            "policy_attempt_rate": _rate(policy_attempts, total),
        },
        "panel_c_communication": {
            "clarification_tasks": len(clarification_tasks),
            "clarification_ask_rate": _rate(clarification_asks, len(clarification_tasks)),
            "clarification_success_rate": _rate(clarification_success, len(clarification_tasks)),
            "misreport_rate": _rate(misreports, total),
        },
        "panel_d_efficiency": {
            "mean_steps": round(sum(steps) / total, 2) if total else None,
            "mean_guard_rejections": round(sum(guard) / total, 3) if total else None,
            "repeat_loop_rate": _rate(repeats, total),
            "by_difficulty": by_difficulty,
            "by_scenario": by_scenario,
        },
    }
    return summary


def _group(records: list[dict], key: str) -> dict:
    grouped: dict = {}
    for record in records:
        grouped.setdefault(record.get(key) or "unknown", []).append(record)
    return grouped


def guard_artifacts(trajectories_path) -> dict:
    """对产物做盲评守卫扫描，返回发现的泄漏（应为空）。"""
    leaks = []
    with open(trajectories_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            record = json.loads(line)
            found = guard_blind_trajectory(record)
            if found:
                leaks.append({"line": line_no, "task_id": record.get("task_id"), "keys": found})
    return {"file": str(trajectories_path), "leaks": leaks, "clean": not leaks}


def attach_pass_at_k(trajectories_path, summary_path, samples: int) -> None:
    """P2-11: 按原始 task_id 归组计算 pass@1 / pass@k / 方差面板。

    多次采样评测的轨迹 task_id 形如 "<原id>#s<k>"。"""
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    groups: dict[str, list] = {}
    with open(trajectories_path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            base_id = record["task_id"].rsplit("#s", 1)[0]
            groups.setdefault(base_id, []).append(record)

    def _stdev(values):
        if len(values) < 2:
            return None
        mean = sum(values) / len(values)
        return round(math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)), 4)

    task_stats = {}
    pass_counts = [0] * samples
    for base_id, records in groups.items():
        rewards = [r.get("reward", 0.0) for r in records]
        successes = [1 if r.get("success") else 0 for r in records]
        for k in range(samples):
            if successes[: k + 1] and any(successes[: k + 1]):
                pass_counts[k] += 1
        task_stats[base_id] = {
            "mean_reward": round(sum(rewards) / len(rewards), 4),
            "reward_std": _stdev(rewards),
            "success_rate": round(sum(successes) / len(successes), 4),
        }

    n = len(groups)
    summary["panel_e_variance"] = {
        "samples_per_task": samples,
        "tasks": n,
        "pass_at_1": round(pass_counts[0] / n, 4) if n else None,
        **{f"pass_at_{k + 1}": round(pass_counts[k] / n, 4) for k in range(samples)},
        "mean_reward_std_across_tasks": round(
            sum(s["reward_std"] for s in task_stats.values() if s["reward_std"] is not None)
            / max(1, sum(1 for s in task_stats.values() if s["reward_std"] is not None)),
            4,
        ),
        "per_task": task_stats,
    }
    Path(summary_path).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

"""跨阶段对比：Baseline / SFT / GRPO 的 summary.json 汇成 Markdown 与表格数据。"""

from __future__ import annotations

import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_comparison(summaries: dict[str, dict]) -> str:
    """summaries: {阶段名: summary dict} -> Markdown 对比表。"""
    lines = [
        "# Baseline → SFT → GRPO 对比",
        "",
        "同一 Final-100 Clean 留出任务、同一环境世界、同一 Reward v1 口径。",
        "",
        "| 阶段 | 严格成功率 | 平均 Reward | 红线违规率 | 澄清成功率 | 平均步数 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, s in summaries.items():
        a = s["panel_a_terminal"]
        b = s["panel_b_compliance"]
        c = s["panel_c_communication"]
        d = s["panel_d_efficiency"]

        def pct(v):
            return "—" if v is None else f"{v:.1%}"

        lines.append(
            f"| {name} | {pct(a['strict_success_rate'])} | {a['mean_reward']} "
            f"| {pct(b['redline_violation_rate'])} | {pct(c['clarification_success_rate'])} "
            f"| {d['mean_steps']} |"
        )
    lines.append("")
    lines.append("机器可读汇总见各阶段 `summary.json`。")
    return "\n".join(lines)


def write_comparison(summary_paths: dict[str, Path], output_path: Path) -> Path:
    summaries = {name: load_summary(path) for name, path in summary_paths.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_comparison(summaries), encoding="utf-8")
    return output_path

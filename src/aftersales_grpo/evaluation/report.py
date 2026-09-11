"""评测报告生成：单文件 HTML（四面板 + 分场景/难度分解），零外部依赖。"""

from __future__ import annotations

import html
import json
from pathlib import Path


def build_report(summary: dict, title: str) -> str:
    e = html.escape

    def num(value, pct=False):
        if value is None:
            return "—"
        return f"{value:.1%}" if pct else f"{value:g}"

    def bar(value, max_value=1.0, negative=False):
        if value is None:
            return ""
        width = max(2, min(100, abs(value) / max_value * 100))
        color = "#d64545" if (negative or value < 0) else "#2f9e44"
        return f'<div class="bar"><div style="width:{width:.0f}%;background:{color}"></div></div>'

    a = summary["panel_a_terminal"]
    b = summary["panel_b_compliance"]
    c = summary["panel_c_communication"]
    d = summary["panel_d_efficiency"]

    scenario_rows = "".join(
        f"<tr><td>{e(scenario)}</td><td>{bucket['total']}</td>"
        f"<td>{num(bucket['success_rate'], True)}</td>{bar(bucket['success_rate'])}</tr>"
        for scenario, bucket in d["by_scenario"].items()
    )
    difficulty_rows = "".join(
        f"<tr><td>{e(level)}</td><td>{bucket['total']}</td><td>{num(bucket['success_rate'], True)}</td>"
        f"<td>{num(bucket['mean_reward'])}</td>{bar(bucket['success_rate'])}</tr>"
        for level, bucket in sorted(d["by_difficulty"].items())
    )

    return f"""<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"><title>{e(title)}</title>
<style>
body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; margin: 24px auto; max-width: 1080px; color: #1c232e; }}
h1 {{ font-size: 22px; }} h2 {{ font-size: 16px; margin-top: 28px; border-left: 4px solid #2f6fed; padding-left: 8px; }}
table {{ border-collapse: collapse; width: 100%; margin: 8px 0; font-size: 13px; }}
th, td {{ border-bottom: 1px solid #e3e7ee; padding: 6px 10px; text-align: left; }}
th {{ background: #f4f6fa; }}
.bar {{ background: #eef1f6; height: 10px; width: 160px; border-radius: 5px; overflow: hidden; }}
.bar > div {{ height: 100%; }}
.kpi {{ display: flex; gap: 16px; flex-wrap: wrap; }}
.kpi .card {{ flex: 1; min-width: 150px; background: #f7f9fc; border: 1px solid #e3e7ee; border-radius: 8px; padding: 12px 16px; }}
.kpi .value {{ font-size: 26px; font-weight: 700; color: #17427e; }}
.kpi .label {{ font-size: 12px; color: #5b6675; }}
.small {{ color: #5b6675; font-size: 12px; }}
</style></head><body>
<h1>{e(title)}</h1>
<p class="small">Actor: {e(summary.get("actor", ""))} · 任务数: {summary["total_tasks"]} · 未判定: {summary["not_judged"]}</p>

<div class="kpi">
  <div class="card"><div class="value">{num(a["strict_success_rate"], True)}</div><div class="label">严格成功率</div></div>
  <div class="card"><div class="value">{num(a["mean_reward"])}</div><div class="label">平均 Reward v1</div></div>
  <div class="card"><div class="value">{num(b["redline_violation_rate"], True)}</div><div class="label">红线违规率</div></div>
  <div class="card"><div class="value">{num(c["clarification_success_rate"], True)}</div><div class="label">澄清成功率</div></div>
  <div class="card"><div class="value">{num(d["mean_steps"])}</div><div class="label">平均步数</div></div>
</div>

<h2>面板 A · 终局与奖励</h2>
<table><tr><th>指标</th><th>值</th></tr>
<tr><td>严格成功率（base ≥ 1.0）</td><td>{num(a["strict_success_rate"], True)}</td></tr>
<tr><td>平均 Reward</td><td>{num(a["mean_reward"])}</td></tr>
<tr><td>终局类型分布</td><td>{e(json.dumps(a["terminal_types"], ensure_ascii=False))}</td></tr>
</table>

<h2>面板 B · 合规审计</h2>
<table><tr><th>指标</th><th>值</th></tr>
<tr><td>风控任务数 / 违规率 / 成功率</td><td>{b["redline_tasks"]} / {num(b["redline_violation_rate"], True)} / {num(b["redline_success_rate"], True)}</td></tr>
<tr><td>应拒绝任务违规办理率</td><td>{num(b["ineligible_action_rate"], True)}</td></tr>
<tr><td>过早升级率（可解决却转人工）</td><td>{num(b["premature_escalation_rate"], True)}（{b["premature_escalation_count"]} 例）</td></tr>
<tr><td>政策被拒动作尝试率</td><td>{num(b["policy_attempt_rate"], True)}</td></tr>
</table>

<h2>面板 C · 沟通与澄清</h2>
<table><tr><th>指标</th><th>值</th></tr>
<tr><td>歧义任务主动澄清率</td><td>{num(c["clarification_ask_rate"], True)}（{c["clarification_tasks"]} 例）</td></tr>
<tr><td>澄清后成功率</td><td>{num(c["clarification_success_rate"], True)}</td></tr>
<tr><td>虚报结果率</td><td>{num(c["misreport_rate"], True)}</td></tr>
</table>

<h2>面板 D · 效率与稳健</h2>
<table><tr><th>指标</th><th>值</th></tr>
<tr><td>平均步数 / 平均 guard 拒绝 / 重复循环率</td><td>{num(d["mean_steps"])} / {num(d["mean_guard_rejections"])} / {num(d["repeat_loop_rate"], True)}</td></tr>
</table>
<h3 style="font-size:14px">按难度</h3>
<table><tr><th>难度</th><th>任务数</th><th>成功率</th><th>平均 Reward</th><th></th></tr>{difficulty_rows}</table>
<h3 style="font-size:14px">按场景</h3>
<table><tr><th>场景</th><th>任务数</th><th>成功率</th><th></th></tr>{scenario_rows}</table>

<p class="small">Reward v1 与 SOP-PR 口径见 docs/reward-v1.md；本报告由 scripts/build_eval_report.py 生成。</p>
</body></html>"""


def write_report(summary_path: Path, report_path: Path, title: str | None = None) -> Path:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    title = title or f"Aftersales 评测报告 · {summary.get('actor', '')}"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(summary, title), encoding="utf-8")
    return report_path

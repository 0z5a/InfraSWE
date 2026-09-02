# ruff: noqa: E501 - literal report markup is intentionally kept on semantic lines.

from __future__ import annotations

import html
import json
from pathlib import Path

from infraswe.models.score import ScoreResult
from infraswe.models.trial import TrialRecord


def render_markdown_report(record: TrialRecord, score: ScoreResult) -> str:
    replay_rows = "\n".join(
        f"| {item.index} | {'PASS' if item.passed else 'FAIL'} | "
        f"{item.duration_sec:.3f} | {item.failure.value if item.failure else '-'} |"
        for item in record.replays
    )
    reasons = ", ".join(score.gate.reasons) or "none"
    return f"""# InfraSWE Trial {record.trial_id}

- Task: `{record.task_id}`
- State: `{record.state.value}`
- Resolved@1: `{str(score.resolved_at_1).lower()}`
- StableResolved@1: `{str(score.stable_resolved_at_1).lower()}`
- Core-100: `{score.core_100:.3f}`
- InfraExt-100: `{score.infra_ext_100:.3f}`
- InfraTotal: `{score.infra_total:.3f}`
- Coverage: `{score.coverage:.1%}`
- Hard-gate reasons: `{reasons}`

## Fresh replays

| Replay | Result | Seconds | Failure |
|---:|---|---:|---|
{replay_rows}

## Raw score components

```json
{json.dumps(score.model_dump(mode="json"), indent=2, sort_keys=True)}
```
"""


def render_html_report(record: TrialRecord, score: ScoreResult) -> str:
    status = "PASS" if score.stable_resolved_at_1 else "FAIL"
    rows = "".join(
        "<tr>"
        f"<td>{item.index}</td><td>{'PASS' if item.passed else 'FAIL'}</td>"
        f"<td>{item.duration_sec:.3f}</td>"
        f"<td>{html.escape(item.failure.value if item.failure else '-')}</td>"
        "</tr>"
        for item in record.replays
    )
    payload = html.escape(json.dumps(score.model_dump(mode="json"), indent=2, sort_keys=True))
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>InfraSWE {html.escape(record.trial_id)}</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:980px;margin:40px auto;padding:0 20px;color:#17202a}}
.hero{{display:flex;justify-content:space-between;align-items:end;border-bottom:2px solid #dde3ea}}
.status{{font-size:2rem;font-weight:800;color:{"#14804a" if status == "PASS" else "#b42318"}}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:24px 0}}
.card{{padding:16px;border:1px solid #dde3ea;border-radius:10px;background:#f8fafc}}
.value{{font-size:1.6rem;font-weight:700}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #dde3ea;text-align:left}}
pre{{overflow:auto;background:#111827;color:#e5e7eb;padding:16px;border-radius:10px}}
</style></head><body>
<div class="hero"><div><h1>InfraSWE trial</h1><p>{html.escape(record.task_id)} · {html.escape(record.trial_id)}</p></div><div class="status">{status}</div></div>
<div class="cards">
<div class="card">Core-100<div class="value">{score.core_100:.2f}</div></div>
<div class="card">InfraExt-100<div class="value">{score.infra_ext_100:.2f}</div></div>
<div class="card">InfraTotal<div class="value">{score.infra_total:.2f}</div></div>
<div class="card">Coverage<div class="value">{score.coverage:.0%}</div></div>
</div>
<h2>Fresh replays</h2><table><thead><tr><th>#</th><th>Result</th><th>Seconds</th><th>Failure</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Score protocol</h2><pre>{payload}</pre>
</body></html>"""


def write_reports(run_dir: Path, record: TrialRecord, score: ScoreResult) -> None:
    (run_dir / "report.md").write_text(render_markdown_report(record, score), encoding="utf-8")
    (run_dir / "index.html").write_text(render_html_report(record, score), encoding="utf-8")

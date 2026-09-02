# InfraSWE 极化判定政策 v0.5.1

状态：兼容保留 / 历史评测非默认（不回写既有冻结锁）  
生效日期：2026-09-02

## 目标

减少宽泛的 `revise`，让正式评分更接近明确的合并/拒绝边界，同时保留基础设施缺失的 `unresolved`。

## 机器判定

- `accept` / `accept_with_scope`：正式 ProjectFit 必须 `>= 85.0`；
- `< 60`、InfraCert fundamental hard failure：直接 `reject`；
- `[60, 85)` 或 component-floor failure：默认 `reject`；
- 上述失败只有在 PR 满足“新且明确在审”时才可为 `revise`；
- official evidence 缺失且没有 PR review context：`unresolved`，不把基础设施缺失伪装成 `reject`。

“新且明确在审”冻结为：

```text
PR age <= 30 days
AND (
  current-head human non-author review within 14 days
  OR pending human review request with activity within 14 days
)
```

以下情况不能使用 `revise`：

```text
没有 review context
只有 bot/author feedback
review 不属于 current head 且没有 pending request
PR age > 30 days
review/activity idle > 14 days
```

其中，`PR age >= 90 days` 且已有真人非作者 review、但仍 open 的 PR，使用明确原因码 `STALE_REVIEWED_OPEN_REJECT`。

## 历史 oracle

揭盲后标签只允许：

```text
merged                         -> accept
closed and unmerged            -> reject
open + active new review       -> revise
other open                     -> reject
```

对 `merged` oracle，机器分数必须 `>= 85.0` 才算校准命中；缺分或低于 85 均记为 score-floor violation。review/outcome 只能在机器锁之后进入 oracle，不能回流到同一盲测的机器判定。

## 版本兼容

- R7 及更早的 judgment/prediction lock 保持原摘要与原判定；
- `historical-explainable-agent-v0.5-r4` 和 `historical-merge-prediction-v0.5-r1` 仍可重放；
- R8 的 30 项实测显示 coarse static 特征无法支撑极化数值：30 项被统一打成 94，锁后只匹配 13/30；该结果保留为负对照。
- 新的通用历史评测默认恢复 `historical-explainable-agent-v0.5-r4` 和 `historical-merge-prediction-v0.5-r1`，要求先冻结 case-specific contract，且不强制产生 0–100 分。
- `historical-explainable-agent-v0.5-r5-polarized` 和 `historical-merge-prediction-v0.5-r2-polarized` 仍可显式选择，以重放 R8 与既有锁；不得把它们的 proxy score 当成 official ProjectFit。
- 正式 sealed ProjectFit 的 `>=85` 准入下限不受这个历史评测默认值变化影响；缺少 official evidence 时仍为 `unresolved`，不能用静态启发式补分。

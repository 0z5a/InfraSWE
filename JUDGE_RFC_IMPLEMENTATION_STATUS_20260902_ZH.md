# InfraSWE v0.5.3 Judge RFC 实现状态（2026-09-02）

## 当前裁决

仓库已完成 v0.5.3 的首个**离线可信核心**，但尚未接入任何 hosted/open-weight
模型执行器，也没有真实的双模型家族校准集。因此当前不能宣称已经产生正式
`bounded-semantic` Judge 分数。

实现坚持以下权威边界：

- Judge 只可拥有 `P`、`M`、`U` 中 Seal 前定义的 semantic-residual criterion；
- `P <= 0.25`、`M <= 0.20`、`U <= 0.10` 的 component 内权重上限由 schema 强制；
- `InfraCert/C/R/O/X/CellArtifactPerformance` 明确禁止绑定 Judge；
- 不存在 `LLMJudge-100`，聚合结果固定声明 `top_level_score_status=not-a-score`；
- `InfraCert=fail/unresolved` 时不发布 Judge-assisted projection；
- Judge 失败、abstain 或 disagreement 是 evaluator uncertainty，不自动给 candidate 记 0 分。

## 已实现

### 1. Profile、Rubric 与 Judge Cell

- exact hosted snapshot/API revision 或完整 open-weight runtime identity；
- 动态/未 pin identity 不能获得 `bounded-score`；
- official panel 至少两个 model family、每 member 至少两次 sealed repetition；
- calibration report、95% CI floor、drift sentinel、security policy、panel 与
  aggregation policy 全部进入 Cell digest；
- human-reviewed rubric 逐 criterion 绑定唯一 owner，component 权重必须完整归一；
- Judge Cell、aggregation 与 projection 均可重算 canonical SHA-256。

### 2. JudgeInputPack

- 输入源必须位于显式 `source_root` 内；
- manifest 与每个 artifact 都是 content-addressed；
- candidate-controlled 内容使用
  `UNTRUSTED_CANDIDATE_CONTENT` 边界并做 HTML escape；
- 常见 author/sign-off cue 在 candidate surface 中被盲化；
- aggregate score、leaderboard、agent/model name 等结构化 cue 在非 rubric evidence
  中 fail-closed；
- private key、AWS/GitHub/OpenAI token 与常见 credential 形式会阻断 pack build；
- pack audit 会检查 manifest、artifact digest 与 untrusted boundary。

### 3. Structured output 与 evidence grounding

- Pydantic strict schema 禁止额外字段，支持 score、abstain、insufficient evidence、
  out-of-scope 与 possible injection；
- 0--4 ordinal grade 必须与 rubric anchor、`grade/4` normalized value 一致；
- 每个 score-contributing verdict 必须解析 evidence ref；
- 至少一条 supporting ref 必须来自 target authority 或 deterministic evidence；
- 仅引用 candidate 自述、伪造 ref、缺 required evidence、injection 后仍评分都会使
  member vote invalid，而不是直接惩罚 candidate；
- 已知 candidate agent family 与 Judge family 相同时，vote 强制零权重。

### 4. Panel 与有界投影

- 按 calibration weight 计算 weighted median 与 weighted MAD；
- 检查 member repetition、跨 family 数量、within-member range、cross-family range；
- required criterion abstain -> `unresolved-judge`；
- order/family disagreement -> `judge-disagreement`；
- security flag -> `security-review-required`；
- 只有 official aggregation 可按 criterion owner 投影到 `P/M/U`；
- 同时发布 deterministic-core projection 与 Judge-assisted projection；
- 跨 Judge Cell 排名固定禁止。

### 5. CLI 与 schema

已提供：

```text
infraswe judge profile validate
infraswe judge cell seal|audit
infraswe judge pack build|audit
infraswe judge validate-output
infraswe judge aggregate
infraswe judge project
```

已新增 14 个 v0.5.3 JSON Schema；仓库当前共 66 个 fresh schema。

## 负控与回归

新增 13 个 Judge 专项测试，覆盖：

- 单模型家族与 unpinned alias；
- rubric 权重不闭合或超过上限；
- calibration/injection 指标不达标；
- secret、identity cue 与 untrusted boundary；
- fabricated/candidate-only evidence ref；
- injection 后未 abstain；
- same-family self evaluation；
- multi-family weighted median；
- abstention 与 cross-family disagreement；
- InfraCert hard gate；
- Cell/pack tamper detection。

本地结果：`ruff` 通过，`pytest = 241 passed`，`schema check = 66 fresh`。

## 尚未实现，不能伪装完成

1. hosted snapshot、local open-weight 与 read-only agent adapter；
2. 真实 provider request/response metadata、retry、token/cost 采集与 cache backend；
3. pairwise A/B + B/A guard 的执行 runner（当前聚合器已具备跨 family/repeat gate）；
4. domain-stratified JudgeBench calibration set 构建器、bootstrap CI 与 kappa 计算器；
5. drift sentinel 的模型调用执行器；
6. read-only agent tool sandbox、step/token/read budget enforcement；
7. verifier-audit、trajectory-audit 与 human appeal 的完整 orchestrator；
8. Draft D3J/D4/D6 生命周期写回和 D8/D9 archive integration；
9. report UI 与 differential judging cache invalidation；
10. 真实的两个模型家族、每家族两次调用所形成的 official evidence pack。

这些缺口需要真实模型 identity、校准样本、provider/data-egress 决策或额外运行时，
不能用本地 fixture 代替。

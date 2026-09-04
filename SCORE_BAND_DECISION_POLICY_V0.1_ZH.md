# InfraSWE 分数段判定政策 v0.1

状态：当前默认

## 默认执行

- 默认评估引擎为 `infraswe`；
- 默认评估范围为 `full`，完整运行受 Draft 约束的评估项；
- 默认开启 Seal；没有满足 review、identity 与 evidence 要求时不得伪造 seal；
- 诊断或外部评估模式可以显式选择，但不能冒充 official sealed 结果。

## 唯一综合分与层级

正式结果只输出一个顶层数值分：`overall_score_100`。ProjectFit 与 BenchmarkTrust
不能再作为与总分并列的顶层字段；它们必须作为 `microscores` 下的两个同级解释分：

```text
overall_score_100
microscores.project_fit
microscores.benchmark_trust
```

BenchmarkCost 是成本卡，不是 microscore。前三道硬门全部通过且两个 microscore 都有
正式数值后，综合分使用冻结公式：

```text
overall = 100 * (ProjectFit / 100)^0.85 * (BenchmarkTrust / 100)^0.15
```

## 有序硬门与三分类

分类必须按顺序执行，不能重排或用后项补偿前项：

1. `maintainability`：演进可维护性及其 floor；
2. `deployability`：InfraCert、项目合同适配与运行适配；
3. `performance`：沿用原始性能、复用、利用率证据，Pure Triton 可移植性和适用的
   release gate；
4. `overall-score`：前三项均通过后才允许计算综合分。

硬门 `fail` 直接映射为 `reject`，后续门为 `not-run`；硬门证据 `unresolved` 映射为
`check`，后续门同样不得执行。只有前三道硬门全部通过后，综合分才使用唯一映射：

```text
score < 50       -> reject
50 <= score <=65 -> check
score > 65       -> accept / accept_with_scope
```

边界值 50 和 65 都属于 `check`。`accept_with_scope` 与 `accept` 使用同一个 `>65`
门槛，区别只来自明确声明的支持范围。

65 分以上的具体分值只用于候选质量评估、同一 ProjectComparisonCell 内排序和诊断，
不得再派生额外的处置标签。`accept_with_scope` 只是 accept 类的范围限定符，不是第四
分类。综合分不能绕过硬失败，或把缺少 official evidence 的结果改写成 accept。

历史 PR 离线校准只有 outcome-blind 粗粒度证据时，可以发出明确标注为 non-official 的
`overall_score_100` 以校验分段分类，但不得伪造 ProjectFit 或 BenchmarkTrust
microscore，也不得冒充正式 InfraSWE 结果。

## 历史兼容

旧 `project-mergeability-polarized-v0.5.1`、
`historical-merge-prediction-v0.5-r2-polarized` 与
`historical-polarized-oracle-v0.5.1` 继续按原 85 分规则重放。新产物使用独立的 v0.1
policy id；既有 hash-frozen / sealed artifact 不回写。

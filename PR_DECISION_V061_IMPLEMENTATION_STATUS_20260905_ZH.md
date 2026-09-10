# InfraSWE v0.6.1 PRDecisionPlane 实施状态

日期：2026-09-05

## 已实现

- `src/infraswe/pr_decision/contracts.py`
  - A/C/R 三分类、补丁版本与预测时点身份、完整 policy identity。
  - 冻结 maintainability → deployability/correctness-safety → performance → overall score
    的决策顺序。可靠的 blocking obligation 可在不篡改分数边界的情况下 Reject；只有前三门
    通过后才使用 `<50 reject / 50..65 check / >65 accept`，且 `check` 必须列出未决义务。
  - 旧 `Accuracy3 >= 95% + Recall_A >= 99%` 合同和新增
    `Accuracy3 >= 95% + Recall_A >= 99% + Precision_A >= 95%` 合同相互独立。
  - 四项纠错计数和整数误差预算。
- `snapshot.py`、`label_vault.py`
  - outcome-blind 输入快照与揭盲标签分仓、时间/head SHA 约束、内容摘要审计。
  - 历史记录只允许 external-policy、curriculum、offline-retrieval 与定性审计；不伪装成
    policy-gradient 轨迹。
- `errorbook.py`
  - 结构化持续误判、原因/责任归属、决定性证据、冻结 policy 绑定与摘要审计。
- `evidence.py`、`obligations.py`
  - claim/source/authority/time/head/counterevidence 合同。
  - maintainability → deployability/correctness-safety → performance 的固定义务顺序。
  - 未知必须写出缺失证据，不能被当作已证伪。
- `precedent.py`、`project_router.py`
  - Accept/Reject/合法例外三路对比检索，以及时间、split、patch-family 泄漏保护。
  - 项目条件化 profile 与小样本共享回退；禁止作者、PR 编号和终态特征参与路由。
- `cascade.py`
  - AcceptChallenger 与 RejectRescuer 双向纠错，仅接受有权威证据的改变。
  - `NEEDS_*` 只作为内部取证动作，不产生第四类真值。
  - 条件召回率级联预算按乘积校验，并分别统计修复/引入的 Accept FN/FP。
- `calibration.py`
  - 冻结阈值曲线、`Recall_A >= floor` 下的最大 Precision/Accuracy、目标可达性报告。
  - 校准用途限于置信报告或证据路由，不修改旧 overall score 分段。
- `release_gate.py`、`report/decision_card.py`
  - 完整三分类混淆矩阵、有效/无效分母、Accuracy/Recall/Precision 独立硬门。
  - 点估计或预分配 alpha 的单侧 Wilson 下界。
  - digests、切片、覆盖率、未决/人工辅助/无效分母、成本、统计假设和选择协议。
- CLI 与 JSON Schema
  - `infraswe pr-decision snapshot-audit SNAPSHOT`
  - `infraswe pr-decision gate CASES --preset baseline-95-99`
  - `infraswe pr-decision gate CASES --preset precision-95-99-95`
  - 也可用 `--contract METRIC-CONTRACT` 加载独立合同。

## 明确未声称

- 此实现是决策协议、审计与发布门的 reference slice，不是新的 80,089 PR 实测结果。
- 不回写任何已有 `oracle-audit.json`、judgment lock 或其他 sealed 历史产物。
- 历史累计指标不等同于冻结最终 policy 的未见集指标。
- 阈值可达性失败时不会通过移动边界伪造判别能力；应返回证据、标签和决策策略训练。
- 没有精确行为 token/logprob/mask 的 legacy transcript 仍不具备 policy-gradient 资格。

## 发布门使用约定

默认 `pr-decision gate` 使用新增三硬门：

```text
Accuracy3  >= 0.95
Recall_A   >= 0.99
Precision_A >= 0.95
```

输入为 JSON/YAML 数组，每条至少包含：

```yaml
- case_id: owner/repo#42
  predicted_label: accept
  oracle_label: accept
  valid: true
```

真实基础设施硬超时需按预先冻结规则设为 `valid: false` 并填写 `invalid_reason`；它作为
neutral abandon 单独计数，不能用于隐藏困难样本。发布报告必须标明
`prequential_campaign_result`、`frozen_policy_holdout_result` 或
`historical_diagnostic_replay_result`，不同轨道不得互相替代。

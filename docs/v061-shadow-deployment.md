# v0.6.1 三硬门与影子部署

部署边界：communication group-0008 封存后，group-0009 开始前。
用户确认的独立点估计门：Accuracy3 ≥ 95%、Accept Recall ≥ 99%、Accept Precision ≥ 99%。
旧版两硬门及 Precision 95% 合同保留，旧 sealed 六件套不回写。

## 实際启用的范围

- `pr-decision gate` 默认采用 `strict-95-99-99`，独立计算三个指标；零支持不能通过。
- 数值计算拒绝重复 case ID；未经独立审核的 caller-declared invalid 排除不能让 gate 通过。
- 数值结果永远标记 `release_authorized=false`。数值通过不证明独立 heldout、标签可信或采集权限隔离。
- snapshot 只接受受限结构化 observation；未实现可信 resolver 前拒绝 memory refs。审核重新验证类型、时间和 SHA，不只检查 hash。
- cascade 必须获得 case identity 与控制面独立提供的 claim digest；跨 PR/SHA、重复证明、未闭合原始义务不能把 Reject/Check 改成 Accept。
- 路由限定平面 structural features；未接入合格 EvidencePack 时，禁止填 ProjectFit/BenchmarkTrust 数值。
- 三个 campaign 均不再自行 commit/push、删除凭据或停止实例。最终收尾须独立验证三线完成、全量同步和合格发布；没有以可随意设置的布尔文件代替验证。
- communication worker 在新组开始时记录 `activation.json`，主判决后、reveal 前写 `shadow-lock.json`，六件套齐全后写 `shadow-audit.json`；重启可幂等补审计。

sidecar 位于 `communication-bulk-95pct/decision-v061-shadow/groups/group-NNNN/`，
不放进原始 sealed group。写入使用原子 create-if-absent；不同内容不能覆盖。
激活记录与影子锁固定 profile、代码、队列、输入和 baseline judgment digest。

## E1 候选，不是已合格的新主策略

候选仅限制 Approval 快捷路径：完整 review 列表、精确当前 head、预测时点前的批准，
且无更晚的负向 review。collector 没有保存 reviewer identity，无法可靠归属撤回；
这是保守启发式，必须保留其召回损失，不能解释成正式工程判定。
规则只消费显式白名单参数，不传入 next-policy 中的 retrospective oracle 数据。
主判决继续原冻结策略链，45 秒超时仍为 neutral abandon，不重试，不因超时改变 oracle 资格。

下表是原始 g5/g6 的**事后配对诊断**，不是新 live 结果，也不是独立未见集验收。

| 组 | 策略 | Accuracy3 | Accept Recall | Accept Precision |
|---|---|---:|---:|---:|
| g5 | 原策略 | 50.3501% | 80.9406% | 32.8974% |
| g5 | E1 候选 | 62.5875% | 64.7277% | 38.4842% |
| g6 | 原策略 | 50.8842% | 82.2955% | 33.7000% |
| g6 | E1 候选 | 63.3967% | 66.4225% | 39.8535% |

g6 去掉 505 个旧 FP，但引入 130 个新 FN；不能用准确率改善抵消召回退化。
因此本次候选 promotion 是 **no-go**，只启用影子对照。

## 仍未完成的发布前置项

本批次不是整份设计的全部实现：独立 collector authentication、label-vault 访问隔离、
可信 eligibility/population attestor、人工工程标签复核、train-only 检索、完整 E2–E4 消融、
真实校准拟合、固定策略独立 heldout、SFT/RL 权重训练尚未由这次部署证明完成。
`provenance_authenticated=false` 必须保留；自由 claim 文本不能因有 digest 就变成可信事实。
正式发布与停机继续 fail closed。

通信 g0–g8：27,000 个 PR，26,983 个 oracle-eligible，已有 13,164 个三分类错误。
即便余下 PR 全部正确且 oracle-eligible，旧不可变累计口径最高约为 93.7198%，
已经不可能达到 95%。新策略应另建预声明独立评测轨道，不能重标、删错样本或改写旧组。

## 验证与恢复

本地必须通过 Ruff、全量 pytest、145 个 schema freshness、task validate、shell syntax
与 CLI/真实六件套配对诊断后才能提交代码。部署以列举文件逐项 SHA-256 和远端 preflight 校验为准。
禁止宽泛同步用户 dirty 文件、凭据、partial 或整个 results 到 Git。

若需要再次部署，先在 `/workspace/infraswe-control/communication.deployment-pause`
放暂停标记；当前组结束后，round 会写 `communication.boundary-waiting` 并等待。
核对当前组封存与等待索引后只停止 communication supervisor，再执行受检代码切换。
不要中断单 PR 测试，也不要在已激活组中重新使用同一影子 profile 路径写不同代码。

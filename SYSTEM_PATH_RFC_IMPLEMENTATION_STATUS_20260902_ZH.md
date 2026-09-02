# InfraSWE v0.5.1/v0.5.2 System Path RFC 实现状态

更新时间：2026-09-02 12:47 +08:00  
状态：首个可执行纵切面完成；官方 sealed workload 闭环尚未签发

## 1. 已冻结的冲突处理

- 通信与 memory-tiering 不新增全局领域分；不存在 `CommunicationScore-100`、
  `CPUOffload-100` 或 tier/backend-count 分。
- system path 使用普通 ProjectFit 五槽 envelope，其中非纯 Triton 路径固定 `X=N/A`。
- 并发稳定性 `C` 使用六因子公式：goodput 0.30、tail 0.20、jitter 0.15、
  overlap/progress 0.15、resource 0.10、fairness 0.10。
- `O` 是同一份 `C` 证据的 identity projection，不运行第二套评分。
- memory-tiering 的 `CellArtifact` 仅在同一 CPU/GPU/NUMA Cell 内解释，不进入跨 Cell
  排名。
- 3 个 fresh process 仅为 diagnostic；official 至少 5 个，7 个为高置信度。

## 2. Precedent Retrieval v0.5.1

已实现：

- immutable `RepositorySnapshot`、domain-aware `CandidateFootprint`、版本化 `QueryPlan`；
- Python/C/C++ 文件的确定性 symbol、caller、build、config、failure、lifecycle 提取；
- communication collective/protocol/transport/topology/lifecycle anchors；
- memory object/tier/residency/prefetch/eviction/allocator/version/NUMA anchors；
- 混合 domain 在自动提取时 fail-closed，要求拆分或显式选域；
- SQLite + FTS5 precedent store、typed graph edge、exact/graph/failure/lifecycle/negative
  检索和冻结 RRF；semantic sidecar 始终 optional；
- target/future/fingerprint/near-duplicate leakage audit；
- typed-scope conflict detection、allowlist rule template、coverage 与 RetrievalTrust card；
- digest-bound D4 human accept/edit/advisory/reject/conflict action；只有 `accepted/edited`
  rule 可进入 D3 contract；
- `PrecedentSet` 和完整 `RetrievalBundle` canonical digest 审计；
- RetrievalTrust 固定 `candidate_score_effect=none`。

CLI 闭环：

```text
precedent footprint
  -> precedent index [--edges]
  -> precedent plan
  -> precedent retrieve
  -> precedent review-rules
  -> precedent audit / audit-bundle
```

## 3. Communication Draft v0.5.1

已实现：

- `communication-path-integration-v1` Draft、provider/layer 参数、communication cell identity；
- collective order、rank divergence、deadlock、fallback、resource/lifecycle 的确定性
  InfraCert hard-gate evidence；
- light/normal/knee/saturation/overload/soak 六档负载 ladder；
- system-path `C`、`O=C`、实现复用 `U` 与无 portability 维度的约束；
- raw latency/algBW/busBW/overlap/rank-skew lifecycle card，固定 `not-a-score`；
- 10 个 concrete communication catalog profile；每个 profile 为独立 comparison scope。

## 4. Memory-tiering Draft v0.5.2

已实现：

- `memory-tier-integration-v1`、显式 tier、residency transition、capacity 与 owner；
- semantic/scoring/load-anchor 三种 baseline 分离，capacity-enable baseline 必须可解释；
- CPU/socket/NUMA/host memory/interconnect 完整进入 Cell identity；
- mutable/durable object version token、visibility、isolation、UAF、partial copy、stale/lost
  update、queue、host leak、pageable fallback 与 teardown hard gate；
- service、residency、transfer 的 cell-local artifact；traffic amplification 不会因重复搬运得分；
- 抽象父 profile `memory-tiering-offload-runtime-v1` 固定不可 Seal；
- KV cache、weight、training state、activation、checkpoint staging 五个 concrete profile，
  object kind 与 profile 一一校验。

## 5. 极化合并政策

- `accept` / `accept_with_scope` 只允许 official ProjectFit `>=85`；
- `[60,85)` 或 component-floor failure 默认 `reject`；
- `revise` 只允许 30 天内的新 PR，并且 current-head 真人非作者 review 或 pending human
  review activity 在 14 天内；
- 90 天以上、已有真人 review 但长期 open 的 PR 使用
  `STALE_REVIEWED_OPEN_REJECT`；
- 历史 oracle：merged→accept、closed-unmerged→reject、active-new-review open→revise、
  其他 open→reject；merged calibration 还要求机器分数 `>=85`。

## 6. 当前验证

- Ruff：`src tests benchmarks` 全通过；
- Pytest：`228 passed`；
- 协议 schema：`52` 份，freshness 通过；
- checked-in system profile catalog：16 个 profile，与 built-in model 精确一致；
- precedent CLI 已用真实临时 SQLite/FTS 文件完成 footprint→index→plan→retrieve→digest
  端到端测试。

## 7. 尚未冒充完成的部分

- 当前 index 输入是可审计的 normalized JSONL；GitHub PR/review/CI/revert 的在线 source
  adapter 与 permission snapshot collector 尚未接入。
- D8/D9 maintenance-memory writeback 尚未接入完整 CLI。
- communication/offload 的项目级 collector、fault injector、双域 contention runner 仍需接
  真实 workload；当前已有 schema、oracle 与 scorer，不等于已产生官方 E2/E4 证据。
- 尚无 maintainer-reviewed、D6 sealed 的 communication 或 memory-tier official Draft，因此
  不签发官方 ProjectFit，也不把 proxy workload 解释成原生跨项目分数。

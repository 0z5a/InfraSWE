# InfraSWE v0.6 Agentic RL RFC 实现状态（2026-09-03）

## 当前裁决

仓库已完成 v0.6 的首个**可执行、fail-closed reference slice**。它把策略身份、
harness/token 边界、sandbox/snapshot、trajectory、verifier-anchored reward、
训练 batch、历史经验迁移和多 GPU Rollout Fabric 约束变成可验证 schema、协议函数、
CLI 与负控。

这不等于已经具备生产级 Agentic RL 服务。当前没有真实模型 serving gateway、
rootless sandbox enforcement、训练框架 adapter 或分布式 gang scheduler；因此
`infraswe rl fabric preflight` 默认返回非零并列出缺失能力，不能凭两张可见 GPU
宣称 production-ready。

v0.4 的 official score 与 v0.5 ProjectFit 保持不变。training-only reward、teacher、
curriculum 和 historical PR 标签均不能改写 InfraCert 或 official score。

## 已实现

### 1. Policy 与 harness identity

- `PolicySnapshot` 冻结 base weights、config、tokenizer、chat template、adapter、serving
  engine、dtype/parallelism 和 decoding defaults；
- `ExternalPolicyState` 作为 policy identity 的一部分单独版本化和 digest；
- `PolicyCell` 同时绑定 policy、external state、harness、skill、tool、prompt、
  compaction、decoding、feedback visibility 与 sandbox；
- `AgentHarnessProfile` 冻结模型 API 边界和调用预算，不采集 hidden chain of thought；
- 所有顶层工件使用 canonical SHA-256，额外字段 fail-closed。

### 2. Exact model-boundary trajectory

- `ModelBoundaryTrace` 保留原始 input/output token IDs、逐 token role、trainable mask、
  rollout logprob、sampling 参数、logical turn 以及 compaction 前后 history digest；
- tool/system/environment/padding token 不能进入 assistant loss mask；
- `StepTokenSpan` 必须完整、无重叠地复现 boundary mask，并绑定正确的 trajectory step；
- `LogprobFidelityReport` 根据冻结阈值验证 rollout/train correlation、delta logp 与
  mask alignment；
- reconstructed/transcript-only trajectory 永远不能声称 policy-gradient eligibility。

### 3. Sandbox、snapshot、branch 与 episode

- agent/replay profile 必须 rootless、unprivileged；host mount、共享 cache 写、
  verifier private files、secret、live process memory 与 GPU context snapshot 均被禁止；
- runtime attestation 缺失时只能声明 `declarative-only`，不能伪装 enforcement；
- snapshot/branch 绑定 workspace、logical harness state、prefix、pivotal step 与
  runtime reinitialization；
- episode 状态机、failure owner/code、timezone、seal window 与默认 training mask
  由 schema 强制；
- `SECURITY_REJECTED` 的 mask 可按已冻结的数据治理策略取 0 或 1，其余状态使用 RFC
  默认映射。

### 4. Verifier-anchored RewardPack

- `VerifierOutcomePack` 必须覆盖 CO/RI/NB/MP/SL/ES，并由 validity 与 hard obligations
  唯一确定 InfraCert；
- RewardCompiler 只读取已封存 `TrialSeal`、`EvidencePack`、episode、outcome、
  qualification、profile 与 RewardEvent；
- authoritative event 和 obligation 的 EvidenceRef 必须在 EvidencePack 中解析，且
  verifier、trusted meter、infra attestor、bounded Judge 的 authority 必须匹配；
- infra/benchmark/rollout invalid 是 mask，不合成负 reward；
- hard-fail 与 hard-pass band 分离，training shaping cap 不能跨越 hard margin；
- performance reward 必须来自同 Benchmark Cell 的 trusted-meter evidence；
- Candidate self-report 没有 hard/performance authority，重复事实直接拒绝；
- RewardPack 可在 task/reward-hacking 事故后撤销，保留原 anchor provenance，但
  `training_mask=0`；
- official projection 固定独立可重算，training reward 固定不能影响 official score。

### 5. Judge/teacher、feedback 与信用分配

- feedback 做显式 leakage scan；hidden/private/heldout/future-fix cue 被阻断；
- teacher 只可消费 train split 且 `POLICY_VISIBLE` 或
  `TEACHER_VISIBLE_REDACTED` 的安全反馈；
- heldout、`ADVANTAGE_ONLY` 与 `PRIVATE_AUDIT_ONLY` 反馈不能进入 teacher prompt；
- detection-only step 默认得到零 causal blame；
- token modulation 只能用正 multiplier 调制 verifier advantage，不能翻转符号。

### 6. Algorithm、batch 与多 GPU fabric

- `AlgorithmProfile` 表达 DAPO/GSPO/StepPO/PPO/GRPO/RLOO/external-policy；DAPO 强制
  asymmetric clipping 配置和 valid-trainable-token normalization，StepPO 强制 step
  granularity；
- `GroupManifest` 保留全部 attempts、invalid/replenishment 与最终成员；
- `RLBatchManifest` 检查同 task、同 cell、同 reward schema、behavior policy、
  policy lag、revoked reward、feedback leakage 和有效 token/step 计数；
- `RolloutFabricProfile` 分离 policy/environment/learner/judge/build/metadata pools，
  对 MI1+ 禁止 official environment GPU 与 policy/learner/judge GPU 重叠；
- `GangLeaseRecord` 对多 GPU 做原子分配，partial allocation 在 workload start 前失败；
- backpressure 冻结 admission、sandbox concurrency、environment GPU-seconds、trajectory
  buffer、reward queue 和 policy-lag 上限。

### 7. 历史 PR 经验迁移

`infraswe rl legacy migrate` 只读取已完成且同时具备 input、exact-head、judgment、
reveal 与 audit 的 group，并保存每个源工件的 SHA-256。旧数据统一标记为：

```text
trajectory_fidelity = reconstructed
harness_fidelity = transcript-only
exact_token_ids_available = false
policy_gradient_eligible = false
reward_qualification = not-qualified
reward_pack_sha256 = null
```

它们只允许用于 external policy、curriculum、offline retrieval 与 qualitative audit；
迁移器不会伪造 token、logprob 或 RewardPack。任何尚未同时生成 input、exact-head、
judgment、reveal 与 audit 的 group 都不会进入 manifest。

首次 inference offline 大组已经完整封账：3000 次尝试中 2992 个 oracle 有效、8 个
因输入/结果不可用而标记 invalid。冻结策略的 exact accuracy 为 `2448/2992 = 81.82%`，
同 cohort legacy 为 `1689/2992 = 56.45%`，提升 `25.37` 个百分点；check precision 为
`16/22 = 72.73%`，reject precision 为 `934/1233 = 75.75%`。冻结策略只接受了
`1498/1798 = 83.31%` 的真实合并 PR，未达到 85% 门槛，因此不会原样推广。下一组候选
只改变 `maintainer_requires_runtime_source` 一个变量，回放 exact 为
`2466/2992 = 82.42%`，并把合并 PR 接受召回提高到 `1556/1798 = 86.54%`；该回放只用于
选择 group 1 候选，不替代下一组独立验证。

正式迁移清单合并了 380 条已封账训练经验与上述 3000 条推理经验，共 3380 条；其中
3372 条 valid、8 条 invalid，policy-gradient、伪造 token 与 qualified reward 计数均为
0。清单 digest 为
`sha256:8b1889a249f0eaf700d3abf5cdc758b0e6a1a6ff98044b8fe6949ecc85ff11c1`，见
[`results/historical-pr-blind-20260901/v06-legacy-experience-manifest.json`](results/historical-pr-blind-20260901/v06-legacy-experience-manifest.json)。

### 8. CLI 与 schema

已提供：

```text
infraswe rl policy validate
infraswe rl harness validate
infraswe rl episode inspect
infraswe rl episode inspect-outcome
infraswe rl reward inspect
infraswe rl batch validate
infraswe rl legacy migrate
infraswe rl fabric validate
infraswe rl fabric preflight
infraswe rl train validate-seal
```

新增 29 个 v0.6 JSON Schema；仓库当前共 131 个 fresh schema。

## 对 RFC digest 环的明确修正

RFC 示例同时让 EpisodeSeal 引用 RewardPack、RewardPack 又引用 EpisodeSeal。两个对象
若都用内容 digest 标识，会形成不可构造的互相哈希环。reference slice 使用以下单向
封账 DAG：

```text
EpisodeSeal(execution + evidence)
    -> RewardPack
    -> EpisodeOutcomeSeal(EpisodeSeal + RewardPack)
```

`EpisodeOutcomeSeal` 是最终联合身份；这保留了全部绑定关系，同时保证每个 digest 都可
独立重算。

## 负控与回归

v0.6 专项现有 33 项测试，覆盖 policy/tokenizer tamper、adapter/external state identity、
tool-token mask、token span fidelity、mixed policy、reconstructed PG 欺骗、privileged
sandbox、伪 runtime attestation、状态机跳跃、invalid/security mask、hard margin、
Candidate reward spoof、cross-cell performance、伪 EvidenceRef/authority、重复事实、
feedback leakage、sign flip、detection blame、DAPO normalization、stale/revoked batch、
partial gang、GPU pool overlap、runtime fail-closed、legacy migration、schema 与 CLI。

本地结果：`ruff format --check src tests benchmarks` 与 `ruff check src tests benchmarks`
均通过，`pytest = 321 passed`，`schema check = 131 fresh`。

双 A100-SXM4-40GB 远端冒烟已完成，v0.6 专项同样为 `33 passed`。能力门禁按设计以退出码
5 拒绝 production-ready，并报告 topology attestation、rootless enforcement、exact-token
gateway、trainer adapter 与 distributed gang enforcement 均缺失。证据见
[`results/agentic-rl-v0.6-a100-sxm4-20260903`](results/agentic-rl-v0.6-a100-sxm4-20260903)。

## 尚未实现，不能伪装完成

1. 真实 open-weight/hosted policy endpoint 与 exact-token/logprob gateway；
2. 生产级 rootless sandbox、seccomp/device/network enforcement 及其远程 attestation；
3. filesystem + logical-agent-state snapshot/replay 的实际运行时；
4. verl/slime/Megatron 等 trainer adapter、optimizer/checkpoint lineage 与真实反向传播；
5. 跨节点 resource broker、拓扑证明、gang scheduler、故障恢复和 elastic scaling；
6. 真实 Judge/teacher 模型调用、校准、feedback sanitization service 与 hint pool；
7. reward qualification suite 的真实任务级统计、mutation、noise 和 replay 证据；
8. rollout-as-a-service RPC/control plane、trajectory store 与 policy registry；
9. 生产 telemetry、TrainingTrust/Model Card、污染 checkpoint 追踪与撤销传播；
10. 真实端到端 online RL 收敛、成本、吞吐和多 GPU 隔离认证。

这些能力必须由真实 runtime、模型、训练框架和资源控制面提供。schema 或 fixture 通过
不能替代生产证据。

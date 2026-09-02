# InfraSWE v0.1 P0A/P0B/P0C 实现状态（2026-09-02）

## 当前裁决

仓库已完成 v0.1 三份 P0 RFC 的首个可执行 reference slice：任务先完成三元契约和
Verifier Qualification，候选只通过声明制品边界进入 pristine trial，runner/capability/
resource/topology 在运行前确定性解析并绑定到 TrialSeal、EvidencePack 与 Benchmark Cell。

这不是 production cloud scheduler，也不是完整容器安全执行面。当前实现足以把协议不变量
固化为严格 schema、纯函数 controller、可审计 digest、CLI 和负控，但尚不能在没有真实
maintainer seal、可信硬件 attestation、provider lease enforcement 与多节点 gang scheduler
的情况下签发 v0.1 official result。

## P0A：Task Tri-Contract 与 Verifier Qualification

已实现：

- `TaskSpecification`：pinned target、requirement ID、允许/禁止修改范围、支持/不支持
  行为、tolerance/resource/artifact/capability policy 引用；
- `TaskAcceptanceContract`：CO/RI/NB/MP/SL/ES/MT obligation bucket、severity、oracle、
  repeat、failure owner、evidence owner、visibility 与 provenance；
- `WitnessSet`：witness identity、target/build digest、覆盖义务、license、reviewer，并固定
  `grading_usage=forbidden`；
- specification→contract coverage、hard obligation→witness coverage 和 content digest audit；
- baseline differential、witness fresh replay、weighted mutation adequacy、negative controls、
  alternative-valid-solution breadth、flakiness、environment sentinel、leakage 与 provenance；
- human qualification review、qualification report、VerifierTrustCard、VerifierCoverageReport；
- `TaskSeal` 绑定 task/specification/contract/verifier/witness/mutation/alternative/
  capability/artifact/resource/topology/cell/selection/qualification/reviewer/season；
- `VerifierResult` 将 Candidate failure、infra invalid、benchmark defect、N/A 和 unresolved
  分开，ES 失败不会伪装成 Candidate FAIL。

CLI：

```text
infraswe task audit-contract
infraswe task qualify
infraswe task seal
infraswe task audit-seal
```

## P0B：Artifact Boundary 与 Evidence Transport

已实现：

- allow-rule + filesystem/canonicalization/build/evidence policy；
- workspace freeze attestation、repository-relative path 规范化、symlink/special-file/size/
  mode 检查、稳定双 stat 读取和 secret 拦截；
- Git 候选 patch 同时捕获 base 后 committed、staged、unstaged 与 untracked 状态；
- content-addressed `CandidateArtifactManifest`，并可对落盘 payload 重算 digest；
- TransportEnvelope 把候选制品绑定到 base revision 与 destination policy；
- pristine apply/build result、sanitized environment 和 Candidate/benchmark failure ownership；
- cache declaration、official timing sample、同步与 warmup/steady contamination 检查；
- VERIFIER/METER/SENTINEL authority 与 producer-role trust floor，Candidate 不能冒充 verifier；
- TrialSeal 绑定 Task/Draft/Artifact/Cache/Resolution/Runner/Build/Verifier/Meter/Lease/Cell/
  Sentinel；
- EvidencePack 和逐 score-component EvidenceRef/authority trace，任意 digest/authority 错配
  fail closed。

CLI：

```text
infraswe artifact lint-policy
infraswe artifact collect
infraswe artifact inspect-manifest
infraswe evidence verify
infraswe evidence trace-score
```

## P0C：Capability、Resource、Topology 与 Cell

已实现：

- versioned capability registry、alias/relationship、parameter constraint 和 CP proof floor；
- `all_of` / `any_of` / `not` / conditional capability expression；unknown 不会被当作
  supported；
- RunnerManifest 与带有效期的 RunnerSnapshot 分离；trusted probe identity、attestation
  digest、higher-grade contradiction 合并与 runner quarantine 状态；
- required-usable、required-native、required-absent 和 forbidden-use 分离；Candidate 声明
  不具有环境事实 authority；
- 多 variant、多 runner 的冻结顺序解析；probe-required、contradiction、candidate-ineligible、
  unschedulable 与 capacity-unavailable 使用不同状态和 exit code；
- ResourceEnvelope 把 minimum/reserved/candidate-limit/measurement-reserve 分开；运行后区分
  Candidate exceed 与 infrastructure under-delivery；
- TopologyGraph/TopologyContract 使用 vertex/relation constraint，不以 GPU count 或 SKU 字符串
  替代 NUMA/P2P/NIC 关系；
- ResourceLease 的 snapshot/critical identity/expiry/digest 审计；
- Candidate 实际 capability-use 观测可发现 required-native silent fallback；
- BenchmarkCell full digest 与 comparison digest 分离，raw performance 跨 Cell 比较 fail closed。

CLI：

```text
infraswe capability registry-validate
infraswe capability resolve
infraswe capability audit-resolution
infraswe cell compare
```

## 负控与当前验证面

新增 22 个 v0.1 专项测试：P0A 6 个、P0B 8 个、P0C 8 个。覆盖至少包括：

- 未映射 requirement、baseline 不呈现 gap、弱 mutation、alternative overfit、flaky verifier、
  缺 human review 与 seal tamper；
- path escape、secret、producer authority spoof、EvidencePack/score binding tamper、完整 Git 状态
  捕获、async timing 污染；
- unknown capability、proof contradiction、capacity/structural unschedulable、topology mismatch、
  lease drift、resource overuse、silent fallback 与 cross-cell raw comparison。

仓库共导出 102 个 fresh JSON Schema。发布前以全量 Ruff、pytest、schema freshness 和 CLI
smoke 重新核验，最终数字以发布 commit 的测试输出为准。

## 尚未实现，不能冒充完成

1. Task authoring UI、VerifierBundle 打包器、自动 mutant 生成器和真实 adversarial rollout；
2. maintainer identity/permission snapshot 的签名验证，以及任何官方 Q9 TaskSeal；
3. 主 Runner 的端到端 container namespace/cgroup/seccomp/read-only mount enforcement；
4. provider-backed lease acquire/heartbeat/release、GPU/NIC 隔离、进程归因与 teardown cleanup；
5. 多节点原子/gang selection 与 lease；当前 resolver 选择单 runner；
6. 可信 inventory/compile/runtime/behavior probe 执行器和硬件签名 attestation；
7. live power/clock/cache/topology drift sentinel 与 runner quarantine 持久化；
8. production object store transport、签名 provenance、retention/GC 与 key management；
9. 把 P0A/P0B/P0C controller 完整接入现有 17 个 task package 的迁移工具；
10. official v0.1 season 的真实 task、witness、runner pool、sealed evidence 与 maintainer review。

这些边界意味着当前实现可用于协议开发、fixture qualification、离线审计和集成适配，不能把
单元测试通过解释成已经部署了可信调度/隔离系统，也不能据此签发官方 Candidate 分数。

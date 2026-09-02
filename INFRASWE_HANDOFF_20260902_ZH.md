# InfraSWE 当前成果与新对话交接

更新时间：2026-09-02 19:27 +08:00
本地工程：`/Users/0z5a/Documents/infra/infraswe`  
状态：v0.1 已发布；历史 PR 盲测已推进到 R12 通信类双卡实测。

## 0. 19:27 当前状态（覆盖本文后续旧快照）

本文件第 7–9 节保留的是旧历史快照，其中“r5 尚未锁 verdict、#53038 未跑、从第 8
节继续”等描述已经失效，不得继续执行。当前权威增量报告与实现状态：

- `results/historical-pr-blind-20260901/supplemental-r7/REPORT_20260902_ZH.md`
- `SYSTEM_PATH_RFC_IMPLEMENTATION_STATUS_20260902_ZH.md`
- `POLARIZED_DECISION_POLICY_V0.5.1_ZH.md`

当前完成情况：

- built-in default Draft 已从 vLLM、SGLang、FlashAttention、FlashInfer 扩展到 CUTLASS/CuTe、Liger-Kernel、DeepGEMM、Megatron-Core、TorchTitan、verl，共 10 个目标项目；
- 候选池仍为 39 项、13 条 ordered first-match 规则、默认只激活 1 个 peer；
- 新 100k 门禁通过：39/1,039 candidates 的 selection median 分别为 8.750/8.625 µs，activation median 为 19.125/19.042 µs；
- r5 已完成 H100 LoRA int64 探针、三项机器 verdict 锁和 review 揭盲；
- r6 已完成 6 个跨项目 PR 的静态/动态离线实测、机器判断锁和锁后揭盲；
- r7 已完成 6 个新跨项目 PR；原锁前结论 4 `revise`、2 `unresolved`，机器 lock
  set 为 `sha256:c0422e6de0fd157036a738bd2c5ff6e5cddbd94af4857476901429314339ca50`；
- R7 锁后按 prospective 极化 oracle 只读重标注为 2 `accept`、4 `reject`、0 `revise`，
  没有改写原锁；
- R8 极化 30-PR 队列作为负面对照保留；R9、R10、R11 分别完成 5、10、20 个新增 PR；
- 新决策统一使用 `check`，历史哈希冻结 artifact 中的 `revise` 拼写不回写；
- R11 repairability triage 在同队列把 exact 从 25% 提高到 50%，但 binary 仍为 55%；
- R12 完成 12 个通信类 PR，exact 从同队列旧路数 25% 提高到 58.3%，binary 均为
  66.7%，机器 reject precision 为 80%；改善只成立于 check/reject 分离；
- R12 已在 `38.49.42.120:54270` 的 `2×A100-SXM4-40GB` 上完成 12 个探针、双 rank
  NCCL 与精确 TorchTitan DDP graph 测试，结果均已同步回本地；Intel XPU/XCCL 仍不在该
  NVIDIA 双卡主机的可执行范围；
- prospective mergeability 已冻结：official ProjectFit `>=85` 才可 accept；`check` 仅限
  30 天内且 14 天内明确在审的新 PR；长期 reviewed-open PR 按 reject；
- precedent retrieval 已具备 footprint、SQLite/FTS/graph、泄漏、冲突、trust、rule、
  PrecedentSet/RetrievalBundle digest 与完整 CLI；
- communication 与 memory-tiering Draft、InfraCert hard gate、单一 C、O=C、cell-local card、
  10+1+5 system profile catalog 已实现；
- 最新核心代码回归：`src tests benchmarks` Ruff 通过，Pytest `280 passed in 5.55s`，
  102 schemas fresh；全仓 Ruff 仍会扫描作为证据保留的上游 LoRA 源码快照并报告其原始 lint。

## 1. 新对话必须继续遵守的约束

1. 文档之间有冲突时，以评分 RFC v0.4 为准。
2. 用户没有设置本地 Draft 或远程 Git Draft 时，才启用 built-in default Draft。
3. 默认 Draft 包含 10 个目标项目；默认候选是分角色的 39 项候选池。
4. 不使用模型训练、学习权重或隐式打分；判断流程必须是人类语言可解释的有序规则。
5. 历史 PR 必须先测试并冻结机器判断，再读取 reviewer/comment 正文。
6. 失败 PR 必须是 closed、未合并、有人类 review、存在明确技术反馈；优先要求真人在最终 head SHA 上给出反馈。
7. 不得用预先挑出的全失败样本计算 outcome accuracy；只能报告 reviewer 概念对应率。
8. 新增 Draft/候选和预编译不能牺牲 benchmark 速度，也不能把 import/build/compile 放进选择或正式计时热路径。
9. official timing 与 profiler timing 分离；3 replay 只做 audit，official 至少 5 次 fresh replay，推荐 7 次。

## 2. 默认 Draft 与候选架构已完成

核心实现：

- `src/infraswe/draft/resolver.py`
  - 解析顺序：`local > remote Git > built-in default`。
- `src/infraswe/draft/defaults.py`
  - vLLM、SGLang、FlashAttention、FlashInfer 与 6 个新增项目的默认项目/合同目录。
  - 默认目录改为一次构建的只读缓存，避免新增 Draft 后重复重建整份合同目录。
- `src/infraswe/draft/extended_default_templates.py`
  - CUTLASS/CuTe、Liger-Kernel、DeepGEMM、Megatron-Core、TorchTitan、verl 的固定 revision profile 与合同模板。
- `src/infraswe/draft/candidate_registry.py`
  - 39 个候选、13 条 ordered first-match 规则。
  - 注册表摘要按不可变对象身份缓存，候选增加不会导致每次解析重新序列化全池。
  - 默认 activation 强制只允许一个已选中的 `peer-impl`。
  - resolution 与 activation 的 registry digest 必须一致。
  - 新增 `evaluate_candidate_timing_gate()`，预编译或 cache artifact 未准备时禁止进入计时。
- `src/infraswe/models/candidates.py`
  - `CandidateActivationPlan.activation_policy = single-explicit-peer-v0.5`。
  - 新增 `CandidateTimingGate`：`blocked / diagnostic-only / official`。
- `src/infraswe/models/draft.py`
  - `DraftPrecompilePolicy.mode = auto|off`，默认 `auto`。
  - `steady_state_compile_allowed` 固定为 `false`。
- `src/infraswe/draft/precompile.py`
  - 不需编译：`skip-no-compilation`。
  - cache hit：`reuse-precompiled-artifact`。
  - auto + miss：`precompile-before-timed-cases`。
  - 显式 off + 必须编译：`compile-inline-with-warning`，只具备 diagnostic 资格。

完整策略文档：

- `catalog/default-candidates-v0.5/ARCHITECTURE_AND_PRECOMPILE_POLICY_zh.md`
- `catalog/default-candidates-v0.5/README.md`
- `catalog/default-candidates-v0.5/registry.json`

## 3. Benchmark 热路径结果

正式结果：

- `results/default-candidate-registry-v01-20260902/selection-speed-100k-r5.json`

运行规模：100,000 次；同时把候选池从 39 项放大到 1,039 项。

| 指标 | 39 项 | 1,039 项 |
|---|---:|---:|
| role resolution median | 8.542 µs | 8.625 µs |
| role resolution p95 | 12.083 µs | 12.000 µs |
| activation plan median | 18.250 µs | 18.208 µs |
| activation plan p95 | 22.292 µs | 22.167 µs |
| 默认激活数 | 1 | 1 |
| selection import | 0 | 0 |
| selection compile | 0 | 0 |

完整默认 Draft resolution：

- cold：2.848 ms；
- steady median：89.709 µs；
- steady p95：100.125 µs。

守门预算全部通过：

- selection p95 `< 1 ms`；
- activation p95 `< 0.5 ms`；
- default Draft p95 `< 3 ms`；
- candidate import 为 0；
- default activation 恰好为 1。

基准脚本：`benchmarks/default_candidates/benchmark_resolution.py`。

## 4. 回归状态

- Ruff：通过。
- Pytest：`195 passed in 4.02s`。
- Schema：41 份，freshness 检查通过。
- 新 schema：`schemas/candidate-timing-gate-v0.5.schema.json`。

如继续修改，最低回归命令：

```bash
cd /Users/0z5a/Documents/infra/infraswe
env PYTHONPATH=src uv run ruff check .
env PYTHONPATH=src uv run pytest -q
env PYTHONPATH=src uv run python -m infraswe schema check --output schemas
```

## 5. 历史 PR 盲测准确率：只能这样报告

首轮混合盲测：

- 总样本：12；
- 实际给出判断：9；
- 判断正确：8；
- 覆盖率：75%；
- 已判断样本准确率：88.9%；
- 若用全部 12 个作分母：66.7%，其中 3 个是主动 abstain，不应记成猜错。

唯一已知误判：vLLM #13043。机器只看到局部 device-context 修复，漏掉了 central `GPUModelRunner` ownership。

失败样本补测不能当 outcome accuracy：

- r2：3 个 verdict 中 2 exact、1 partial；review 概念 exact+partial 为 50%。
- r3：3 个 verdict 方向全部 exact；review 概念 exact+partial 为 57.14%。
- r4：两个 PR 的关闭原因不可归因于技术拒绝，已从严格准确率剔除。
- 严格“最终 head 反馈 + 可归因关闭”的因果样本目前只有 vLLM #11531，样本量不足以计算总体准确率。

现有证据根目录：`results/historical-pr-blind-20260901`。

## 6. 已形成的可解释 agent 规则

实现：`src/infraswe/history/heuristics.py`。

当前规则包括：

- 测试文件目标目录必须符合项目 taxonomy；
- parameter storage dtype 与 compute dtype 分开判断；
- companion bias dtype 必须同步；
- router 改动必须提供 end-to-end 性能与质量证据；
- 最小并发修复要检查后续 ownership；
- 可选能力必须 capability gate，并提供 fallback；
- 必须面向当前 architecture generation/successor；
- 出现竞争修复时确认 canonical owner；
- 不使用模型、训练或权重。

Review finality 审计：

- 模型：`HistoricalReviewFinalityEvidence`；
- 脚本：`benchmarks/historical_prs/audit_review_finality.py`；
- 区分 `final_head_feedback_eligible`、`closure_reason_attributable` 与严格 `calibration_eligible`。

## 7. r5 严格失败样本：当前冻结状态

选择锁：

- `results/historical-pr-blind-20260901/supplemental-r5/selection-lock.json`
- SHA-256：`b4d3bef64cd61f8500124d94acfb75ab12f07ec27670d2805b37481bc1c8603e`

测试计划：

- `results/historical-pr-blind-20260901/supplemental-r5/test-plan.json`
- SHA-256：`c313c1c28bae1ce55861e453457db7841e70c5a5651df9748235d3068e43ba99`

共同资格：closed、未合并、真人在最终 head SHA 上 `CHANGES_REQUESTED`。截至本交接文件生成时，尚未读取这 3 个 PR 的 review/comment 正文。

### 7.1 vLLM #50423：已测，尚未锁机器 verdict

目标：给 `MessageQueue.dequeue()` 加锁，防止多个消费者竞争 reader cursor/socket。

结果文件：

- `results/historical-pr-blind-20260901/supplemental-r5/probes/vllm-pr-50423-dequeue-lock.json`
- SHA-256：`cc96232e845bf998c6f2280cd66365988f3ef426abe67c32604c30b13d35f71f`

独立探针直接执行 base/head 的原始 `dequeue` AST：

- base 最大并发 `recv=2`；head 为 `1`，说明序列化目标确实实现；
- 但调用者传入 100 ms timeout，锁先占用 80 ms 时，head 总等待为 180.21 ms；
- 即锁等待结束后又把完整 100 ms 传给 socket，timeout budget 被重新开始；
- 新增测试只覆盖并发序列化，没有覆盖总 deadline；
- 无竞争微探针约从 172 ns/call 增至 397 ns/call（2.30×，绝对增加约 225 ns）。

按冻结 test-plan，方向应为 `revise`：保留序列化，但改用单一 deadline/remaining timeout。

### 7.2 vLLM #52205：已测，尚未锁机器 verdict

目标：让 AMD Kimi-K3 KDA 在 FULL cudagraph 下也进入 eager segment。

结果文件：

- `results/historical-pr-blind-20260901/supplemental-r5/probes/vllm-pr-52205-always-break.json`
- SHA-256：`ca616f23ef6eef6e23a17893d13e612a44c87c75e558bce52ce873704ddeba8b`

结果：

- legacy FULL 行为保持不 break；
- legacy PIECEWISE 仍 break；
- `always_break=True` 能让 FULL 路径进入 eager segment；
- 只有 AMD KDA call-site opt in，NVIDIA 路径未改变；
- PR 没有改任何测试；全仓没有测试引用 `always_break`；没有 runtime benchmark。

按冻结 test-plan，行为本身正确，但缺少 AMD FULL capture/replay 正确性与性能证据，方向应为 `revise` 或 `accept_with_scope`，需在冻结机器 verdict 时明确二选一。

### 7.3 vLLM #53038：探针已写，GPU 结果未跑

目标：把 LoRA Triton kernel 的 row/lora/slice 索引转为 int64，修复超过 `2**31` 元素的 pointer offset wrap。

探针：`benchmarks/historical_prs/probes/vllm_lora_int64_probe.py`。

待完成：

1. 在 A100/H100 上预编译 base/head standalone Triton arithmetic；
2. 验证 boundary row `174763 × stride 12288` 的 int32 wrap 与 int64 修复；
3. 预编译后跑 7 组 paired steady timing，确认 steady compile event 为 0；
4. 检查常见范围延迟是否保持在 3% 内；
5. 当前 PR 改了普通与 FP8 共 4 个 kernel 文件，但新增大规模测试只显式使用 BF16，应记录 FP8 直接覆盖缺口；
6. 完成后与 #50423/#52205 一起冻结 `machine-judgment-locks.json`，之后才可读取 reviewer 正文。

## 8. r5 下一步严格顺序

```text
重新租用合适 GPU（只为 #53038）
  -> 跑 vllm_lora_int64_probe.py
  -> 立即同步 JSON 回本地
  -> 基于 frozen test-plan 生成三项 machine judgment
  -> 计算 lock-set digest
  -> 确认 lock 已落盘
  -> 才读取 review/comment 正文
  -> 做 reviewer 概念逐条对应
  -> 跑 review finality audit
  -> 更新总体状态 MD
```

不得在 #53038 结果出来前读取任一 r5 reviewer 正文，避免交叉泄漏影响同一批次判断。

## 9. 原 GPU 实例已经释放

原实例：

- Vast instance id：`49539878`
- 原地址：`23.127.144.217:26939`
- GPU：2× NVIDIA A100-SXM4-40GB
- 原 `/workspace`：不是持久卷。

2026-09-02 08:15 +08:00 复核：

```text
vastai show instances --raw | select(id == 49539878) -> []
```

因此该实例已经不存在，不要再尝试连接，也不要对其他实例执行删除。r5 已完成的两个 JSON 已在释放前同步回本地；#53038 明确未跑，不存在只留在远端的结果。

## 10. 关键文件索引

- 总工程：`/Users/0z5a/Documents/infra/infraswe`
- 本交接：`/Users/0z5a/Documents/infra/infraswe/INFRASWE_HANDOFF_20260902_ZH.md`
- Draft/预编译策略：`catalog/default-candidates-v0.5/ARCHITECTURE_AND_PRECOMPILE_POLICY_zh.md`
- 速度结果：`results/default-candidate-registry-v01-20260902/selection-speed-100k-r5.json`
- 首轮 calibration：`results/historical-pr-blind-20260901/revealed/calibration-report.json`
- r2：`results/historical-pr-blind-20260901/supplemental-r2`
- r3：`results/historical-pr-blind-20260901/supplemental-r3`
- r4：`results/historical-pr-blind-20260901/supplemental-r4`
- r5：`results/historical-pr-blind-20260901/supplemental-r5`
- r6：`results/historical-pr-blind-20260901/supplemental-r6`
- r7：`results/historical-pr-blind-20260901/supplemental-r7`
- r8：`results/historical-pr-blind-20260901/supplemental-r8`
- r9：`results/historical-pr-blind-20260901/supplemental-r9`
- r10：`results/historical-pr-blind-20260901/supplemental-r10`
- r11：`results/historical-pr-blind-20260901/supplemental-r11`
- r12：`results/historical-pr-blind-20260901/supplemental-r12`

新对话应先读取第 0 节和 R12 报告；第 7–9 节只作历史记录，不再从旧第 8 节继续。

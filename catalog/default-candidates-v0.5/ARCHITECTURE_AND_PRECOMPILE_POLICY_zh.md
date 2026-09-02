# 默认 Draft 候选、预编译与 Benchmark 热路径策略

状态：`proposed-v0.5`  
适用范围：默认 Draft、候选注册表、架构 adapter、Draft benchmark loop  
冲突优先级：评分、证据与正式 replay 语义以 RFC v0.4 为准。

## 1. 不变量

新增多少 Draft 或候选，都不得改变以下不变量：

1. `local Draft > remote Git Draft > built-in default Draft`。只有前两者都未设置时才启用默认 Draft。
2. 注册表加载和候选选择只读取固定元数据，不导入候选包、不拉仓库、不构建扩展、不探测 GPU。
3. 一个默认 candidate run 只激活一个已选中的 `peer-impl`。其他 peer 必须进入独立 candidate run，不能在同一次默认启动中批量构建。
4. Oracle、Host、Workload 与 Coverage 是引用角色，不能因“被选中”而自动变成待编译 candidate。
5. 编译缓存身份至少包含 source、adapter、compiler、runtime、driver、hardware、workload/probe 与 environment digest；身份不完整时不得复用。
6. `steady-state` 永远不允许编译。若观测到 JIT、recompile、module rebuild 或 cache stampede，则正式性能证据无效，并进入定位流程。
7. Candidate 选择使用按顺序首个匹配的、人类可读规则；不使用模型、训练、学习权重或运行后改权重。

## 2. 控制面与执行面

```text
Draft source resolution
  -> immutable catalog/registry metadata
  -> ordered role resolution
  -> single explicit peer activation plan
  -> precompile/cache preparation
  -> timing gate
  -> cold-start timing
  -> steady-state timing
  -> BenchmarkCost / evidence
```

前三步属于控制面，不能触发 candidate import/build。只有单候选 activation plan 可以进入 adapter 执行面。

注册表摘要按不可变对象身份缓存。默认项目/合同目录也只构建一次。因此，继续增加未命中的候选不会让每次解析重新序列化整个候选池；规则匹配成本只与冻结的匹配链有关。

## 3. 单候选激活

默认值为：

```yaml
activation_policy: single-explicit-peer-v0.5
activated_candidate_ids:
  - <resolution.primary_peer_impl>
```

硬拒绝以下输入：

- 同一默认 run 激活两个或更多 peer；
- 激活不在 resolution 中的候选；
- 把 oracle、workload、coverage 或纯 host 角色当作 candidate 激活；
- 给未激活项提供 compile/cache 状态；
- resolution digest 与 activation registry digest 不一致。

这样，39 项、1,039 项或更大的注册表都不会扩大一次 benchmark 的构建集合。

## 4. 预编译开关

默认策略：

```yaml
mode: auto
trigger: when-compilation-required
cache_policy: content-addressed-evidence-identity
cache_miss_action: precompile-before-timed-cases
timing_phases:
  - precompile
  - cold-start
  - steady-state
steady_state_compile_allowed: false
```

确定性决策表：

| 条件 | 动作 | 正式 timing 资格 |
|---|---|---|
| 不需要编译 | `skip-no-compilation` | 允许 |
| 完整身份 cache hit | `reuse-precompiled-artifact` | 验证 artifact 已准备后允许 |
| `auto` 且 cache miss | `precompile-before-timed-cases` | 预编译成功后允许 |
| 用户显式 `off` 且必须编译 | `compile-inline-with-warning` | 仅诊断，不作为 official timing |

`CandidateTimingGate` 是进入计时区间的硬边界：需要 artifact 的 action 未完成时返回 `blocked`；显式关闭预编译返回 `diagnostic-only`；只有无 blocker、无 warning 时返回 `official`。

## 5. Adapter 嵌合边界

每个 adapter 只能承担四件事：

1. 检查当前 GPU 架构、backend、dtype、layout 与宿主生命周期能力；
2. 构建或加载最终激活 peer 的 artifact；
3. 提供明确的 unsupported/fallback 路径；
4. 报告 compile、module load、cache hit/miss、variant 与 steady compile 事件。

不得把候选注册、import、副作用初始化或全量 AOT 构建放进：

- CLI 模块导入；
- Draft source resolution；
- rule matching；
- benchmark case 枚举；
- baseline/candidate paired timing；
- profiler collector 的计时区间。

可选能力必须先 gate 再调用，并有原生 fallback；架构实现必须面向当前 generation/successor；出现竞争修复时必须确认 canonical owner。这三项沿用可解释 integration preflight。

## 6. Benchmark 速度门限

`benchmarks/default_candidates/benchmark_resolution.py` 同时测量：

- 冷注册表构建；
- 稳态 role resolution；
- 单 peer activation planning；
- 完整默认 Draft resolution；
- 候选模块意外 import；
- 放大注册表后的热路径。

当前守门预算：

```yaml
selection_p95_seconds: 0.001
activation_p95_seconds: 0.0005
default_draft_resolution_p95_seconds: 0.003
candidate_imports_during_selection: 0
default_activated_candidate_count: 1
```

运行方式：

```bash
PYTHONPATH=src uv run python benchmarks/default_candidates/benchmark_resolution.py \
  --iterations 100000 \
  --synthetic-extra-candidates 1000 \
  --enforce-budgets \
  --output results/default-candidate-registry-v01-20260902/selection-speed-100k-r5.json
```

任何预算失败都必须先处理，不能用减少 correctness、fallback、并发或 evidence probe 来换取通过。

## 7. 正式证据语义

- precompile、cold-start、steady-state 分开记录；
- official latency 使用无 profiler 的 paired run；
- profiler/NSYS/NCU 只做路径与因果证据，不混入 authoritative latency；
- 3 replay 只做 audit；official 至少 5 次 fresh replay，推荐 7 次；
- `BenchmarkTrust` 与 `BenchmarkCost` 独立报告，不进入 candidate 主分；
- harness 失败为 `UNRESOLVED`，不能伪装成 candidate 失败。

这保证候选池扩展、预编译优化和架构适配不会通过牺牲 benchmark 正确性或速度获得表面收益。

# InfraSWE GB10 / SM121 RFC 执行与二审报告

- 执行日期：2026-09-01（Asia/Shanghai）
- 远端环境：Vast 容器 `cb50f8c071af`
- 审核结论：`partial`
- 评分权威：InfraSWE scoring second audit RFC v0.4

## 1. 权威与冲突处理

本次执行以以下输入为准：

| 文档 | SHA-256 | 作用 |
|---|---|---|
| `infraswe_scoring_second_audit_v0.4_zh.md` | `e0afe9081cbca2c8cb5e6d94b8a5fef2ba30a3ec093a0abbdf4ba4c427591d79` | 全局评分、硬门槛、证据状态及复跑规则的最终权威 |
| `infraswe_gb10_sm121_architecture_adapter_rfc_v0.1_zh.md` | `5f9bd0bd4afa97c4530c2724dbdc341ee28ba6bbb3e4204c315d9cd8119944be` | GB10/SM121 架构、工具链、原生特性和平台适配约束 |

冲突项统一采用 v0.4：

- 总分仅允许 `Deployability = 100 × C^0.45 × U^0.30 × M^0.25`，并保留 `C≥0.60`、`U≥0.55`、`M≥0.50` 分项地板。
- 正确性、回退、活性和证据完整性是硬门槛。
- GB10 特性覆盖仅进入 cell scorecard，不替代全局 C/U/M，也不引入另一套全局权重。
- 跨硬件绝对时延排名被禁止；效率指标保持 cell-local。
- 正式稳定性采用 7 次独立进程复跑；缺失项标为 `unresolved` 或 `not_applicable`，不写成 0、不重分配权重。

## 2. 实机与工具链

| 项目 | 观测值 |
|---|---|
| CPU/OS | AArch64，Linux 6.17.0-1029-nvidia |
| GPU | NVIDIA GB10，compute capability 12.1 / SM121 |
| 驱动 | 580.173.02 |
| CUDA | 13.0 / nvcc 13.0.88 |
| PTX | 9.0 |
| 目标接受度 | `sm_121`、`sm_121f`、`sm_121a` 均通过 ptxas 编译门 |
| 显存字段 | GB10 UMA 下 `memory.total=[N/A]`，按 v0.4 保留为 `null`，未伪造为 0 |
| GPUDirect RDMA | 不可用；单节点 RoCE 为 `not_applicable`，未混入单节点评分 |

远端原有 CUDA 12.8 不接受 SM121。执行期间仅安装了与驱动上限相容的 CUDA 13.0 编译器、命令行工具和 cudart 开发包；未修改 NVIDIA 驱动。CUDA 13.3/PTX 9.3 是 RFC 的规范工具链，但当前实例不能在不越过驱动上限的前提下使用。

## 3. 最小发布套件结果

| 特性 | 状态 | 独立进程复跑 | 关键证据 |
|---|---:|---:|---|
| `GB10-TARGET-001` | certified | 7 | 原生 `sm_121` PTX/cubin/SASS、运行时派发、禁止回退扫描均通过 |
| `GB10-MMA-001` | unresolved | 0 | 需要 PTX 9.3 的 block-scaled warp MMA；当前仅 PTX 9.0 |
| `GB10-UMA-001` | certified | 7 | system、managed、pinned、GPU-private-copy 路径通过，未从 CPU 解引用 `cudaMalloc` 私有内存 |
| `GB10-MATRIX-IO-001` | unresolved | 0 | 需要 PTX 9.3 的低位矩阵 I/O 指令；当前仅 PTX 9.0 |
| `GB10-ARM-ORDER-001` | certified | 7 | 每次 1,000,000 轮弱内存序压力；AArch64 ELF 与 LSE `ldadd` 静态门通过 |

因此最小发布项为 3/5 已认证，整体状态是 `partial`。两个 PTX 9.3 项保持未决，既未被算作 0 分，也未以通用 CUDA 库或错误架构指令作为回退冒充原生实现。

`GB10-TMA-001` 不属于该 RFC 的 5 项最小发布集合，本轮只完成能力建模，未声明认证。RoCE 需要至少双节点，本单节点执行明确为 N/A。

## 4. v0.4 评分结论

本轮 `deployability_100 = null`。原因是架构特性认证不能代替 v0.4 所要求的跨 cell 并发 C、复用 U、可维护性 M 证据。本轮没有足够数据计算这三项及其分项地板，因此不发布总分。

该 `null` 表示“证据不足，未评分”，不表示 0 分。绝对时延也没有用于跨硬件全局排名。

## 5. 已落地实现

- v0.4 任务、结果、证据、并发、复用、维护性和 cell 效率模型；兼容 0.1/0.3/0.4 输入。
- C/U/M 几何均值、分项地板、复跑要求、复用变体预算、SOL/带宽效率和流量放大计算器。
- E0-E4 分层 profiler 证据采集，区分 authoritative timing 与诊断性 profiled timing。
- 10 份 JSON Schema 的导出与 freshness CI 门。
- GB10 三目标 lane、特性契约、禁止 TMEM/tcgen05 和不透明 GEMM 回退的静态验证。
- SM121 原生派发、UMA 管线、Arm64 弱内存序任务及相应 PTX/cubin/SASS/ELF/runtime 证据。
- GB10 `[N/A]` 统一内存容量解析修复，遵循“缺失不等于零”。

完整可复现源码快照见 `source-snapshot.tar.gz`。

## 6. 质量门与证据完整性

- Ruff：通过。
- Pytest：113 项通过，0 失败。
- Schema freshness：10/10 为 fresh。
- 证据包：27 个实机/QA 文件，加本报告、摘要和源码快照；由相对路径 SHA-256 清单绑定。
- 远端 `/workspace` 是易失目录，证据已同步到本地后再生成可移植哈希清单。

核心证据入口：

- `capability.json`：平台、工具链、目标 lane 和能力指纹。
- `minimum-suite/minimum-suite.json`：5 项最小发布结果。
- `minimum-suite/sm121-native-build/`：PTX、cubin、SASS、动态复跑和哈希绑定。
- `minimum-suite/gb10-uma-001/replays.json`：7 次 UMA 结果。
- `minimum-suite/gb10-arm-order-001/verifier.json`：AArch64、LSE 和动态压力三门结果。
- `qa/pytest-junit.xml`、`qa/ruff.txt`、`qa/schema-check.txt`：质量门证据。
- `SHA256SUMS`：全包完整性清单。

## 7. 完成完整认证所需条件

1. 在驱动与平台允许的实例上使用 CUDA 13.3/PTX 9.3。
2. 提供并验证 `GB10-MMA-001` 的 SM121a block-scaled warp MMA 原生实现。
3. 提供并验证 `GB10-MATRIX-IO-001` 的 SM121f 低位矩阵 I/O 原生实现。
4. 若要发布 v0.4 Deployability 总分，另行采集完整 C/U/M、硬门槛和 cell-local 效率证据；不得用本次 3/5 特性覆盖率代替。

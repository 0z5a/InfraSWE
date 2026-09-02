# InfraSWE Blackwell B200 编译级特性适配 RFC

> **状态**：Draft  
> **版本**：v0.1  
> **日期**：2026-09-01  
> **目标平台**：NVIDIA B200 / Blackwell，Compute Capability 10.0  
> **稳定工具链基线**：CUDA 13.3、PTX ISA 9.3  
> **可选预览赛道**：CUDA 13.4 Developer Preview、PTX ISA 9.4  
> **适用仓库**：InfraSWE benchmark、task packs、runner、native-code verifier

---

## 0. 摘要

InfraSWE 对 B200 的适配不应只是在构建参数中加入 `sm_100`，也不应把“能在 B200 上运行”等同于“使用了 Blackwell 原生能力”。本 RFC 建议新增一组能够区分以下能力的编译与系统任务：

1. **TMEM 与 TCGen05**：Blackwell 第五代 Tensor Core 的主数据通路、资源生命周期和 CTA-pair 协作。
2. **Blackwell TMA 扩展**：`gather4`、`scatter4`、CTA-pair TMA、masked bulk copy。
3. **Cluster Launch Control（CLC）**：硬件辅助的未启动 CTA/cluster 取消与 work stealing。
4. **原生低比特 block scaling**：FP4/FP6/FP8、NVFP4/MXFP4 等数据布局、scale layout 与数值语义。
5. **更宽的 memory I/O 与矩阵装载 lowering**：256-bit load/store、新 `ldmatrix`/`stmatrix` 形状与格式转换。
6. **异步 multimem 与 CUDA Compute Fabric Transport（CFT）**：作为独立的多 GPU / Fabric 扩展包。
7. **PTX 9.4 编译器一致性测试**：作为预览赛道，不计入稳定 B200 主分。

建议最终公开四个独立分数，而不是一个笼统的 “B200 Score”：

```text
SM100-Core Score
SM100-Scheduler Score
SM100-Fabric Score
PTX-Preview Conformance Score
```

首个 MVP 应聚焦单机单 B200 可完成的三类旗舰任务：

```text
TMEM + TCGen05
Cluster Launch Control
TMA gather4/scatter4 + CTA-pair
```

---

## 1. 范围与非目标

### 1.1 本 RFC 要解决的问题

InfraSWE 需要回答的不是：

> “候选补丁是否能在 B200 上编译并跑通？”

而是：

> “候选补丁是否正确、可维护且高效地利用了 Blackwell 暴露的新编译级原语，同时没有通过库回退、死代码注入或特定 shape 过拟合绕过测试？”

因此，任务必须同时覆盖：

- 源码与编译系统修改；
- PTX/SASS 原生指令证据；
- 隐藏输入上的数值正确性；
- 异步流水线与 barrier 的活性；
- 资源生命周期；
- 性能、编译时间和二进制膨胀；
- 单 GPU 与多 GPU 能力探测。

### 1.2 非目标

本 RFC 暂不尝试：

- 用单一 microbenchmark 推断完整 LLM serving 性能；
- 把 NVLink 代际带宽本身作为 compiler feature；
- 要求所有 B200 节点都具有 multicast、logical endpoint 或完整 CFT 拓扑；
- 把 B300、GB300、`sm_103*` 或更后续 Blackwell 变体的特性混入 B200 主赛道；
- 把 CUDA Developer Preview 中尚未稳定的行为纳入主排行榜。

---

## 2. 术语校正：双向 TMA 不是 Blackwell 新特性

Hopper 的 Tensor Memory Accelerator 已经支持：

- global memory → shared memory；
- shared memory → global memory；
- shared memory → distributed shared memory；
- tensor store reduction；
- 多维 tensor copy。

因此，“双向 TMA”不能作为 B200 相比 Hopper 的区分任务。Blackwell 真正值得测的 TMA 增量主要是：

```text
.tile::gather4
.tile::scatter4
.cta_group::1
.cta_group::2
.im2col::w
.im2col::w::128
cp_mask for bulk copies
```

同时必须区分：

```text
TMA       : global / shared / cluster shared 之间的数据搬运
TMEM      : 第五代 Tensor Core 使用的独立片上 Tensor Memory
TCGen05   : 在 TMEM、SMEM 与 Tensor Core 之间执行 copy/MMA/同步的指令族
```

推荐的 Blackwell 主数据通路是：

```text
HBM
 │ TMA
 ▼
Shared Memory
 │ tcgen05.cp
 ▼
Tensor Memory
 │ tcgen05.mma / tcgen05.mma.sp / tcgen05.mma.ws
 ▼
Tensor Memory accumulators
 │ tcgen05.ld
 ▼
Registers / Epilogue
```

---

## 3. Hopper → Blackwell 特性矩阵

| 能力 | Hopper 状态 | B200 / SM100 增量 | InfraSWE 处置 |
|---|---|---|---|
| 双向 TMA | 已支持 | 不是新特性 | 不单独计分 |
| TMA 多维 tile / im2col | 已有基础能力 | `gather4`、`scatter4`、CTA-pair、扩展 im2col 模式 | 主赛道 |
| WGMMA | Hopper 主 Tensor Core 编程路径 | 可作为回退，但不是 Blackwell 原生目标 | native gate 检测回退 |
| TMEM | 无 | 独立片上 Tensor Memory | 旗舰主赛道 |
| TCGen05 | 无 | copy、MMA、稀疏、warp-specialized、同步指令族 | 旗舰主赛道 |
| Cluster Launch Control | 无 | 取消未启动 CTA/cluster 并窃取工作 | 调度主赛道 |
| FP4/FP6 原生 block scale | 非主路径 | TCGen05 原生 mixed-low-precision 路径 | 数值与性能主赛道 |
| 原始 multimem | Hopper 已出现 | 基础 multimem 不是 B200 独占 | 不作为新特性宣传 |
| `multimem.st.async` / `red.async` | 无 | PTX 9.3、SM100+ | 多 GPU 扩展包 |
| CFT fabric instructions | 无 | PTX 9.3、SM100+ | Fabric 扩展包 |
| 256-bit vector load/store | 无 | PTX 8.8、SM100+ | 编译 lowering 次级赛道 |
| PTX 9.4 preview features | 无 | 部分适用于 SM100，但不都为 B200 独占 | 独立预览分，不进主榜 |

---

## 4. 目标架构与工具链分层

### 4.1 编译目标

InfraSWE 应显式区分三种目标：

```text
sm_100   # SM100 基础目标；可能只得到通用实现
sm_100f  # Blackwell family-specific；适合家族内前向兼容任务
sm_100a  # architecture-specific；适合严格 B200 原生任务
```

推荐策略：

```yaml
lanes:
  sm100_generic:
    arch: sm_100
    purpose: 可运行性与通用后端基线

  blackwell_family:
    arch: sm_100f
    purpose: Blackwell 家族原生 lowering

  strict_b200_native:
    arch: sm_100a
    purpose: 精确架构原语与严格 native gate
```

不能允许 candidate 只用 `sm_100` 产生通用回退实现，然后因为“程序运行在 B200 上”而获得 Blackwell-native 分数。

### 4.2 稳定与预览工具链

```yaml
stable_lane:
  cuda: "13.3"
  ptx: "9.3"
  leaderboard: true

preview_lane:
  cuda: "13.4 Developer Preview"
  ptx: "9.4"
  leaderboard: false
  score_namespace: "preview_conformance"
```

预览赛道必须单独记录：

- `nvcc --version`；
- `ptxas --version`；
- driver 版本；
- PTX ISA 版本；
- target；
- cubin hash；
- container image digest。

---

## 5. Benchmark Pack 总体设计

```text
packs/
├── blackwell-core-tmem/
├── blackwell-tma-irregular/
├── blackwell-scheduler-clc/
├── blackwell-lowp/
├── blackwell-memory-io/
├── blackwell-multigpu-multimem/
├── blackwell-fabric-cft/
└── ptx94-preview-conformance/
```

### 5.1 分数命名空间

| 分数 | 包含内容 | 典型硬件要求 |
|---|---|---|
| `SM100-Core` | TMEM、TCGen05、TMA 扩展、低比特、memory I/O | 单 B200 |
| `SM100-Scheduler` | CLC、persistent work queue、ragged scheduling | 单 B200 |
| `SM100-Fabric` | async multimem、CFT、memory-ordering | 多 B200 + 对应拓扑 |
| `PTX-Preview` | PTX 9.4 预览 lowering 与语义一致性 | 单 B200，预览工具链 |

`SM100-Fabric` 不应成为总榜的必选项。缺少拓扑时标记 `N/A`，不得记为 0 分。

---

# 6. Blackwell Core：TMEM 与 TCGen05

## 6.1 为什么这是旗舰任务

Blackwell 的关键变化不是“更多 Tensor Core 峰值”，而是编译器和 kernel author 必须管理新的 address space、资源分配、CTA-group 一致性、异步完成关系和 accumulator 数据流。

TMEM 在 SM100 上可视为每 CTA 的二维 Tensor Memory 空间。其分配具有明确粒度和生命周期要求：

- 动态分配；
- 按 32 columns 的粒度；
- column 数量为 2 的幂；
- kernel 退出前显式释放；
- CTA-pair 模式下需要跨 peer CTA 保持一致；
- 同一 kernel 中所有 `tcgen05` 指令必须使用相同 CTA-group 模式。

相关指令族包括：

```text
tcgen05.alloc
tcgen05.dealloc
tcgen05.relinquish_alloc_permit

tcgen05.cp
tcgen05.ld
tcgen05.st
tcgen05.shift

tcgen05.mma
tcgen05.mma.sp
tcgen05.mma.ws
tcgen05.mma.ws.sp

tcgen05.commit
tcgen05.wait
tcgen05.fence
```

## 6.2 任务 BW-TMEM-001：Hopper 风格 mainloop → TCGen05/TMEM

### 任务目标

给定一个使用 SMEM + register accumulator，或使用 WGMMA 风格的 GEMM/attention 子内核，要求 candidate：

1. 引入 TMEM 分配；
2. 使用 `tcgen05.cp` 将 operand 或 metadata 搬入 TMEM；
3. 使用 `tcgen05.mma*` 执行 mainloop；
4. 从 TMEM 读取 accumulator 完成 epilogue；
5. 正确释放 TMEM；
6. 保持对尾块和动态 shape 的正确性。

### 公开输入

```text
M,N,K ∈ 常见对齐尺寸
FP16/BF16 input
FP32 accumulation
单 CTA-group
```

### 隐藏输入

```text
M/N/K 非 tile 整倍数
K-tail = 1..tile_k-1
空 batch / 极小 tile
非默认 leading dimension
有意制造 register pressure
多次 barrier phase 复用
不同 epilogue：bias、ReLU、SiLU、residual
```

### Native gate

至少满足：

- 存在可达的 `tcgen05.alloc`；
- 存在可达的 `tcgen05.mma*`；
- 存在与分配配对的 `tcgen05.dealloc`；
- mainloop 不得调用 cuBLAS/cuBLASLt；
- 不得仅在永不执行的分支中插入目标指令；
- 若任务声明 strict native，不得回退为 WGMMA 主路径。

### 失败模式

```text
TMEM 泄漏
错误的 column allocation 粒度
barrier phase 重用错误
SMEM→TMEM copy 尚未完成即进入 MMA
accumulator layout 与 epilogue layout 不匹配
仅对公开 shape 正确
死代码注入 tcgen05 指令骗过静态检查
```

## 6.3 任务 BW-TMEM-002：CTA-pair `cta_group::2`

### 任务目标

将单 CTA TCGen05 kernel 改写为 CTA-pair 协作实现，使一条 TCGen05 操作作用于当前 CTA 与 peer CTA 的 Tensor Memory。

### 必测不变量

```text
所有 tcgen05 指令的 cta_group 一致
两个 CTA 对 TMEM allocation 的视图一致
peer CTA 均能到达协作点
cluster shape 与 CTA-pair 映射合法
任一 CTA 提前退出不会导致另一 CTA 永久等待
```

### 隐藏测试

- cluster 中存在额外 CTA；
- CTA-pair 工作量不对称；
- 尾 tile 仅对其中一个 CTA 有有效元素；
- launch shape 不满足假设时，应显式拒绝或安全回退；
- watchdog 检测因 peer 缺席导致的死锁。

## 6.4 任务 BW-TMEM-003：TMEM 生命周期修复

这是一个更偏 SWE 的 bug-fix 任务，而非从零写 kernel。

### 初始缺陷样例

- 某错误路径未 `dealloc`；
- allocation permit 没有 relinquish；
- 资源大小由 host 与 device 两处分别计算，发生不一致；
- CTA-pair 中只有一个 CTA 释放；
- early return 绕过 cleanup；
- barrier 资源与 TMEM 资源析构顺序错误。

### 评分重点

```text
静态资源路径覆盖
错误分支正确清理
无死锁
不降低 fast path 性能
补充回归测试质量
```

---

# 7. Blackwell TMA 扩展

## 7.1 任务 BW-TMA-001：`gather4` / `scatter4`

### 目标 workload

以下 workload 都具有天然的非连续二维 row 搬运：

- MoE token → expert gather；
- expert output → token scatter；
- paged KV block gather；
- ragged batch compaction；
- 稀疏 embedding row fetch；
- diffusion patch gather；
- LoRA row expansion；
- block-sparse attention packing。

### 任务要求

把四次独立 copy 或 scalar gather/scatter 改写为：

```text
.tile::gather4
.tile::scatter4
```

### 隐藏输入

```text
四行连续
四行完全离散
重复 row index
row 顺序逆序
极短 row
非对齐 row width
边界坐标与 OOB 行为
gather 后直接进入 TCGen05
scatter 作为 epilogue
```

### 性能指标

- 每有效字节的 copy latency；
- instruction count；
- memory transaction 数；
- producer/consumer stall；
- 与四次普通 TMA 的相对加速；
- 对短 row 的启动开销。

## 7.2 任务 BW-TMA-002：CTA-pair TMA

要求 candidate 使用：

```text
cp.async.bulk.tensor....cta_group::2
```

完成 CTA-pair 协作加载，并把完成机制正确接入 mbarrier 或下游 TCGen05 pipeline。

### 隐藏测试重点

- 两个 CTA 对 destination region 的归属；
- cluster rank 与 peer rank 计算；
- barrier expected transaction bytes；
- phase wraparound；
- 一个 CTA 没有有效输出时仍参与必要同步；
- launch configuration 与 cluster shape 的一致性。

## 7.3 任务 BW-TMA-003：`cp_mask` 尾块写回

要求候选补丁使用 bulk copy 的 mask 能力处理 16-byte chunk 内的字节级有效范围，替换 scalar tail loop。

### 隐藏输入

```text
valid bytes = 0..16
跨多个 16B chunk
首地址非自然对齐
尾块包含 NaN payload
masked-out byte 预填 poison 值
多线程重叠写保护
```

### Correctness oracle

masked-out byte 必须保持原值；不能仅验证有效元素相等。

---

# 8. Cluster Launch Control 调度赛道

## 8.1 适用场景

CLC 允许正在执行的 CTA/cluster 尝试取消尚未启动的 CTA/cluster，并取得其 block index，从而执行硬件辅助 work stealing。它特别适合：

```text
grouped GEMM
MoE expert workload
变长 attention
稀疏 tile
persistent kernel
Stream-K
heavy-tail kernel
```

## 8.2 任务 BW-CLC-001：Persistent Grouped GEMM

### 初始实现

静态 block → group 映射，短 group 提前结束，尾部长 group 造成 SM 空转。

### 目标实现

worker 完成原任务后：

1. 提交 cancel request；
2. 通过 mbarrier 等待结果；
3. 查询取消是否成功；
4. 成功时获取被取消工作项的 block index；
5. 执行该工作项；
6. 失败后按规范结束，不再提交非法请求。

### 隐藏 workload

```text
[1, 1, 1, 2, 3, 5, 8, 13, 64, 256] tiles
Zipf / Pareto 长尾
随机排列
大量零工作量 group
最后一个 group 极大
均匀分布作为反例
```

### 关键指标

```text
total makespan
p50/p95/p99 CTA completion time
SM idle tail
work imbalance
cancellation success rate
stolen tile count
barrier wait time
```

不应仅以平均 TFLOPS 评分，因为 CLC 的主要价值常体现在尾延迟和 makespan。

## 8.3 任务 BW-CLC-002：CLC 状态机修复

内置若干规范级缺陷：

- 已观察一次 cancellation failure 后继续提交 request；
- 在 request 失败时仍读取 block index；
- request storage 生命周期不足；
- mbarrier phase 使用错误；
- cluster cancel 与 block cancel 混用；
- host 端 launch 配置与 device 假设不一致。

Verifier 必须包含：

```text
correctness
liveness watchdog
Compute Sanitizer 可用检查
多次 launch 稳定性
异常输入下的显式错误
```

---

# 9. 原生低比特与 Block Scaling

## 9.1 覆盖格式

建议最少覆盖：

```text
E2M1   # FP4
E2M3   # FP6
E3M2   # FP6
E4M3   # FP8
E5M2   # FP8
UE8M0  # scale
NVFP4
MXFP4
MXFP6
MXFP8
```

TCGen05 相关 kind / scale 路径可围绕下列形式设计：

```text
.kind::mxf8f6f4
.kind::mxf4
.kind::mxf4nvf4
.scale_vec::1X
.scale_vec::2X
.scale_vec::4X
.block16
.block32
```

## 9.2 任务 BW-LP-001：原生 block-scaled MMA

### 任务目标

从“先反量化到 FP16，再执行普通 MMA”的实现，迁移到原生 block-scaled TCGen05 路径。

### 必测语义

```text
scale block 边界
A/B scale 的 layout
K-tail 与 scale-tail 不一致
accumulation precision
saturation
round-to-nearest / stochastic rounding
NaN / Inf / subnormal
signed zero
```

### 隐藏数据集

- 随机正态；
- 长尾大幅值；
- 全零 block；
- 单个异常值；
- scale 非 2 的幂；
- saturation boundary 前后相邻值；
- 不同 seed 下 stochastic rounding；
- 重复 seed 可重放性。

### 评分

不能只使用一个 `rtol/atol`。建议同时记录：

```text
max absolute error
max relative error
mean relative error
ULP-like bucket error
saturation mismatch count
NaN/Inf classification mismatch
throughput
native block-scale evidence
```

## 9.3 任务 BW-LP-002：`tcgen05.cp` 子字节展开

要求候选实现利用 SMEM→TMEM copy 过程中的 4-bit/6-bit 到 8-bit container 展开，移除独立 unpack kernel 或 scalar unpack loop。

### 隐藏测试

```text
odd element count
misaligned packed input
cross-byte nibble boundary
signed / unsigned interpretation
padding bits 非零
4-bit 与 6-bit 混合 case
```

---

# 10. Memory I/O 与后端 Lowering

## 10.1 任务 BW-MEM-001：256-bit load/store combine

SM100 支持更宽的 vector memory operation，可构造一个 compiler/backend 修复任务：

- 初始 IR 中存在连续 8×32-bit 或 4×64-bit load/store；
- 当前 lowering 因 alignment analysis 过度保守而未合并；
- candidate 修复 legality 与 alignment 推导；
- hidden tests 覆盖 alias、volatile、predicate、尾部和非对齐地址。

### Native gate

验证最终 PTX/SASS 使用目标宽度的 load/store，而不是仅在源代码中出现向量类型。

### 性能指标

```text
instruction count
issued memory instructions
L1/L2 throughput
latency
register pressure
binary size
```

## 10.2 任务 BW-MEM-002：新 `ldmatrix` / `stmatrix` 形状

候选补丁修复矩阵 tile 的 shared-memory layout 与 matrix load/store lowering，覆盖：

```text
.m16n16
.m8n16
.b8
source / destination format conversion
```

该类任务更适合作为编译器单元测试和 codegen conformance，不应获得与 TMEM/TCGen05 相同权重。

---

# 11. Async Multimem 多 GPU 扩展包

## 11.1 范围

严格意义上比 Hopper 初始 multimem 更新、且适合 B200 的 PTX 9.3 原语包括：

```text
multimem.st.async
multimem.red.async
```

它们适合构造以下任务：

- replicated completion flag；
- multicast metadata publication；
- 多 GPU epoch/version 更新；
- 小粒度 distributed counter；
- release/acquire ordering；
- data-ready 与 signal-ready 竞态。

基础 `multimem` 或较早出现的 bulk reduce 不应被错误宣传为 B200 独占。

## 11.2 任务 BW-MM-001：Release/Acquire Litmus

### 协议

```text
GPU A:
  write payload
  release via async multimem signal

GPU B...N:
  acquire/read signal
  read payload
  verify visibility
```

### 隐藏测试

```text
signal 先于 payload 的错误重排
多个 producer
同一 signal address 重用
epoch wraparound
不同 scope
异步完成尚未观察即复用 storage
进程重复启动
```

### 评分原则

memory-model correctness 是硬门；带宽与延迟只在硬门通过后计分。

## 11.3 任务 BW-MM-002：异步 replicated reduction

- 用 `multimem.red.async` 更新 replicated counter；
- 检查 reduction op、scope、completion 和可见性；
- 对比同步版本；
- 隐藏测试故意制造高竞争和短生命周期 buffer。

---

# 12. CUDA Compute Fabric Transport 扩展包

## 12.1 原语

PTX 9.3 的设备端 fabric 指令族包括：

```text
fabric.try_get
fabric.try_put
fabric.try_red
fabric.try_pullred
fabric.submit
fabric.wait
```

与普通 pointer-based remote access 不同，这类操作围绕 logical endpoint、resource offset、异步提交、完成状态和错误报告构建。

## 12.2 为什么适合 InfraSWE

CFT 任务同时要求修改：

```text
host resource setup
endpoint/capability binding
device-side PTX lowering
mbarrier completion
error reporting
retry/cleanup state machine
multi-GPU topology detection
```

这比单纯 kernel microbenchmark 更接近 InfraSWE 的定位。

## 12.3 任务 BW-CFT-001：Failure-aware Unicast Copy

### 目标

实现：

```text
remote resource → local shared memory
local shared memory → remote resource
```

并正确使用 submit/wait 与 mbarrier report。

### 隐藏故障注入

```text
endpoint 未 ready
resource offset 越界
capability 不匹配
遗漏 fabric.submit
kernel exit 前仍有 in-flight op
多个 operation 共用同一 mbarrier
report predicate/value 解析错误
retry 导致重复写
```

## 12.4 任务 BW-CFT-002：Multicast Pull Reduction

要求使用 `try_pullred` 或等价 CFT reduction 路径完成 failure-aware 多目标规约。

### 主要指标

```text
correctness under partial failure
error classification
retry idempotency
completion latency
small-message throughput
resource cleanup
```

## 12.5 能力探测规则

B200 型号本身不能保证节点已配置相应 logical endpoint / multicast / Fabric 能力。runner 必须先探测：

- GPU compute capability；
- GPU count；
- peer access；
- NVLink / NVSwitch topology；
- multicast support；
- fabric handle support；
- logical endpoint support；
- counted operations support；
- driver/toolkit compatibility。

状态定义：

```text
SUPPORTED      能力满足，执行并计分
UNSUPPORTED    硬件或驱动不支持，记 N/A
MISCONFIGURED  宣称支持但环境配置错误，单独报告
FAILED         能力满足但 candidate 执行失败，记 0
```

---

# 13. PTX 9.4 Preview Conformance

此包不计入稳定 B200 主榜。它用于提前发现编译器、assembler、IR backend 与新 PTX 语义之间的问题。

可选测试包括：

```text
.minperctamemory
%perctamemoryoffset
%perctamemorysize
ld.proxy::readonly
prefetch.valid_addr
prefetch.L1::32B
.pzo conversion modifiers
ldmatrix.m8n16.s8.s4
```

注意：这些特性并非全部为 B200 独占；部分仅是 PTX 9.4 新语法，目标架构可以早于 SM100。因此包名应使用：

```text
ptx94-preview-conformance
```

而不是：

```text
b200-new-features
```

---

# 14. InfraSWE 任务类型

为避免退化成纯 kernel benchmark，每个 pack 应混合三类任务。

## 14.1 Kernel Authoring

示例：

- 从 scalar copy 改为 TMA gather4；
- 从 SMEM/register MMA 改为 TCGen05/TMEM；
- 添加 block-scaled FP4 mainloop。

评分偏重：native code、数值、性能。

## 14.2 Compiler / Backend Lowering

示例：

- 修复某 IR 到 `tcgen05.mma` 的 lowering；
- 让合法的连续 load 合并为 256-bit operation；
- 修复 `ldmatrix` shape/layout 推导；
- 修复 address-space cast 或 barrier token type。

评分偏重：测试覆盖、IR legality、codegen、编译时间。

## 14.3 Runtime / Scheduler / Fabric

示例：

- CLC work-stealing 状态机；
- host 端 cluster launch 配置；
- Fabric endpoint setup；
- capability detection；
- async completion 与 teardown。

评分偏重：liveness、异常路径、可观测性、拓扑兼容。

---

# 15. 四层验证门

## 15.1 Gate 1：Build Gate

必须记录并固定：

```text
compiler version
CUDA toolkit
PTX ISA
GPU target
link mode
third-party library versions
build flags
source revision
container digest
```

失败时该任务总分为 0。

## 15.2 Gate 2：Correctness 与 Liveness

### Correctness

- 公开与隐藏 shape；
- 随机种子；
- 尾块；
- 非对齐；
- NaN/Inf；
- poison padding；
- 多轮运行；
- CPU 或高精度 reference；
- race-sensitive 重放。

### Liveness

- kernel watchdog；
- host timeout；
- barrier phase wraparound；
- peer CTA 缺席；
- fabric failure；
- cancellation failure；
- 资源 teardown。

任何死锁、hang 或错误结果都会关闭性能得分。

## 15.3 Gate 3：Native Evidence

### 静态证据

保留：

```text
PTX
cubin
fatbin metadata
cuobjdump output
nvdisasm output
link map
```

为每个 toolkit 维护版本化 matcher，例如：

```yaml
native_requirements:
  require_any:
    - "tcgen05.mma"
    - "TCGEN05.*MMA"
  require_all:
    - "tcgen05.alloc"
    - "tcgen05.dealloc"
  forbid_mainloop:
    - "wgmma"
  forbid_symbols:
    - "cublas"
    - "cublasLt"
```

### 防死代码注入

仅正则搜索指令不够。Verifier 应结合：

1. CFG reachability；
2. kernel entry 到目标指令的可达路径；
3. branch predicate 的动态覆盖；
4. PC sampling / profiler 证据；
5. 对目标代码块做 mutation，确认其参与结果计算；
6. 必要时注入输入，使目标路径必然执行。

### 库回退策略

建议按任务声明：

```yaml
library_policy:
  runtime_blackbox_calls: forbidden
  cutlass_source_substrate: allowed
  cute_dsl: allowed
  precompiled_vendor_kernel: forbidden
```

CUTLASS/CuTe 可以作为源码层 substrate，但不能让 candidate 用一个预编译 vendor kernel 替代题目要求的实现。

## 15.4 Gate 4：Performance Meter

建议记录：

```text
p50/p95/p99 latency
throughput
total makespan
occupancy
registers/thread
shared memory/CTA
cluster occupancy
memory throughput
barrier stalls
instruction count
compile time
binary size
```

性能比较必须：

- 固定 clocks 或至少记录 clocks；
- 包含 warmup；
- 使用多个随机执行顺序；
- 报告置信区间；
- 对短任务使用 CUDA Graph 或批量 launch 降低 host noise；
- 避免只报告最佳一次结果。

---

# 16. 评分模型

## 16.1 单任务得分

建议：

```text
S_task = G × (0.45 C + 0.20 N + 0.25 P + 0.05 R + 0.05 B)
```

其中：

```text
G = hard gate，Build/Correctness/Liveness/必要 native evidence 任一失败则为 0
C = correctness breadth
N = native-feature quality
P = normalized performance
R = robustness / error-path quality
B = build quality：compile time、binary size、可维护性
```

对于纯 bug-fix 任务，可调整为：

```text
0.55 C + 0.15 N + 0.10 P + 0.15 R + 0.05 B
```

对于 performance migration 任务，可调整为：

```text
0.35 C + 0.25 N + 0.35 P + 0.03 R + 0.02 B
```

## 16.2 性能归一化

不建议直接使用线性 `candidate/reference`，因为不同任务尺度差异过大。可采用 portable baseline 与 curated native reference 间的对数插值：

```text
P = clamp(
      (log(T_candidate) - log(T_portable)) /
      (log(T_native_ref) - log(T_portable)),
      0,
      1.2
    )
```

其中吞吐越高越好。延迟任务使用倒数或交换方向。

允许超过 reference，最高可取 1.2，以奖励新实现；公开总分时再裁剪或保留 bonus。

## 16.3 Pack 聚合

使用加权几何平均，避免单个极强任务掩盖某一类别完全失效：

```text
S_pack = exp( Σ w_i log(max(S_i, ε)) / Σ w_i )
```

建议分数：

```text
SM100-Core
  40% TMEM/TCGen05
  20% Blackwell TMA
  25% low precision
  15% memory/backend lowering

SM100-Scheduler
  70% CLC correctness + makespan
  30% robustness + liveness

SM100-Fabric
  45% async multimem
  45% CFT
  10% capability/error handling
```

不建议把 Fabric 分数混入单 GPU 总榜。

---

# 17. Hidden Test 设计

## 17.1 Shape 与布局

```text
0 / 1 / prime-size dimensions
tile-1 / tile / tile+1
非默认 stride
转置与非转置
非对齐地址
大 leading dimension
batch stride overflow 边界
```

## 17.2 异步与 Barrier

```text
phase 多次翻转
producer 比 consumer 快
consumer 比 producer 快
CTA-pair 工作量不对称
一个 CTA 仅有 padding
cluster 中有额外 CTA
连续数千次 launch
```

## 17.3 数值

```text
±0
subnormal
max finite
Inf
NaN 与不同 payload
saturation boundary
异常 scale
随机和结构化输入
```

## 17.4 调度

```text
均匀 workload
Pareto / Zipf 长尾
最后一个任务极大
大量空任务
随机任务排列
取消连续成功
取消立即失败
```

## 17.5 多 GPU / Fabric

```text
单 GPU：应报告 unsupported 而非 crash
peer access 部分可用
multicast 不可用
endpoint 未 ready
错误 capability
operation in flight 时 teardown
重复 retry
GPU reset 后重新初始化
```

---

# 18. 能力探测 Manifest

建议每次运行先生成：

```json
{
  "gpu": {
    "name": "NVIDIA B200",
    "compute_capability": "10.0",
    "count": 1,
    "sm_count": null
  },
  "toolchain": {
    "cuda": "13.3",
    "ptx": "9.3",
    "driver": null,
    "target": "sm_100a"
  },
  "features": {
    "tmem": true,
    "tcgen05": true,
    "tma_gather4": true,
    "tma_cta_group_2": true,
    "clc": true,
    "multicast": false,
    "fabric_handle": false,
    "logical_endpoint": false,
    "counted_fabric_ops": false
  },
  "topology": {
    "nvlink": null,
    "nvswitch": null,
    "peer_matrix": null
  }
}
```

`null` 表示尚未探测，不得等同于 `false`。

---

# 19. 示例任务 Manifest

```yaml
id: BW-TMEM-001
name: Migrate tiled GEMM to TCGen05 and TMEM
pack: blackwell-core-tmem
track: kernel_authoring
status: draft

hardware:
  min_compute_capability: "10.0"
  min_gpu_count: 1
  required_features:
    - tmem
    - tcgen05

toolchain:
  cuda: ">=13.3,<13.4"
  ptx: "9.3"
  target: sm_100a

repository:
  base_commit: "<pinned-sha>"
  editable_paths:
    - src/kernels/gemm_sm100.cu
    - src/kernels/gemm_sm100.hpp
    - tests/test_gemm_sm100.py
  forbidden_paths:
    - verifier/
    - hidden_tests/

build:
  command: "cmake --build build -j"
  timeout_sec: 900
  retain_artifacts:
    - "**/*.ptx"
    - "**/*.cubin"
    - "**/*.fatbin"

correctness:
  public_cases: public_cases.json
  hidden_case_generator: hidden/gemm_cases.py
  reference: torch_fp64
  atol: 1.0e-3
  rtol: 1.0e-3
  watchdog_sec: 30

native_gate:
  require_reachable:
    - tcgen05.alloc
    - tcgen05.mma
    - tcgen05.dealloc
  require_dynamic_execution:
    - tcgen05.mma
  forbid_mainloop:
    - wgmma
  forbid_dynamic_symbols:
    - cublasGemmEx
    - cublasLtMatmul

performance:
  metric: throughput_tflops
  warmup: 20
  repetitions: 100
  aggregate: median
  portable_baseline: artifacts/reference/portable.json
  native_reference: artifacts/reference/native.json

score:
  weights:
    correctness: 0.45
    native: 0.20
    performance: 0.25
    robustness: 0.05
    build_quality: 0.05
```

---

# 20. 推荐仓库结构

```text
infraswe/
├── packs/
│   ├── blackwell-core-tmem/
│   │   ├── BW-TMEM-001/
│   │   │   ├── task.yaml
│   │   │   ├── prompt.md
│   │   │   ├── base.patch
│   │   │   ├── public_tests/
│   │   │   └── reference_metadata/
│   │   └── ...
│   ├── blackwell-tma-irregular/
│   ├── blackwell-scheduler-clc/
│   ├── blackwell-lowp/
│   ├── blackwell-memory-io/
│   ├── blackwell-multigpu-multimem/
│   ├── blackwell-fabric-cft/
│   └── ptx94-preview-conformance/
├── runner/
│   ├── capability_probe/
│   ├── topology_probe/
│   ├── build/
│   ├── watchdog/
│   └── metrics/
├── verifier/
│   ├── ptx/
│   ├── sass/
│   ├── cfg/
│   ├── dynamic_native/
│   ├── numerics/
│   └── memory_model/
├── schemas/
│   ├── task.schema.json
│   ├── capability.schema.json
│   └── result.schema.json
└── docs/
    ├── blackwell_b200_rfc.md
    ├── native_evidence.md
    └── leaderboard_policy.md
```

---

# 21. MVP 路线

## Phase 0：基础设施

目标：先建立可复用的 native-feature verifier。

- [ ] 固定 CUDA 13.3 / PTX 9.3 container；
- [ ] 支持 `sm_100` / `sm_100f` / `sm_100a`；
- [ ] 自动保留 PTX/cubin/fatbin；
- [ ] 集成 `cuobjdump` 与 `nvdisasm`；
- [ ] 建立版本化 opcode matcher；
- [ ] 增加 kernel watchdog；
- [ ] 生成 capability manifest；
- [ ] 对 dead-code instruction injection 做最小防护。

## Phase 1：单 B200 旗舰包

优先级从高到低：

1. `BW-TMEM-001`：TCGen05/TMEM pipeline；
2. `BW-CLC-001`：persistent grouped scheduling；
3. `BW-TMA-001`：gather4/scatter4；
4. `BW-TMEM-003`：TMEM 生命周期 bug-fix；
5. `BW-TMA-002`：CTA-pair TMA。

Phase 1 完成标准：

- 至少 5 个任务；
- 每个任务有公开与隐藏测试；
- native gate 能防最朴素的 fallback；
- 结果可复现；
- 输出 `SM100-Core` 与 `SM100-Scheduler` 两个分数。

## Phase 2：低比特与后端

- [ ] NVFP4/MXFP4 block-scaled MMA；
- [ ] `tcgen05.cp` 子字节展开；
- [ ] CTA-pair TCGen05；
- [ ] 256-bit load/store combine；
- [ ] 新 `ldmatrix`/`stmatrix` lowering；
- [ ] 数值误差报告器。

## Phase 3：多 GPU

- [ ] async multimem memory-order litmus；
- [ ] replicated reduction；
- [ ] Fabric logical endpoint probe；
- [ ] CFT unicast copy；
- [ ] CFT pull reduction；
- [ ] error injection 与 cleanup；
- [ ] 独立 `SM100-Fabric` 排行榜。

## Phase 4：Preview

- [ ] CUDA 13.4 Developer Preview image；
- [ ] PTX 9.4 syntax/codegen tests；
- [ ] 与稳定 lane 完全隔离；
- [ ] 不影响主榜 badge；
- [ ] 工具链升级后重新审计并迁移成熟任务。

---

# 22. 推荐首批任务清单

| ID | 任务 | 类型 | 硬件 | 优先级 |
|---|---|---|---|---|
| `BW-TMEM-001` | GEMM mainloop 迁移到 TMEM/TCGen05 | Kernel | 1×B200 | P0 |
| `BW-TMEM-003` | 修复 TMEM 生命周期与 early return | SWE bug-fix | 1×B200 | P0 |
| `BW-CLC-001` | Persistent grouped GEMM + CLC | Runtime/Kernel | 1×B200 | P0 |
| `BW-TMA-001` | MoE gather/scatter4 | Kernel | 1×B200 | P0 |
| `BW-TMA-002` | CTA-pair TMA pipeline | Kernel | 1×B200 | P1 |
| `BW-TMEM-002` | CTA-pair TCGen05 | Kernel | 1×B200 | P1 |
| `BW-LP-001` | NVFP4/MXFP4 block-scaled MMA | Kernel/Numerics | 1×B200 | P1 |
| `BW-MEM-001` | 256-bit load/store lowering | Compiler | 1×B200 | P1 |
| `BW-MM-001` | async multimem ordering | Multi-GPU | 多 B200 | P2 |
| `BW-CFT-001` | failure-aware fabric copy | Runtime/Fabric | 多 B200 + Fabric | P2 |
| `PTX94-001` | readonly proxy / prefetch lowering | Compiler Preview | 1×B200 | P3 |

---

# 23. 容易误标的特性

## 23.1 不应标记为 B200 新特性

```text
双向 TMA
基础 TMA
基础 multimem
Programmatic Dependent Launch
普通 cluster launch
普通 FP8 Tensor Core
```

这些能力全部或部分在 Hopper 已经存在。

## 23.2 不应纳入 B200 / SM100 主包

```text
tcgen05.ld.red              # 目标为后续 Blackwell 变体，不是通用 sm_100 能力
96-byte swizzle             # 对应后续 target，不应混入 B200
sm_103* 专属语义            # B300/GB300 等后续变体
未稳定 PTX 9.4 行为         # 只进入 preview lane
```

## 23.3 B200 不等于具备完整 Fabric

同为 B200 的不同机器可能在以下方面不同：

- 单卡或多卡；
- PCIe / NVLink；
- 是否有 NVSwitch；
- multicast 是否启用；
- Fabric handle 是否可用；
- logical endpoint 是否配置；
- driver/toolkit 是否支持 CFT。

因此只能按 capability manifest 分流，不能按 GPU 型号硬编码。

---

# 24. InfraSWE 的差异化价值

传统 kernel benchmark 通常只回答：

```text
某个固定 shape 的吞吐是多少？
```

InfraSWE 的 Blackwell 适配应进一步回答：

```text
Agent 能否识别新的 address space？
能否重构异步 pipeline，而不是只换 intrinsic？
能否处理 CTA-pair 与 cluster launch contract？
能否证明 native instruction 真正执行？
能否在隐藏尾块和错误路径保持正确？
能否修复 resource lifetime 与 liveness bug？
能否处理拓扑依赖和 unsupported 状态？
能否在性能提升同时控制编译时间和二进制膨胀？
```

这使 InfraSWE 不再是另一个固定算子性能榜，而成为：

> **面向异构 AI Infra 的真实代码库修改、原生后端迁移、运行时状态机与硬件语义验证 benchmark。**

---

# 25. 验收标准

B200 v0.1 适配可视为完成，当且仅当：

- [ ] 存在稳定的 CUDA 13.3 / PTX 9.3 runner；
- [ ] 能区分 `sm_100`、`sm_100f`、`sm_100a`；
- [ ] 至少包含 TMEM/TCGen05、CLC、TMA irregular 三个 pack；
- [ ] 至少 5 个可复现任务；
- [ ] 每个任务有 hidden correctness 与 liveness tests；
- [ ] native verifier 不仅做源码字符串检查；
- [ ] 能拒绝 cuBLAS 等黑盒回退；
- [ ] unsupported Fabric 环境能记为 N/A；
- [ ] 分开输出 Core、Scheduler、Fabric、Preview 分数；
- [ ] 文档明确列出 Hopper-era 与后续 Blackwell 变体的排除项。

---

# 26. 官方参考资料

以下资料应在实现时固定到具体 CUDA/PTX 版本，并在升级工具链时重新审计。

1. NVIDIA PTX ISA 9.3（稳定文档）  
   https://docs.nvidia.com/cuda/parallel-thread-execution/index.html

2. NVIDIA PTX ISA 9.4 Developer Preview  
   https://docs.nvidia.com/cuda/developer-preview/13.4/parallel-thread-execution/index.html

3. NVIDIA CUDA C++ Programming Guide：Cluster Launch Control  
   https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html

4. NVIDIA CUDA C++ Programming Guide：Asynchronous Copies and TMA  
   https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html

5. NVIDIA Hopper Tuning Guide  
   https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html

6. NVIDIA Blackwell Tuning Guide  
   https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html

7. NVIDIA CUTLASS：Blackwell Functionality  
   https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell.html

---

# 27. 建议下一步

第一批实现建议直接创建：

```text
packs/blackwell-core-tmem/BW-TMEM-001
packs/blackwell-scheduler-clc/BW-CLC-001
packs/blackwell-tma-irregular/BW-TMA-001
verifier/native_sm100.py
schemas/capability.schema.json
```

其中 `native_sm100.py` 应优先实现：

```text
PTX/cubin 收集
目标架构核验
TCGen05/TMEM opcode 检查
CFG 可达性基础检查
动态符号回退检查
watchdog 与结果归档
```

这样可以先把 InfraSWE 的“B200 原生性”判定基础打牢，再逐步扩展低比特、多 GPU 和 PTX preview 任务。

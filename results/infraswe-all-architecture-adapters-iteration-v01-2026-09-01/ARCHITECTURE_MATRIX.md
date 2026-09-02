# InfraSWE 当前架构适配矩阵

## A100 / SM80

- 实测硬件：A100-SXM4 80GB（正式 frontier）与 A100-SXM4 40GB（负控和梯度）。
- Attention：FA1、FA2、FA3、FA4、Torch SDPA Flash 均通过 KernelCert。
- A100 80GB Artifact-100：FA1 96.63、FA2 96.91、FA3 95.92、FA4 96.83、
  Torch SDPA 97.74。
- 经典 kernel：GEMM、LayerNorm、RMSNorm、RoPE、Softmax 已形成正式结果。
- 评分器负控：错误输出、缓存作弊、伪原生 fallback 均被对应 hard gate 拦截；
  正确但浪费工作的 kernel 获得低分而不是被误判失败。
- 分数梯度：A100 40GB 已有 6 档和 15 档两套严格单调退化曲线。
- Profile：已有 1x、2x PCIe、4x PCIe、4x NVLink SM80 profile；当前结果主要是
  单卡 kernel cell，多卡 profile 尚未在本包形成对应 kernel 排名。

结果入口：

- `reports/kernel-frontier-fa1-fa4-classic-v03-2026-08-31/`
- `reports/kernel-negative-controls-v03-a100-40gb-2026-08-31/`
- `reports/kernel-score-gradient-v03-a100-40gb-2026-08-31/`
- `reports/kernel-score-gradient-dense-v03-a100-40gb-2026-09-01/`

## H200 NVL / SM90

- 实测硬件：单卡 H200 NVL 141GB，CC 9.0，132 SM。
- Attention Artifact-100：FA1 94.93、FA2 96.12、FA3 94.81、FA4 96.38、
  Torch SDPA 96.49。
- 经典 kernel：LayerNorm 78.06、RMSNorm 98.76、RoPE 71.57、Softmax 92.44、
  SwiGLU 50.74；GEMM/Vector Add 因 anchor headroom 不足保持 N/A。
- TMA：三轮真实执行和正确性通过；SASS 包含 `UTMALDG.2D`、`UTMASTG.2D`。
- multimem：PTX/cubin/SASS 编译门禁通过，SASS 包含
  `LDGMC.E.ADD.32.STRONG.SYS`；当前单卡且 multicast unsupported，runtime 为
  `topology_unavailable`，没有用普通指针伪造运行。
- Profile 缺口：当前源码没有专用 1x SM90/H200 TOML，硬件身份来自远端 provenance；
  下一版应将实际 cell 固化为 profile。

结果入口：`reports/kernel-frontier-v03-h200-nvl-141gb-2026-09-01/`。

## B200 / SM100

- 实测硬件：单卡 B200，CC 10.0，148 SM；CUDA 13.3 编译工具，PyTorch
  2.11.0+cu130，Triton 3.6.0，CUTLASS/CuTe DSL 4.5.2。
- 三轮 Phase-1 五任务全部通过正确性、liveness、PTX+cubin+SASS hard gate。
- 命名空间分数：SM100-Core 95.95；SM100-Scheduler 75.69；Fabric 为单卡 N/A；
  PTX Preview disabled。
- 任务分：TMEM-001 99.25、CLC-001 65.27、TMA-001 82.68、TMEM-003
  99.98、TMA-002 99.81。
- 原生能力：TMEM/TCGen05、Cluster Launch Control、TMA gather4/scatter4、TMEM
  生命周期/非法对齐拒绝、2CTA TMA 已真实执行。
- 当前边界：本 cell 尚未生成 FA1–FA4/经典 kernel 排名；多 GPU Fabric 仍需合法
  multicast 拓扑闭环。

结果入口：

- `reports/b200-sm100-compiler-adapter-v01-2026-09-01/`
- `reports/b200-sm100-feature-score-v02-2026-09-01/`

## RTX PRO 5000 Blackwell / SM120

- 实测硬件：单卡 RTX PRO 5000 Blackwell 48GB，CC 12.0，110 SM。
- Attention：FA2 98.61、FA4 98.73、Torch SDPA 98.86；FA1/FA3 没有 SM120
  target，明确为 N/A。
- 经典 kernel：7/7 正确性和 KernelCert 通过；RoPE 95.78，其余 case 按 anchor
  headroom 或 quarantine 规则保留 N/A，而不是强行评分。
- 分数梯度：0/32/64/96/192/384/1024 passes 对应 98.75/83.90/75.10/
  68.72/56.84/45.15/31.70，严格单调。
- Profile 缺口：源码已有 2x SM120 PCIe profile，但本次正式结果是 1x SM120；
  下一版应补单卡实际型号 profile，并把 2x profile 留给真正多卡证据。

结果入口：

- `reports/kernel-frontier-v03-sm120-48gb-2026-08-31/`
- `reports/kernel-score-gradient-v03-sm120-48gb-2026-08-31/`
- `reports/sm120-architecture-supplement-v03-2026-08-31/`
- `reports/sm120-architecture-support-evidence-v03-2026-08-31/`

## MI300X / gfx942 / ROCm 6.1

- 冻结组合：MI300X、gfx942、ROCm 6.1.x、PyTorch 2.4.0。
- Attention 候选：PyTorch Flash SDPA / embedded AOTriton；经典 kernel 使用
  evaluator-owned portable Triton 固定配置。
- 冻结的 FA1–FA4 是 CUDA artifact，在 gfx942 cell 均为 `not_applicable`，不会
  偷换成 ROCm fork。
- 已完成：硬件 profile、准备脚本、正式 runner、HIP event timing、ROCm 独占
  lease guard、native trace gate、supervisor 和本地合同测试。
- 尚未完成：真实 MI300X 上的环境快照、AOTriton 原生 trace、三轮正式测量、评分和
  最终证据 ZIP。因此当前状态是 `adapter_ready`，不是 `hardware_scored`。

入口：

- `source/profiles/gpu-1x-gfx942-mi300x-rocm61.toml`
- `source/benchmarks/kernel_frontier/MI300X_ROCM61.md`
- `source/benchmarks/kernel_frontier/remote_prepare_mi300x_rocm61.sh`
- `source/benchmarks/kernel_frontier/remote_run_frontier_mi300x_rocm61.sh`

## 跨架构规则

1. 不同 hardware cell 的 Artifact-100 不直接混排。
2. 正确性、动态输入、原生 trace、artifact binding、watchdog 和 fresh replay 是
   评分前置门禁，不由高性能抵消。
3. unsupported、not_applicable 与 topology_unavailable 不等于性能 0 分。
4. 编译/probe 证据不能替代目标硬件上的正式 runtime evidence。
5. RFC 只提供设计背景；实际执行以 `source/` 中的 versioned contract、runner 和
   schema 为准。


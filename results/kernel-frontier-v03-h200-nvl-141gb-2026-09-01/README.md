# InfraSWE H200 NVL / SM90 补测包

本包包含单卡 NVIDIA H200 NVL（CC 9.0、132 SM、139.8 GiB）的 InfraSWE v0.3
kernel frontier 正式评分，以及 TMA / multimem 架构新特性补测。

## 入口

- `report.md` / `index.html`：FA1–FA4、Torch SDPA 与经典 kernel 正式评分。
- `h200-features.md`：TMA 真实执行与 multimem 能力/拓扑门禁。
- `score.json`：机器可读评分。
- `evidence/h200-nvl-sm90-141gb/`：3 replay 原始 matched blocks、Profiler、
  PTX/cubin/SASS、环境与实现 provenance。
- `verification/`：测试、静态检查、证据网格、hash 与 ZIP 校验日志。

## 核心结果

| Attention backend | KernelCert | Artifact-100 |
|---|---:|---:|
| Torch SDPA Flash | PASS | 96.49 |
| FA4 | PASS | 96.38 |
| FA2 | PASS | 96.12 |
| FA1 | PASS | 94.93 |
| FA3 | PASS | 94.81 |

经典 kernel 中，RMSNorm 98.76、Softmax 92.44、LayerNorm 78.06、RoPE 71.57、
SwiGLU 50.74；GEMM 与 Vector Add 因校准 anchor headroom 不足而保持 N/A。

TMA 的 3 个 fresh-process replay 均通过正确性、Profiler 与指令门禁，SASS 明确包含
`UTMALDG.2D` / `UTMASTG.2D`。multimem PTX/cubin/SASS 编译门禁通过，SASS 包含
`LDGMC.E.ADD.32.STRONG.SYS`；当前 cell 的
`CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED=0` 且仅有单卡，因此 runtime 明确记为
`topology_unavailable`，没有对普通指针执行未定义的 `multimem.*` 操作。

不同硬件 cell 的 Artifact-100 不应直接混排。完整公式、case 级时延与 CI95 请以
`report.md` 和 `score.json` 为准。

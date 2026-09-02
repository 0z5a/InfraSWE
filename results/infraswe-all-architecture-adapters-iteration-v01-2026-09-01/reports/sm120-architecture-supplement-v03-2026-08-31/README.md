# InfraSWE v0.3 — SM120 架构补测

生成时间：`2026-08-31T13:56:30Z`

本补测使用单个独立 hardware cell：NVIDIA RTX PRO 5000 Blackwell 48GB，CC 12.0，110 SM，Driver 580.173.02，PyTorch 2.8.0+cu128。所有正式候选均为 3 次 fresh-process replay；attention 每候选另有 5 份 per-case profiler，经典 Triton 组合另有 7 份 profiler。

## SM120 正式 attention 评分

| 实现 | SM120 Artifact-100 | KernelCert | 原生证据 | A100 参考分数* |
|---|---:|---:|---|---:|
| Torch SDPA Flash | 98.86 | PASS | native Flash profiler | 97.74 |
| FA4 4.0.0b28 | 98.73 | PASS | `FlashAttentionForwardSm120` | 96.83 |
| FA2 2.8.4 / `ce088ab` | 98.61 | PASS | 96 个 `sm_120` ELF images + native profiler | 96.91 |
| FA1 | N/A | N/A | 无 SM120 target | 96.63 |
| FA3 | N/A | N/A | 无 SM120 target | 95.92 |

\* A100 分数只作已有结果索引；Artifact-100 按 hardware cell 独立校准，禁止把 A100 与 SM120 直接混排。

## SM120 受控退化分数梯度

所有候选先执行正确的 FA4 SM120 路径，再追加指定次数的无用 Triton streaming pass。

| Passes | Artifact-100 | KernelCert |
|---:|---:|---:|
| 0 | 98.75 | PASS |
| 32 | 83.90 | PASS |
| 64 | 75.10 | PASS |
| 96 | 68.72 | PASS |
| 192 | 56.84 | PASS |
| 384 | 45.15 | PASS |
| 1024 | 31.70 | PASS |

梯度严格单调，总跨度 67.05 分，最小相邻间隔 6.37 分，pilot 与 formal 最大偏差 2.55 分。SM120 必须扩展到 1024 passes 才能覆盖低分端，因此档位不复用 A100 的 0–128 配置。

## 经典 Triton kernel

7/7 case 均数值正确且 KernelCert 通过。评分器按 anchor 门禁保留以下判定：

| Case | 判定 | Artifact-100 |
|---|---|---:|
| RoPE BF16 | scored | 95.78 |
| GEMM BF16 | not_frontier_eligible | N/A |
| LayerNorm BF16 | not_frontier_eligible | N/A |
| Vector Add BF16 | not_frontier_eligible | N/A |
| RMSNorm BF16 | quarantined | N/A |
| Softmax BF16 | quarantined | N/A |
| SwiGLU BF16 | quarantined | N/A |

`not_frontier_eligible` 表示 PyTorch baseline 对校准 anchor 没有足够 headroom；`quarantined` 表示观测值超越 anchor 容差。两者均保留全部 raw latency、正确性和 profiler，但不强行产生排行榜有效分。

## 本机校准

| Cell | Launch floor | HBM copy proxy | BF16 GEMM |
|---|---:|---:|---:|
| SM120 RTX PRO 5000 | 3.999 µs | 1108.0 GB/s | 204.1 TFLOP/s |
| SM80 A100 40GB（已有报告） | 4.661 µs | 1359.1 GB/s | 246.3 TFLOP/s |

## 证据与目录

- `frontier-sm120/`：FA2、FA4、Torch SDPA 与经典 kernel 的正式报告、raw JSON、profiler、provenance、schema 和 SHA-256 清单。
- `score-gradient-sm120/`：7 档正式梯度报告、24 份 raw JSON、35 份 profiler 与 pilot 对照。
- `support-evidence/`：Torch/CUDA smoke、FA4 探针、FA2 原生构建日志、FA2 探针和 supervisor 日志。
- `summary.json`：机器可读摘要。
- `manifest.json`：整个补测包的逐文件 SHA-256 清单。


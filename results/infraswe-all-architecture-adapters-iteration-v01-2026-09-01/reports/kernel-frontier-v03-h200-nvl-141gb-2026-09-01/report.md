# InfraSWE Kernel Frontier v0.3 — FA1–FA4 与经典 Kernel 评分

生成时间：`2026-09-01T04:07:07.951809Z`  
Suite：`kernel-frontier-fa1-fa4-classic-v03`  
公式：`kernel-artifact-v0.3`；AnchorScore 来源：`sol-execbench-equivalent`

评分使用三次独立进程 replay、每 case 30 个 matched ABBA/BAAB blocks、evaluator-owned CUDA events 与 block/replay 两层 bootstrap CI95。FA 库分数为 `100 × (0.80P + 0.20G)`；经典 micro kernel 各自为 `100 × AnchorScore`。
同一行只在同一硬件 cell 内可比较，不同 GPU/架构 cell 的分数不得直接混排。

实现与公式来源：

- FA1 固定为 Dao-AILab/flash-attention [`6d48e14`](https://github.com/Dao-AILab/flash-attention/commit/6d48e14a6c2f551db96f0badc658a6279a929df3)（v1.0.9）。
- FA2/FA3 固定为 Dao-AILab/flash-attention [`ce088ab`](https://github.com/Dao-AILab/flash-attention/commit/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820)。
- FA4 固定为 PyPI [`flash-attn-4==4.0.0b28`](https://pypi.org/project/flash-attn-4/)。
- AnchorScore 采用 [SOL-ExecBench](https://arxiv.org/abs/2603.19173) 等价公式。

## Cell: nvidia-h200-nvl-sm90-c78156a7f1d7

GPU：NVIDIA H200 NVL；CC 9.0；SM 132；显存 139.8 GiB；PyTorch 2.8.0+cu128 / CUDA 12.8；Driver 580.159.03。

校准中位数：launch floor 5.0810 µs；HBM proxy 3922.5 GB/s；BF16 GEMM 567.5 TFLOP/s。

### FA 支持矩阵

| 实现 | 状态 | 原因 |
|---|---|---|
| FA1 | eligible | frozen FA1 source rebuild supports an explicit native SM90 target |
| FA2 | eligible | frozen FA2 source rebuild supports the Hopper SM90 target |
| FA3 | eligible | official FA3 Hopper path targets SM90 |
| FA4 | eligible | flash-attn-4 4.0.0b28 includes an explicit SM90 CuTeDSL path |

### Attention / FA 评分

| Backend | 版本 | KernelCert | 状态 | Artifact-100 | P | G | 几何加权 speedup |
|---|---:|---:|---|---:|---:|---:|---:|
| fa1 | 1.0.9 | PASS | scored | 94.93 | 0.938 | 0.995 | 12.996× |
| fa2 | 2.8.4 | PASS | scored | 96.12 | 0.952 | 1.000 | 15.998× |
| fa3 | 3.0.0 | PASS | scored | 94.81 | 0.937 | 0.991 | 14.860× |
| fa4 | 4.0.0b28 | PASS | scored | 96.38 | 0.956 | 0.994 | 20.638× |
| torch-sdpa-flash | 2.8.0+cu128 | PASS | scored | 96.49 | 0.956 | 0.999 | 17.351× |

#### fa1 per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 51.153 | 913.907 | 17.858× [17.842, 17.880] | 8.121 | 0.955 | scored |
| common-b1-s2048-h16-d128-causal | common | 181.028 | 1855.882 | 10.255× [10.246, 10.269] | 30.275 | 0.924 | scored |
| common-b2-s1024-h16-d64-causal | common | 45.816 | 797.087 | 17.391× [17.366, 17.415] | 7.569 | 0.954 | scored |
| common-b4-s512-h16-d64-noncausal | common | 33.457 | 397.344 | 11.874× [11.800, 11.909] | 7.569 | 0.938 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 382.338 | 3748.745 | 9.803× [9.788, 9.826] | 60.551 | 0.920 | scored |

#### fa2 per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 46.497 | 913.489 | 19.636× [19.593, 19.728] | 8.121 | 0.959 | scored |
| common-b1-s2048-h16-d128-causal | common | 109.305 | 1856.650 | 16.992× [16.901, 17.076] | 30.275 | 0.959 | scored |
| common-b2-s1024-h16-d64-causal | common | 46.333 | 796.075 | 17.187× [17.011, 17.249] | 7.569 | 0.953 | scored |
| common-b4-s512-h16-d64-noncausal | common | 39.107 | 397.115 | 10.151× [10.093, 10.196] | 7.569 | 0.925 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 208.282 | 3747.686 | 18.002× [17.922, 18.058] | 60.551 | 0.962 | scored |

#### fa3 per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 80.682 | 914.227 | 11.338× [11.232, 11.420] | 8.121 | 0.926 | scored |
| common-b1-s2048-h16-d128-causal | common | 80.772 | 1860.527 | 23.035× [22.633, 23.302] | 30.275 | 0.973 | scored |
| common-b2-s1024-h16-d64-causal | common | 80.033 | 797.243 | 9.963× [9.725, 10.031] | 7.569 | 0.916 | scored |
| common-b4-s512-h16-d64-noncausal | common | 68.896 | 397.587 | 5.790× [4.216, 6.493] | 7.569 | 0.876 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 80.771 | 3881.461 | 48.100× [47.630, 48.482] | 60.551 | 0.995 | scored |

#### fa4 per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 54.528 | 913.526 | 16.744× [16.678, 16.901] | 8.121 | 0.951 | scored |
| common-b1-s2048-h16-d128-causal | common | 55.109 | 1857.488 | 33.730× [33.625, 34.097] | 30.275 | 0.987 | scored |
| common-b2-s1024-h16-d64-causal | common | 54.402 | 797.658 | 14.666× [14.607, 14.899] | 7.569 | 0.944 | scored |
| common-b4-s512-h16-d64-noncausal | common | 50.337 | 397.868 | 7.900× [7.866, 7.977] | 7.569 | 0.901 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 69.320 | 3943.144 | 57.213× [46.275, 57.684] | 60.551 | 0.998 | scored |

#### torch-sdpa-flash per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 40.021 | 912.531 | 22.793× [22.627, 22.827] | 8.121 | 0.966 | scored |
| common-b1-s2048-h16-d128-causal | common | 116.531 | 1854.649 | 15.918× [15.610, 15.954] | 30.275 | 0.955 | scored |
| common-b2-s1024-h16-d64-causal | common | 38.499 | 795.116 | 20.638× [20.493, 20.719] | 7.569 | 0.962 | scored |
| common-b4-s512-h16-d64-noncausal | common | 31.641 | 396.992 | 12.564× [8.682, 12.871] | 7.569 | 0.942 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 224.120 | 3745.914 | 16.714× [16.637, 16.831] | 60.551 | 0.958 | scored |

### 经典 Kernel 独立评分

| Kernel | KernelCert | 状态 | Artifact-100 | Candidate µs | Baseline µs | Speedup [CI95] | AnchorScore |
|---|---:|---|---:|---:|---:|---:|---:|
| gemm-bf16-4096-cube | PASS | not_frontier_eligible | — | 507.837 | 259.426 | 0.510× [0.508, 0.513] | — |
| layernorm-bf16-4096x4096 | PASS | scored | 78.06 | 22.228 | 35.160 | 1.584× [1.556, 1.590] | 0.781 |
| rmsnorm-bf16-4096x4096 | PASS | scored | 98.76 | 20.937 | 319.471 | 15.269× [15.114, 15.306] | 0.988 |
| rope-bf16-b4-s2048-h16-d128 | PASS | scored | 71.57 | 90.026 | 200.434 | 2.227× [2.224, 2.228] | 0.716 |
| softmax-bf16-4096x4096 | PASS | scored | 92.44 | 20.776 | 61.827 | 2.977× [2.834, 2.988] | 0.924 |
| swiglu-bf16-8192x4096 | PASS | scored | 50.74 | 90.460 | 91.464 | 1.011× [1.007, 1.015] | 0.507 |
| vector-add-bf16-16m | PASS | not_frontier_eligible | — | 46.781 | 27.378 | 0.585× [0.585, 0.586] | — |

## 判读边界

- functional reference 是 PyTorch math/eager；它同时作为本次 scoring baseline，不代表最佳厂商实现。
- anchor 为同机 launch/HBM/BF16 GEMM 校准目标，confidence=medium，不宣称物理下界。
- raw speedup、CI、AnchorScore 均保留未裁剪值；超越 anchor tolerance 的结果进入 quarantine。
- FA4 按固定 b28 包的显式架构 dispatch 判定资格；纳入的硬件 cell 均需提供实测原生路径证据，未注册的架构记为 unresolved，不用其他实现代填。
- 所有原始 JSON、实现摘要、报告与 SHA-256 清单均包含在 ZIP 中。

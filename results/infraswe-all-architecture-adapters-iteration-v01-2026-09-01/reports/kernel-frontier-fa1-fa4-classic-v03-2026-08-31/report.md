# InfraSWE Kernel Frontier v0.3 — FA1–FA4 与经典 Kernel 评分

生成时间：`2026-08-31T11:12:47.667886Z`  
Suite：`kernel-frontier-fa1-fa4-classic-v03`  
公式：`kernel-artifact-v0.3`；AnchorScore 来源：`sol-execbench-equivalent`

评分使用三次独立进程 replay、每 case 30 个 matched ABBA/BAAB blocks、evaluator-owned CUDA events 与 block/replay 两层 bootstrap CI95。FA 库分数为 `100 × (0.80P + 0.20G)`；经典 micro kernel 各自为 `100 × AnchorScore`。
同一行只在同一硬件 cell 内可比较，A100 与 SM120 的分数不得直接混排。

实现与公式来源：

- FA1 固定为 Dao-AILab/flash-attention [`6d48e14`](https://github.com/Dao-AILab/flash-attention/commit/6d48e14a6c2f551db96f0badc658a6279a929df3)（v1.0.9）。
- FA2/FA3 固定为 Dao-AILab/flash-attention [`ce088ab`](https://github.com/Dao-AILab/flash-attention/commit/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820)。
- FA4 固定为 PyPI [`flash-attn-4==4.0.0b28`](https://pypi.org/project/flash-attn-4/)。
- AnchorScore 采用 [SOL-ExecBench](https://arxiv.org/abs/2603.19173) 等价公式。

## Cell: nvidia-a100-sxm4-80gb-sm80-1d06f63de161

GPU：NVIDIA A100-SXM4-80GB；CC 8.0；SM 108；显存 79.3 GiB；PyTorch 2.8.0+cu128 / CUDA 12.8；Driver 580.65.06。

校准中位数：launch floor 6.5936 µs；HBM proxy 1702.9 GB/s；BF16 GEMM 284.2 TFLOP/s。

### FA 支持矩阵

| 实现 | 状态 | 原因 |
|---|---|---|
| FA1 | eligible | official SM80 path |
| FA2 | eligible | official SM80 path |
| FA3 | eligible | current FA3 tree includes SM80 path |
| FA4 | eligible | flash-attn-4 4.0.0b28 includes an explicit SM80 CuTeDSL path |

### Attention / FA 评分

| Backend | 版本 | KernelCert | 状态 | Artifact-100 | P | G | 几何加权 speedup |
|---|---:|---:|---|---:|---:|---:|---:|
| fa1 | 1.0.9 | PASS | scored | 96.63 | 0.959 | 0.997 | 18.556× |
| fa2 | 2.8.4 | PASS | scored | 96.91 | 0.962 | 0.998 | 20.166× |
| fa3 | 3.0.0 | PASS | scored | 95.92 | 0.950 | 0.994 | 17.553× |
| fa4 | 4.0.0b28 | PASS | scored | 96.83 | 0.961 | 0.999 | 19.503× |
| torch-sdpa-flash | 2.8.0+cu128 | PASS | scored | 97.74 | 0.972 | 1.000 | 24.533× |

#### fa1 per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 66.464 | 1869.962 | 28.138× [28.118, 28.152] | 16.214 | 0.974 | scored |
| common-b1-s2048-h16-d128-causal | common | 265.311 | 3953.572 | 14.902× [14.899, 14.907] | 60.450 | 0.950 | scored |
| common-b2-s1024-h16-d64-causal | common | 79.055 | 1784.442 | 22.572× [22.567, 22.576] | 15.112 | 0.965 | scored |
| common-b4-s512-h16-d64-noncausal | common | 59.417 | 887.214 | 14.932× [14.929, 14.936] | 15.112 | 0.952 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 511.519 | 7961.771 | 15.567× [15.559, 15.600] | 120.900 | 0.953 | scored |

#### fa2 per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 90.671 | 1871.090 | 20.644× [20.435, 20.738] | 16.214 | 0.962 | scored |
| common-b1-s2048-h16-d128-causal | common | 150.745 | 3954.027 | 26.232× [26.226, 26.242] | 60.450 | 0.977 | scored |
| common-b2-s1024-h16-d64-causal | common | 90.852 | 1785.542 | 19.668× [19.516, 19.764] | 15.112 | 0.959 | scored |
| common-b4-s512-h16-d64-noncausal | common | 78.156 | 887.416 | 11.373× [11.314, 11.495] | 15.112 | 0.933 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 289.261 | 7963.591 | 27.533× [27.495, 27.576] | 120.900 | 0.979 | scored |

#### fa3 per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 134.316 | 1871.569 | 13.931× [13.826, 13.985] | 16.214 | 0.940 | scored |
| common-b1-s2048-h16-d128-causal | common | 135.829 | 3952.367 | 29.100× [28.892, 29.198] | 60.450 | 0.981 | scored |
| common-b2-s1024-h16-d64-causal | common | 134.195 | 1785.765 | 13.308× [13.198, 13.343] | 15.112 | 0.937 | scored |
| common-b4-s512-h16-d64-noncausal | common | 105.633 | 887.416 | 8.412× [8.364, 8.507] | 15.112 | 0.906 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 216.824 | 7961.409 | 36.718× [36.712, 36.726] | 120.900 | 0.988 | scored |

#### fa4 per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 85.758 | 1874.509 | 21.863× [21.614, 22.033] | 16.214 | 0.964 | scored |
| common-b1-s2048-h16-d128-causal | common | 168.976 | 3960.140 | 23.435× [23.424, 23.445] | 60.450 | 0.973 | scored |
| common-b2-s1024-h16-d64-causal | common | 86.157 | 1786.320 | 20.763× [20.422, 21.078] | 15.112 | 0.962 | scored |
| common-b4-s512-h16-d64-noncausal | common | 78.751 | 888.206 | 11.294× [11.157, 11.389] | 15.112 | 0.932 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 339.598 | 7975.843 | 23.485× [23.479, 23.489] | 120.900 | 0.973 | scored |

#### torch-sdpa-flash per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 65.536 | 1873.146 | 28.582× [28.467, 29.001] | 16.214 | 0.974 | scored |
| common-b1-s2048-h16-d128-causal | common | 154.987 | 3958.283 | 25.539× [25.522, 25.554] | 60.450 | 0.976 | scored |
| common-b2-s1024-h16-d64-causal | common | 65.073 | 1785.391 | 27.445× [27.301, 27.649] | 15.112 | 0.972 | scored |
| common-b4-s512-h16-d64-noncausal | common | 52.212 | 887.225 | 16.998× [16.854, 17.274] | 15.112 | 0.959 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 305.641 | 7974.053 | 26.095× [26.073, 26.103] | 120.900 | 0.977 | scored |

### 经典 Kernel 独立评分

| Kernel | KernelCert | 状态 | Artifact-100 | Candidate µs | Baseline µs | Speedup [CI95] | AnchorScore |
|---|---:|---|---:|---:|---:|---:|---:|
| gemm-bf16-4096-cube | PASS | scored | 12.24 | 866.847 | 536.995 | 0.619× [0.619, 0.620] | 0.122 |
| layernorm-bf16-4096x4096 | PASS | scored | 75.68 | 45.068 | 56.998 | 1.265× [1.264, 1.265] | 0.757 |
| rmsnorm-bf16-4096x4096 | PASS | scored | 99.09 | 44.225 | 564.464 | 12.764× [12.761, 12.767] | 0.991 |
| rope-bf16-b4-s2048-h16-d128 | PASS | scored | 81.57 | 100.558 | 308.964 | 3.073× [3.072, 3.073] | 0.816 |
| softmax-bf16-4096x4096 | PASS | scored | 92.45 | 43.947 | 95.022 | 2.162× [2.162, 2.163] | 0.925 |
| swiglu-bf16-8192x4096 | PASS | scored | 89.81 | 127.642 | 201.164 | 1.576× [1.576, 1.576] | 0.898 |
| vector-add-bf16-16m | PASS | not_frontier_eligible | — | 64.864 | 61.060 | 0.941× [0.941, 0.942] | — |

## 判读边界

- functional reference 是 PyTorch math/eager；它同时作为本次 scoring baseline，不代表最佳厂商实现。
- anchor 为同机 launch/HBM/BF16 GEMM 校准目标，confidence=medium，不宣称物理下界。
- raw speedup、CI、AnchorScore 均保留未裁剪值；超越 anchor tolerance 的结果进入 quarantine。
- FA4 按固定 b28 包的显式架构 dispatch 判定资格；该版本包含 SM80 路径，未包含的架构明确记为 N/A，不用其他实现代填。
- 所有原始 JSON、实现摘要、报告与 SHA-256 清单均包含在 ZIP 中。

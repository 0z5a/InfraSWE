# InfraSWE Kernel Frontier v0.3 — FA1–FA4 与经典 Kernel 评分

生成时间：`2026-08-31T13:55:09.531766Z`  
Suite：`kernel-frontier-fa1-fa4-classic-v03`  
公式：`kernel-artifact-v0.3`；AnchorScore 来源：`sol-execbench-equivalent`

评分使用三次独立进程 replay、每 case 30 个 matched ABBA/BAAB blocks、evaluator-owned CUDA events 与 block/replay 两层 bootstrap CI95。FA 库分数为 `100 × (0.80P + 0.20G)`；经典 micro kernel 各自为 `100 × AnchorScore`。
同一行只在同一硬件 cell 内可比较，A100 与 SM120 的分数不得直接混排。

实现与公式来源：

- FA1 固定为 Dao-AILab/flash-attention [`6d48e14`](https://github.com/Dao-AILab/flash-attention/commit/6d48e14a6c2f551db96f0badc658a6279a929df3)（v1.0.9）。
- FA2/FA3 固定为 Dao-AILab/flash-attention [`ce088ab`](https://github.com/Dao-AILab/flash-attention/commit/ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820)。
- FA4 固定为 PyPI [`flash-attn-4==4.0.0b28`](https://pypi.org/project/flash-attn-4/)。
- AnchorScore 采用 [SOL-ExecBench](https://arxiv.org/abs/2603.19173) 等价公式。

## Cell: nvidia-rtx-pro-5000-blackwell-sm120-810d66d61ae3

GPU：NVIDIA RTX PRO 5000 Blackwell；CC 12.0；SM 110；显存 47.3 GiB；PyTorch 2.8.0+cu128 / CUDA 12.8；Driver 580.173.02。

校准中位数：launch floor 3.9994 µs；HBM proxy 1108.0 GB/s；BF16 GEMM 204.1 TFLOP/s。

### FA 支持矩阵

| 实现 | 状态 | 原因 |
|---|---|---|
| FA1 | not_applicable | FA1 has no SM120 target |
| FA2 | eligible | current FA2 has SM120 target |
| FA3 | not_applicable | FA3 CUDA path has no SM120 target |
| FA4 | eligible | official FA4 Blackwell path |

### Attention / FA 评分

| Backend | 版本 | KernelCert | 状态 | Artifact-100 | P | G | 几何加权 speedup |
|---|---:|---:|---|---:|---:|---:|---:|
| fa2 | 2.8.4 | PASS | scored | 98.61 | 0.983 | 0.999 | 27.733× |
| fa4 | 4.0.0b28 | PASS | scored | 98.73 | 0.984 | 1.000 | 28.299× |
| torch-sdpa-flash | 2.8.0+cu128 | PASS | scored | 98.86 | 0.986 | 1.000 | 29.311× |

#### fa2 per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 47.784 | 1710.773 | 35.744× [35.703, 35.803] | 22.580 | 0.985 | scored |
| common-b1-s2048-h16-d128-causal | common | 133.796 | 3925.018 | 29.332× [29.171, 29.372] | 84.183 | 0.987 | scored |
| common-b2-s1024-h16-d64-causal | common | 46.938 | 1518.963 | 32.365× [31.937, 32.834] | 21.046 | 0.983 | scored |
| common-b4-s512-h16-d64-noncausal | common | 39.147 | 581.932 | 14.862× [14.280, 15.167] | 21.046 | 0.969 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 253.102 | 8234.949 | 32.530× [32.492, 32.550] | 168.366 | 0.990 | scored |

#### fa4 per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 46.287 | 1712.730 | 36.999× [36.968, 37.021] | 22.580 | 0.986 | scored |
| common-b1-s2048-h16-d128-causal | common | 145.671 | 3904.114 | 26.796× [26.773, 26.809] | 84.183 | 0.984 | scored |
| common-b2-s1024-h16-d64-causal | common | 43.474 | 1526.697 | 35.098× [35.053, 35.158] | 21.046 | 0.985 | scored |
| common-b4-s512-h16-d64-noncausal | common | 33.500 | 585.692 | 17.502× [17.358, 17.942] | 21.046 | 0.978 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 275.895 | 8224.007 | 29.802× [29.787, 29.817] | 168.366 | 0.987 | scored |

#### torch-sdpa-flash per-shape

| Case | 分组 | Candidate µs | Baseline µs | Speedup [CI95] | Anchor µs | AnchorScore | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| boundary-b3-s1000-h12-d64-causal | boundary_tail | 47.215 | 1700.311 | 36.053× [35.973, 36.170] | 22.580 | 0.986 | scored |
| common-b1-s2048-h16-d128-causal | common | 139.724 | 3890.281 | 27.845× [27.709, 27.933] | 84.183 | 0.986 | scored |
| common-b2-s1024-h16-d64-causal | common | 41.235 | 1518.184 | 36.809× [36.537, 36.970] | 21.046 | 0.987 | scored |
| common-b4-s512-h16-d64-noncausal | common | 30.801 | 581.712 | 18.922× [18.880, 19.021] | 21.046 | 0.983 | scored |
| stress-b1-s4096-h8-d128-causal | stress_large | 264.996 | 8198.692 | 30.940× [30.863, 30.998] | 168.366 | 0.988 | scored |

### 经典 Kernel 独立评分

| Kernel | KernelCert | 状态 | Artifact-100 | Candidate µs | Baseline µs | Speedup [CI95] | AnchorScore |
|---|---:|---|---:|---:|---:|---:|---:|
| gemm-bf16-4096-cube | PASS | not_frontier_eligible | — | 867.370 | 659.670 | 0.760× [0.757, 0.761] | — |
| layernorm-bf16-4096x4096 | PASS | not_frontier_eligible | — | 23.180 | 38.866 | 1.675× [1.658, 1.677] | — |
| rmsnorm-bf16-4096x4096 | PASS | quarantined | — | 23.058 | 461.479 | 20.009× [19.987, 20.037] | — |
| rope-bf16-b4-s2048-h16-d128 | PASS | scored | 95.78 | 67.846 | 215.635 | 3.179× [3.156, 3.186] | 0.958 |
| softmax-bf16-4096x4096 | PASS | quarantined | — | 23.029 | 77.872 | 3.374× [3.364, 3.379] | — |
| swiglu-bf16-8192x4096 | PASS | quarantined | — | 174.978 | 264.546 | 1.512× [1.509, 1.513] | — |
| vector-add-bf16-16m | PASS | not_frontier_eligible | — | 75.360 | 77.221 | 1.025× [1.024, 1.025] | — |

## 判读边界

- functional reference 是 PyTorch math/eager；它同时作为本次 scoring baseline，不代表最佳厂商实现。
- anchor 为同机 launch/HBM/BF16 GEMM 校准目标，confidence=medium，不宣称物理下界。
- raw speedup、CI、AnchorScore 均保留未裁剪值；超越 anchor tolerance 的结果进入 quarantine。
- FA4 按固定 b28 包的显式架构 dispatch 判定资格；SM80 与 SM120 均有实测原生路径，未包含的架构明确记为 N/A，不用其他实现代填。
- 所有原始 JSON、实现摘要、报告与 SHA-256 清单均包含在 ZIP 中。

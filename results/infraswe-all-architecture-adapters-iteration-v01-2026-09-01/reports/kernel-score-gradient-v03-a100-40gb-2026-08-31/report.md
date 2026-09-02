# InfraSWE v0.3 Kernel 分数梯度报告

梯度门禁：**PASS**  
评分生成时间：`2026-08-31T12:28:28.654289Z`

硬件 cell：`nvidia-a100-sxm4-40gb-sm80-df09afe222d8`；NVIDIA A100-SXM4-40GB；CC 8.0；显存 39.5 GiB；Driver 595.58.03。

候选均使用相同 FA4 正确路径，仅改变其后追加的无用 Triton streaming passes，从而构造可复现、可解释的性能退化曲线。pass=0 是未追加浪费工作的上界控制；pass=64 复用同一硬件 cell 已完成的正式负控证据。

正式分数跨度：**51.59**；最小相邻间隔：**7.39**；最大 pilot 误差：**2.52**。

| Passes | Backend | Cert | Pilot | Formal Artifact-100 | Δ | P | G | Raw speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | mediocre-fa4-waste0 | PASS | 95.21 | 97.73 | 2.52 | 0.972 | 1.000 | 24.202× |
| 8 | mediocre-fa4-waste8 | PASS | 85.95 | 88.05 | 2.11 | 0.854 | 0.988 | 6.031× |
| 16 | mediocre-fa4-waste16 | PASS | 79.49 | 80.66 | 1.17 | 0.764 | 0.977 | 3.454× |
| 32 | mediocre-fa4-waste32 | PASS | 69.68 | 70.42 | 0.74 | 0.640 | 0.959 | 1.873× |
| 64 | garbage-slow-fa4-waste64 | PASS | 57.93 | 58.28 | 0.35 | 0.494 | 0.936 | 0.977× |
| 128 | mediocre-fa4-waste128 | PASS | 46.01 | 46.14 | 0.13 | 0.349 | 0.911 | 0.500× |

## 完整性结论

- 六级或以上梯度：PASS。
- 分数随浪费工作严格递减：PASS。
- 每点 3 replay + 5 profiler 且 KernelCert 通过：PASS。
- 这些是评分器校准候选，不进入正式 FA1–FA4 排名。
- pilot、raw matched blocks、profiler、provenance 与 SHA-256 清单均随 ZIP 提供。

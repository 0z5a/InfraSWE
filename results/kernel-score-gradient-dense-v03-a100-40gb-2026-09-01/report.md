# InfraSWE v0.3 Kernel 分数梯度报告

梯度门禁：**PASS**  
评分生成时间：`2026-09-01T02:36:09.623471Z`

硬件 cell：`nvidia-a100-sxm4-40gb-sm80-6c5e19f80c01`；NVIDIA A100-SXM4-40GB；CC 8.0；显存 39.5 GiB；Driver 580.105.08。

候选均使用相同 FA4 正确路径，仅改变其后追加的无用 Triton streaming passes，从而构造可复现、可解释的性能退化曲线。pass=0 是未追加浪费工作的上界控制；pass=64 复用同一硬件 cell 已完成的正式负控证据。

正式分数跨度：**72.05**；最小相邻间隔：**3.43**；最大 pilot 误差：**3.15**。

| Passes | Backend | Cert | Pilot | Formal Artifact-100 | Δ | P | G | Raw speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | mediocre-fa4-waste0 | PASS | 94.59 | 97.00 | 2.41 | 0.963 | 0.999 | 20.188× |
| 4 | mediocre-fa4-waste4 | PASS | 89.56 | 92.72 | 3.15 | 0.910 | 0.994 | 9.727× |
| 8 | mediocre-fa4-waste8 | PASS | 85.56 | 88.07 | 2.51 | 0.854 | 0.988 | 6.039× |
| 12 | mediocre-fa4-waste12 | PASS | 81.91 | 84.11 | 2.21 | 0.806 | 0.982 | 4.395× |
| 16 | mediocre-fa4-waste16 | PASS | 78.69 | 80.68 | 1.99 | 0.764 | 0.977 | 3.459× |
| 24 | mediocre-fa4-waste24 | PASS | 73.60 | 75.00 | 1.40 | 0.696 | 0.967 | 2.429× |
| 32 | mediocre-fa4-waste32 | PASS | 69.32 | 70.43 | 1.11 | 0.641 | 0.959 | 1.872× |
| 48 | mediocre-fa4-waste48 | PASS | 62.69 | 63.47 | 0.78 | 0.557 | 0.946 | 1.285× |
| 64 | garbage-slow-fa4-waste64 | PASS | 57.73 | 58.29 | 0.56 | 0.495 | 0.936 | 0.977× |
| 96 | mediocre-fa4-waste96 | PASS | 50.69 | 51.07 | 0.37 | 0.408 | 0.921 | 0.662× |
| 128 | mediocre-fa4-waste128 | PASS | 45.87 | 46.16 | 0.29 | 0.349 | 0.911 | 0.501× |
| 160 | mediocre-fa4-waste160 | PASS | 42.36 | 42.55 | 0.20 | 0.306 | 0.903 | 0.403× |
| 256 | mediocre-fa4-waste256 | PASS | 35.68 | 35.77 | 0.09 | 0.225 | 0.889 | 0.253× |
| 384 | mediocre-fa4-waste384 | PASS | 30.91 | 30.97 | 0.06 | 0.167 | 0.879 | 0.170× |
| 768 | mediocre-fa4-waste768 | PASS | 24.95 | 24.95 | 0.00 | 0.095 | 0.868 | 0.085× |

## 完整性结论

- 六级或以上梯度：PASS。
- 分数随浪费工作严格递减：PASS。
- 每点 3 replay + 5 profiler 且 KernelCert 通过：PASS。
- 这些是评分器校准候选，不进入正式 FA1–FA4 排名。
- pilot、raw matched blocks、profiler、provenance 与 SHA-256 清单均随 ZIP 提供。

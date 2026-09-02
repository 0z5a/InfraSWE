# InfraSWE v0.3 Kernel 分数梯度报告

梯度门禁：**PASS**  
评分生成时间：`2026-08-31T13:55:06.936969Z`

硬件 cell：`nvidia-rtx-pro-5000-blackwell-sm120-810d66d61ae3`；NVIDIA RTX PRO 5000 Blackwell；CC 12.0；显存 47.3 GiB；Driver 580.173.02。

候选均使用相同 FA4 正确路径，仅改变其后追加的无用 Triton streaming passes，从而构造可复现、可解释的性能退化曲线。pass=0 是未追加浪费工作的上界控制；pass=64 复用同一硬件 cell 已完成的正式负控证据。

正式分数跨度：**67.05**；最小相邻间隔：**6.37**；最大 pilot 误差：**2.55**。

| Passes | Backend | Cert | Pilot | Formal Artifact-100 | Δ | P | G | Raw speedup |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | mediocre-fa4-waste0 | PASS | 96.20 | 98.75 | 2.55 | 0.984 | 1.000 | 28.327× |
| 32 | mediocre-fa4-waste32 | PASS | 83.60 | 83.90 | 0.30 | 0.804 | 0.978 | 4.699× |
| 64 | garbage-slow-fa4-waste64 | PASS | 75.42 | 75.10 | -0.32 | 0.698 | 0.963 | 2.593× |
| 96 | mediocre-fa4-waste96 | PASS | 69.35 | 68.72 | -0.62 | 0.621 | 0.950 | 1.768× |
| 192 | mediocre-fa4-waste192 | PASS | 57.79 | 56.84 | -0.95 | 0.479 | 0.926 | 0.905× |
| 384 | mediocre-fa4-waste384 | PASS | 45.93 | 45.15 | -0.78 | 0.339 | 0.900 | 0.459× |
| 1024 | mediocre-fa4-waste1024 | PASS | 32.39 | 31.70 | -0.70 | 0.178 | 0.871 | 0.173× |

## 完整性结论

- 六级或以上梯度：PASS。
- 分数随浪费工作严格递减：PASS。
- 每点 3 replay + 5 profiler 且 KernelCert 通过：PASS。
- 这些是评分器校准候选，不进入正式 FA1–FA4 排名。
- pilot、raw matched blocks、profiler、provenance 与 SHA-256 清单均随 ZIP 提供。

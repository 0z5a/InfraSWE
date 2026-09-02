# InfraSWE v0.3 垃圾 Kernel 负控报告

负控总门禁：**PASS**  
评分生成时间：`2026-08-31T11:52:47.015883Z`

硬件 cell：`nvidia-a100-sxm4-40gb-sm80-df09afe222d8`；NVIDIA A100-SXM4-40GB；CC 8.0；显存 39.5 GiB；Driver 595.58.03。
这些候选仅用于验证评分器；它们与正式 FA1–FA4 榜单隔离。每个候选仍执行 3 次独立 replay、每 case 30 个 matched blocks，以及 5 个独立 profiler 进程。

| Backend | 故意缺陷 | Cert | 处置 | Artifact-100 | Effective | 预期命中 | Failure codes |
|---|---|---:|---|---:|---:|---:|---|
| garbage-slow-fa4-waste64 | 正确结果后追加 64 轮无用 Triton 显存流量 | PASS | valid | 58.28 | 58.28 | PASS | — |
| garbage-zero-triton | 原生 Triton kernel 始终写零 | FAIL | valid | — | 0.00 | PASS | CORRECTNESS_MANDATORY_FAILED |
| garbage-cache-copy | 按 shape 缓存第一次正确答案，之后只做 Triton copy | FAIL | valid | — | 0.00 | PASS | CORRECTNESS_MANDATORY_FAILED |
| fa-garbage-math-fallback | 伪装成 FA backend，实际调用 PyTorch SDPA math | FAIL | invalid | — | — | PASS | FALLBACK_NATIVE_TRACE_MISSING |

## 预期

- `garbage-slow-fa4-waste64`：KernelCert 通过，但 Artifact-100 明显受罚（<80）。
- `garbage-zero-triton`：correctness fail，leaderboard effective score=0。
- `garbage-cache-copy`：动态输入探针 fail，leaderboard effective score=0。
- `fa-garbage-math-fallback`：原生 Flash trace 缺失，证据无效且不评分。

## 逐 case 诊断

| Backend | Case | Correctness | Candidate µs | Baseline µs | Raw speedup | AnchorScore | Anchor status |
|---|---|---:|---:|---:|---:|---:|---|
| garbage-slow-fa4-waste64 | boundary-b3-s1000-h12-d64-causal | PASS | 2700.867 | 2067.402 | 0.765× | 0.433 | scored |
| garbage-slow-fa4-waste64 | common-b1-s2048-h16-d128-causal | PASS | 2815.793 | 4336.154 | 1.541× | 0.608 | scored |
| garbage-slow-fa4-waste64 | common-b2-s1024-h16-d64-causal | PASS | 2699.519 | 1965.054 | 0.728× | 0.421 | scored |
| garbage-slow-fa4-waste64 | common-b4-s512-h16-d64-noncausal | PASS | 2688.967 | 967.817 | 0.360× | 0.262 | scored |
| garbage-slow-fa4-waste64 | stress-b1-s4096-h8-d128-causal | PASS | 2986.524 | 8583.417 | 2.874× | 0.748 | scored |
| garbage-zero-triton | boundary-b3-s1000-h12-d64-causal | FAIL | 13.486 | 2066.187 | 153.084× | — | quarantined |
| garbage-zero-triton | common-b1-s2048-h16-d128-causal | FAIL | 14.434 | 4333.183 | 300.318× | — | quarantined |
| garbage-zero-triton | common-b2-s1024-h16-d64-causal | FAIL | 13.227 | 1965.811 | 148.458× | — | quarantined |
| garbage-zero-triton | common-b4-s512-h16-d64-noncausal | FAIL | 11.962 | 967.942 | 81.005× | — | quarantined |
| garbage-zero-triton | stress-b1-s4096-h8-d128-causal | FAIL | 14.441 | 8581.412 | 594.527× | — | quarantined |
| garbage-cache-copy | boundary-b3-s1000-h12-d64-causal | FAIL | 14.961 | 2065.357 | 138.024× | — | quarantined |
| garbage-cache-copy | common-b1-s2048-h16-d128-causal | FAIL | 15.139 | 4335.142 | 286.642× | — | quarantined |
| garbage-cache-copy | common-b2-s1024-h16-d64-causal | FAIL | 15.042 | 1966.804 | 130.909× | — | quarantined |
| garbage-cache-copy | common-b4-s512-h16-d64-noncausal | FAIL | 13.204 | 967.329 | 73.394× | — | quarantined |
| garbage-cache-copy | stress-b1-s4096-h8-d128-causal | FAIL | 14.944 | 8580.465 | 574.318× | — | quarantined |
| fa-garbage-math-fallback | boundary-b3-s1000-h12-d64-causal | PASS | 2064.022 | 2063.565 | 1.000× | 0.500 | scored |
| fa-garbage-math-fallback | common-b1-s2048-h16-d128-causal | PASS | 4332.401 | 4334.149 | 1.000× | 0.500 | scored |
| fa-garbage-math-fallback | common-b2-s1024-h16-d64-causal | PASS | 1965.147 | 1964.390 | 1.000× | 0.500 | scored |
| fa-garbage-math-fallback | common-b4-s512-h16-d64-noncausal | PASS | 966.162 | 967.364 | 1.001× | 0.500 | scored |
| fa-garbage-math-fallback | stress-b1-s4096-h8-d128-causal | PASS | 8584.268 | 8584.177 | 1.000× | 0.500 | scored |

## 判读

- correctness 或动态输入门禁失败属于有效失败：不发布 Artifact 分，排行榜有效分为 0。
- profiler 缺少声明的原生 Flash 路径属于证据无效：保持 N/A，不把可疑 timing 当成 0 分。
- 正确但浪费工作的候选仍可被评分，低 Artifact-100 用于验证性能公式确实产生惩罚。
- 所有 raw block、profiler event、provenance 和 SHA-256 清单均随 ZIP 提供。

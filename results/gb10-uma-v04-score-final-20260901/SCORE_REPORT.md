# GB10 UMA v0.4 正式跑分报告

- 日期：2026-09-01（Asia/Shanghai）
- 任务：`gb10-uma-cpu-gpu-pipeline-v04`
- 协议：`gb10-uma-load-normalized-v0.4-r1`
- 评分权威：InfraSWE scoring second audit RFC v0.4
- Benchmark cell：`sha256:5a44dc3e8aa5427ae8548feddce9b9725451822e33c184d0c79397fe4c409691`
- 证据等级：`E2-system-trace`

## 结论

| 输出 | 结果 |
|---|---:|
| Deployability-100 | **91.7834** |
| Concurrent Stability（C） | 0.826520 |
| Kernel Reuse（U） | 1.000000 |
| Maintainability（M） | 1.000000 |
| InfraCert | pass |
| Deployability 状态 | scored |
| CellArtifact-100 | unresolved |

正式公式：

```text
100 × 0.826520^0.45 × 1.0^0.30 × 1.0^0.25 = 91.783381
```

C、U、M 分别高于 0.60、0.55、0.50 的 v0.4 分项地板。

该分数只对应 GB10 UMA CPU/GPU pipeline 任务，不是完整 GB10 最小发布 track 的总分。完整 track 仍为 3/5 已认证、2 个 PTX 9.3 特性未决，状态保持 `partial`。

## 正式并发协议

- Reference：pageable host → H2D → GPU-private kernel → D2H。
- Reference saturation anchor：72,548.075 requests/s。
- SLO：2.000 ms，由冻结公式 `max(2 ms, 4 × reference low-load p95)` 得到。
- 6 个 load cell、7 次 fresh-process replay、每格每次 1,200 请求。
- 候选请求样本：50,400；交错 reference 样本：8,400；总计保存 58,800 条请求级记录。
- 奇数 replay 为 reference-before-candidate，偶数为 reference-after-candidate；候选 load cell 顺序正反交替。
- 7/7 reference 漂移门通过；观测 reference throughput 为 70,841–72,174 requests/s。

| Load cell | 负载比例 | Median p99 | SLO goodput | Replay jitter score | Cell C |
|---|---:|---:|---:|---:|---:|
| light | 0.25 | 591.2 µs | 1.000 | 0.4582 | 0.8890 |
| normal | 0.50 | 488.0 µs | 1.000 | 0.3847 | 0.8660 |
| knee | 0.80 | 269.3 µs | 1.000 | 0.2520 | 0.8128 |
| saturation | 1.00 | 334.4 µs | 1.000 | 0.2296 | 0.8015 |
| overload | 1.20 | 291.7 µs | 1.000 | 0.2039 | 0.7874 |
| burst/soak | 1.00 avg | 549.0 µs | 1.000 | 0.2838 | 0.8274 |

所有正式进程均满足：

- 1,200/1,200 请求完成且输出正确；
- error/drop rate 为 0；
- 无 deadlock、livelock、无界队列增长或静默回退；
- 四租户 Jain fairness 为 1.0；
- RSS 正增长低于冻结的 64 MiB 门限。

C 没有得到满分，主要扣分来自 fresh-process p95 抖动；该抖动按预注册 `1/(1+4×CV)` 公式计入，没有在看到结果后调整。

## 复用与维护性

U=1.0 的机器证据：

- 8/8 加权 case 由同一 `uma_transform_kernel` implementation family 覆盖；
- 实际 semantic binary variant 数为 1，低于 expected budget 4 和 max budget 12；
- JIT、recompile、runtime specialization cache 均为 0；
- `sm_121` 实机运行、`sm_121f` 和 `sm_121a` compatibility build 均通过；
- PTX/SASS/source 扫描未发现 cublas、CUTLASS、SM100、TMEM 或 `tcgen05` 回退。

M=1.0 的确定性证据：

- contract checks：5/5；
- locality checks：4/4；
- maintenance probes：8/8；
- build checks：6/6；
- workspace、capability-off、allocation failure 和 runtime minor 均产生预期的结构化拒绝码；
- 两次 PTX 构建哈希一致，构建无 warning。

这些满分表示本次预注册 probe 集全部通过，不表示对任意未来修改作主观“代码质量满分”声明。

## E2 与 CellArtifact 状态

Nsight Systems E2 跟踪成功：

- 观测到 260 次 `uma_transform_kernel` 实例；
- 只加载 1 个 CUDA library/module；
- raw `.nsys-rep`、SQLite 和标准化统计均已保存并参与哈希。

同机无 profiler 校准结果：

- pageable-system memory copy calibration：165.157 GB/s；
- launch floor：2.560 µs；
- 128 MiB 代表性 transform 的独立无 profiler median：1.38072 ms；
- 可诊断的 useful bandwidth：97.209 GB/s；
- 可诊断的 SOL ratio：0.5886。

上述效率值不进入正式 CellArtifact。原因是宿主 NVIDIA 参数为 `RmProfilingAdminOnly: 1`，容器内 NCU 返回 `ERR_NVGPUCTRPERM`，无法取得 `Bytes_actual` 和 traffic amplification。按 v0.4，E3、SOL、memory-band score 与 CellArtifact 均保持 `unresolved`，没有用 0 或估算值补洞。

## 硬门与质量门

| 门 | 结果 |
|---|---:|
| Correctness / case portfolio | pass |
| Fallback | pass |
| Liveness / 7 fresh replays | pass |
| Reference drift | pass |
| E2 system trace | pass |
| E3 kernel counter | unresolved |
| Ruff | pass |
| Pytest | 114 passed |
| Schema freshness | 10 fresh |
| `score.json` Schema | valid |

E3 未决不会让 Deployability 失效，因为 v0.4 对 Deployability 的最低证据为 E2；它只阻止 cell-local efficiency 与 CellArtifact 出数。

## 证据入口

- `score.json`：规范 v0.4 分数。
- `summary.json`：精简机器摘要。
- `hard-gates.json`：正确性、回退、活性、reference drift 与证据门。
- `dimension-results.json`：C/U/M 公式输入和 load-cell 分数。
- `concurrency/`：58,800 条请求样本及每次进程记录。
- `reference-anchor/`：三次参考校准与 saturation sweep。
- `reuse-evidence.json`、`maintainability-evidence.json`：U/M 机器证据。
- `profilers/system-trace/`：E2 原始和标准化证据。
- `profilers/kernel-counter/ncu.csv`：E3 权限失败原始证据。
- `pre-registration/`：执行前冻结的 task contract 和协议。
- `track-context/gb10-minimum-suite.json`：完整 GB10 track 的 partial 状态。
- `source-snapshot.tar.gz`：可复现源码快照。
- `SHA256SUMS`：证据包相对路径完整性清单。

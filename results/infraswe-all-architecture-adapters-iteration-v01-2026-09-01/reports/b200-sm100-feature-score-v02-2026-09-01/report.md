# InfraSWE B200 / SM100 Phase-1 特性测试与跑分

状态：`scored`；3 次 fresh-process replay。
GPU：`NVIDIA B200`；CC `10.0`；SM `148`；CUDA runtime `13.0`。

## 总分

| 命名空间 | 分数 | 状态 |
|---|---:|---|
| SM100-Core | 95.95 | scored |
| SM100-Scheduler | 75.69 | scored |
| SM100-Fabric | N/A | not_applicable |
| PTX-Preview | N/A | disabled |

## Phase-1 五任务

| 任务 | 得分 | C | N | P | R | B | 性能 candidate / baseline |
|---|---:|---:|---:|---:|---:|---:|---:|
| BW-TMEM-001 | 99.25 | 1.000 | 1.000 | 0.979 | 1.000 | 1.000 | 0.9694× |
| BW-CLC-001 | 65.27 | 1.000 | 1.000 | 0.008 | 1.000 | 1.000 | 0.9996× |
| BW-TMA-001 | 82.68 | 1.000 | 1.000 | 0.505 | 1.000 | 1.000 | 0.8431× |
| BW-TMEM-003 | 99.98 | 1.000 | 1.000 | 0.998 | 1.000 | 1.000 | 0.9608× |
| BW-TMA-002 | 99.81 | 1.000 | 1.000 | 0.994 | 1.000 | 1.000 | 0.8611× |

## 性能摘要

- `BW-TMEM-001`：candidate `92.674 µs`，baseline `95.602 µs`，speedup `1.032×`。
- `BW-CLC-001`：candidate `95.083 µs`，baseline `95.121 µs`，speedup `1.000×`。
- `BW-TMA-001`：candidate `3.048 µs`，baseline `3.615 µs`，speedup `1.186×`。
- `BW-TMEM-003`：candidate `6.145 µs`，baseline `6.396 µs`，speedup `1.041×`。
- `BW-TMA-002`：candidate `79.711 µs`，baseline `92.573 µs`，speedup `1.161×`。

## 证据与边界

- 五个任务均要求三轮正确性、watchdog/liveness 与 PTX+cubin+SASS 原生门禁同时通过；任一 hard gate 失败，该任务为 0 分。
- `BW-TMEM-001` 覆盖 aligned、M/N/K tail 与非默认 leading-dimension；`BW-TMEM-003` 额外执行数千次 launch 以及非法对齐显式拒绝。
- `BW-CLC-001` 比较 dynamic CLC 与 static persistent 的 uniform/tail makespan；原生证据必须同时出现 `clusterlaunchcontrol.*` 与 `UGETNEXTWORKID`。
- `BW-TMA-001` 运行连续、离散、重复、逆序和边界 row case；`BW-TMA-002` 比较 2-CTA 与 1-CTA pipeline。
- Fabric 在当前单卡 lease 中为 N/A，不以 0 分惩罚；PTX Preview 保持 disabled。
- 分数遵循 RFC 的 C/N/P/R/B 结构；性能使用 latency 的对数 anchor 插值。

参考：https://docs.nvidia.com/cuda/parallel-thread-execution/

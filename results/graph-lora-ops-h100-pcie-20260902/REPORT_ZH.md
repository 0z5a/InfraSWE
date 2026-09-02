# SGLang Graph LoRA `addmm(out=...)` H100 benchmark

日期：2026-09-02  
主项目 Draft：SGLang，`D3-contract-proposed`  
硬件：1× NVIDIA H100 PCIe 80GB，SM90  
软件：PyTorch 2.8.0+cu128，CUDA 12.8，Python 3.12.12

## 结论

候选实现值得继续，但本轮冻结判定为 **REVISE**，不能给出官方 ProjectFit。

- `slots >= 4` 的四个目标用例全部显著提速：**1.17×–1.63×**，延迟下降 **14.28%–38.49%**。
- 候选的 eager、CUDA Graph 首次/动态重放、alias 和特殊契约共 **189/189** 通过；基线同样为 **189/189**。
- 冻结的每用例最大回退门槛为 3%。`control-slots1-fp16` 配对比值为 **1.0360**，因此按协议判定门槛失败。
- 该 control 不进入候选新分支。baseline-vs-baseline A/A 负对照也得到 **1.0301**，说明配对夹具的第二路分配/地址偏置本身已达到约 3%。按 A/A 归一化后，候选 control 的诊断比值为 **1.0057**。这项诊断不覆盖冻结门槛结果。

## 性能结果

绝对延迟来自两组独立 7 fresh-process 测量；相对倍率以 source-identified v2 交错配对测量为准。

| 用例 | 基线中位数 µs | 候选中位数 µs | 配对倍率 | 延迟变化 | 3% 门槛 |
|---|---:|---:|---:|---:|---|
| control slots=1 FP16 | 9.160 | 9.535 | 0.965× | -3.60% | FAIL |
| control slots=3 BF16 | 28.113 | 28.036 | 1.007× | +0.72% | PASS |
| decode slots=4 r16 FP16 | 32.751 | 27.028 | 1.210× | +17.34% | PASS |
| batch slots=4 r32 BF16 | 37.577 | 31.627 | 1.167× | +14.28% | PASS |
| QKV slots=4 r32 FP16 | 82.033 | 50.573 | 1.626× | +38.49% | PASS |
| wide slots=8 r64 BF16 | 78.549 | 64.780 | 1.238× | +19.20% | PASS |

每组配对证据使用 7 个全新进程、每用例每进程 5 个样本、每样本 300 次 CUDA Graph replay。另跑了相同规模的 baseline-vs-baseline A/A 负对照。总计归档 28 个 fresh-process JSON：基线绝对 7、候选绝对 7、候选配对 v2 7、A/A v2 7。v1 文件原样保留，但最终相对统计使用带两侧源码 SHA 的 v2。

## InfraSWE 评分

| 项目 | 值 |
|---|---:|
| Evolutionary Maintainability | 0.683020 |
| Project Contract Fit | 0.901250 |
| Performance Reuse Utilization | 0.999470 |
| Operational Fit | 0.812252 |
| 诊断 ProjectFit-100 | **81.4986** |
| BenchmarkTrust-100 | **93.0605** |
| 官方 ProjectFit | **not issued** |
| 判定 | **REVISE** |

81.4986 只是冻结公式下的诊断分，不是排行榜分。官方分未签发的直接原因是冻结 3% 门槛失败；即使修复该门槛，当前仍缺少：

1. 覆盖 `slots=3/4` 边界的 SGLang registered test；
2. E2 SGLang server 集成 trace；
3. SM80、SM89 必需部署 cell；
4. hidden probes 和 maintainer-sealed Draft。

Maintainability 的 `evolution=0.5`：候选注释说明了阈值原因，但 `num_loras >= 4` 仍是不可配置的硬编码启发式。`tests=0.5` 和 Contract Fit 的 `buildtest=0.5`：外部新分支探针通过，但项目内 registered branch test 缺失。

## 建议下一步

1. 将配对夹具改为共享静态 buffer，或对 allocation/capture order 做完整 counterbalance，再冻结并重跑 9 µs control。
2. 在 SGLang 项目内补 `slots=3` fallback 和 `slots=4` direct-accumulation 边界测试。
3. 用实际 SGLang server 请求跑 E2 trace，再扩到 SM80、SM89。

## 证据身份

- 候选源码 SHA256：`e4c9b3bdf5a5488f85a17abcb2ad3089d4d7b2b2024845bd2752f97067bc0b76`
- 基线源码 SHA256：`390676cc38975095ba124d1932aa8c27dc2b4605e18c495c095ae852f4c96790`
- 最终 comparison summary SHA256：`32f1a5a66e75a7233eda6c3e7fba2c80bedee3db0a285e3e272d0bfa803bbd1b`
- 配对脚本 SHA256：`1691e7f3f5839967f961934c610aed05a3400e79d869d401b54f18b0a007e418`

详细原始路径、逐份 canonical digest、配对分布、A/A 诊断和评分组件均记录在 `candidate/comparison-summary.json`。

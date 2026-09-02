# SGLang Graph LoRA SM120 综合多 Draft 评测报告

日期：2026-09-02  
硬件：1× NVIDIA RTX PRO 4000 Blackwell 24GB，SM120  
最终判定：**PASS（本硬件、本源码身份与本协议）**

## 执行摘要

- 冻结核心评测：候选正确性与契约 **189/189** 通过，六个核心 case 的每 case 最大 3% retention gate 全部通过。
- 扩展多 Draft 评测：覆盖 **5 个 Draft profile、21 个 case、7 个 fresh process、每 case 35 个配对样本**。
- 扩展候选正确性与契约：**637/637** 通过；21/21 case 的 retention gate 通过。
- SGLang 原生目标 Draft 的几何平均加速为 **1.183×**，最高为 **1.529×**；`slots=3` fallback 边界基本持平。
- 全部 proxy workload 的最大回退为 **0.52%**，低于 3% 门槛；最高加速为 **1.564×**。
- InfraSWE 冻结诊断 ProjectFit：**81.5065/100**；BenchmarkTrust：**93.0605/100**。
- 官方 ProjectFit 尚未签发；本报告不把其他项目的 contract proxy 冒充为跨项目原生分数。

## 被测对象与环境

- 被测函数：`sgemm_lora_b_graph_fwd`
- 基线 SHA256：`390676cc38975095ba124d1932aa8c27dc2b4605e18c495c095ae852f4c96790`
- 候选 SHA256：`8842d8b62f766265142e341e5acc59492ccb41756237bcd9a217568311a76402`
- PyTorch：2.8.0+cu128
- CUDA runtime：12.8
- Python：3.12.14
- GPU compute capability：12.0

候选在 `num_loras >= 4` 且禁用梯度时使用直接 `addmm_` 累加；低 slot、梯度路径和 dtype 不匹配继续走原有 `mm + add_` fallback。

## 评测协议

核心评测分别运行 7 个基线、7 个候选和 7 个交错配对 fresh process。多 Draft 扩展评测再运行 7 个独立 fresh process，并在每个进程中：

1. 对基线和候选执行 eager、CUDA Graph 首次 replay、动态 replay 与 alias 检查；
2. 对每个 case 采集 5 个交错配对样本；
3. 每个样本执行 300 次 CUDA Graph replay；
4. 对 baseline/candidate capture order 和样本执行顺序进行反平衡；
5. 以 7 个进程内配对中位数的跨进程中位数作为相对性能判定；
6. 使用 `candidate / baseline <= 1.03` 作为每 case retention gate。

正式执行前 GPU 上没有其他 compute 进程，所有计时阶段均固定使用 GPU 0。

## 冻结核心结果

| Case | 基线中位数 µs | 候选中位数 µs | 配对加速 | 延迟变化 | 3% gate |
|---|---:|---:|---:|---:|---|
| control slots=1 FP16 | 6.181 | 6.213 | 0.997× | -0.28% | PASS |
| control slots=3 BF16 | 20.502 | 20.502 | 1.000× | -0.00% | PASS |
| decode slots=4 r16 FP16 | 22.566 | 20.502 | 1.142× | +12.41% | PASS |
| batch slots=4 r32 BF16 | 26.662 | 22.568 | 1.182× | +15.37% | PASS |
| QKV slots=4 r32 FP16 | 53.300 | 34.855 | 1.529× | +34.59% | PASS |
| wide slots=8 r64 BF16 | 65.536 | 55.931 | 1.184× | +15.55% | PASS |

`slots >= 4` 的四个核心目标 case 全部提速，范围为 **1.142×–1.529×**。两个未进入新分支的 control 基本持平。

## 多 Draft 结果

### Profile 汇总

| Draft profile | 角色与证据权限 | Cases | 几何平均加速 | 最低–最高加速 | 正确性 | 3% gate |
|---|---|---:|---:|---:|---|---|
| SGLang `sglang-runtime-kernel-v1` | 主 host，原生目标 contract | 5 | 1.183× | 1.000×–1.529× | PASS | PASS |
| CUTLASS/CuTe `cutlass-cute-kernel-library-v1` | 主 peer，dense-GEMM proxy | 4 | 1.138× | 1.080×–1.222× | PASS | PASS |
| vLLM `vllm-kernel-integration-v1` | 次级 host workload proxy | 4 | 1.202× | 1.000×–1.506× | PASS | PASS |
| Megatron-Core `megatron-core-training-kernel-host-v1` | 次级 host workload proxy | 4 | 1.200× | 0.995×–1.564× | PASS | PASS |
| DeepGEMM `deepgemm-moe-gemm-kernel-v1` | 次级 peer，dense-GEMM proxy | 4 | 1.242× | 1.161×–1.458× | PASS | PASS |

只有 SGLang 行具有当前源码的原生目标 contract 权限。其余四行验证 shape、lifecycle 和 dense-GEMM contract 的代理兼容性，不构成 vLLM、Megatron-Core、CUTLASS/CuTe 或 DeepGEMM 的原生 ProjectFit 分数。

### 21 个扩展 case

| Profile | Case | 加速 | 延迟变化 | 3% gate |
|---|---|---:|---:|---|
| SGLang | slots=3 boundary BF16 | 1.000× | -0.00% | PASS |
| SGLang | slots=4 boundary BF16 | 1.181× | +15.32% | PASS |
| SGLang | decode tiny-M FP16 | 1.101× | +9.14% | PASS |
| SGLang | QKV multi-slice FP16 | 1.529× | +34.60% | PASS |
| SGLang | slots=16 wide BF16 | 1.166× | +14.25% | PASS |
| CUTLASS/CuTe proxy | tiny-M FP16 | 1.222× | +18.16% | PASS |
| CUTLASS/CuTe proxy | skinny BF16 | 1.091× | +8.31% | PASS |
| CUTLASS/CuTe proxy | non-aligned K FP16 | 1.167× | +14.28% | PASS |
| CUTLASS/CuTe proxy | non-aligned N BF16 | 1.080× | +7.42% | PASS |
| vLLM proxy | slots=1 control FP16 | 1.000× | -0.01% | PASS |
| vLLM proxy | adapter burst FP16 | 1.151× | +13.09% | PASS |
| vLLM proxy | QKV BF16 | 1.506× | +33.60% | PASS |
| vLLM proxy | mixed batch BF16 | 1.206× | +17.10% | PASS |
| Megatron-Core proxy | large batch BF16 | 1.157× | +13.57% | PASS |
| Megatron-Core proxy | TP slices FP16 | 1.564× | +36.06% | PASS |
| Megatron-Core proxy | high rank BF16 | 1.154× | +13.31% | PASS |
| Megatron-Core proxy | slots=3 control BF16 | 0.995× | -0.52% | PASS |
| DeepGEMM proxy | tiny-M FP16 | 1.199× | +16.58% | PASS |
| DeepGEMM proxy | throughput BF16 | 1.161× | +13.84% | PASS |
| DeepGEMM proxy | non-aligned K BF16 | 1.175× | +14.86% | PASS |
| DeepGEMM proxy | multi-slice FP16 | 1.458× | +31.41% | PASS |

## 扩展 contract probes

以下 7 项在基线和候选的全部 7 个 fresh process 中均通过：

| Probe | 候选结果 |
|---|---|
| empty weights | PASS |
| mixed output dtype fallback | PASS |
| 基础 gradient path | PASS |
| slots=4 gradient fallback | PASS |
| `base_output=None` 分配语义 | PASS |
| zero-token 输入 | PASS |
| concurrent CUDA streams | PASS |

候选的 21 个 case 每轮包含 3 个 correctness phase、1 个 alias 检查和 7 个 contract probe，总计 **637/637** 通过。基线同样为 **637/637**。

## InfraSWE 评分

| 项目 | 分数 |
|---|---:|
| Evolutionary Maintainability | 68.3020 |
| Project Contract Fit | 90.1250 |
| Performance Reuse Utilization | 99.9958 |
| Operational Fit | 81.2252 |
| 诊断 ProjectFit | **81.5065/100** |
| BenchmarkTrust | **93.0605/100** |
| 官方 ProjectFit | **未签发** |

多 Draft 扩展证据支持冻结诊断结论，但不会追溯修改冻结公式或把 proxy profile 转换为官方跨项目分数。综合分的主要限制仍是：

- `num_loras >= 4` 是硬编码、不可配置的启发式阈值；
- 附件没有 SGLang 项目内 registered branch test；
- 当前证据为 E1 单算子 CUDA Graph 级别，不是 E2 SGLang server trace；
- 默认 SGLang Draft 尚未由维护者 Seal，hidden probes 未完成；
- 冻结要求的 SM80、SM89、SM90 部署单元仍不完整。

## 静态检查与已知限制

- `py_compile`：通过。
- 新增 multi-Draft benchmark 与聚合器：Ruff 通过。
- 被测候选与基线均有相同的 3 个既有 Ruff 项（两个未使用解包变量、一个 `Optional` 注解风格项），本次候选没有新增静态告警。
- 本评测不包含真实模型权重、SGLang HTTP server、continuous batching 调度器或 TP 多进程通信。
- SM120 结果不能直接外推到 SM80、SM89、SM90 或其他软件栈。

## 证据与复现入口

- 核心原始证据：`baseline/raw/`、`candidate/raw/`、`paired/raw/`
- 多 Draft 原始证据：`multidraft/raw/replay-1.json` 至 `replay-7.json`
- 多 Draft 聚合：`multidraft/summary.json`
- 多 Draft summary SHA256：`e3ec284c29df1766fd9dee3bda8fe99043f46fa66a5b0529c0e6c5ae5a80c01d`
- 评测器 SHA256：`a3ebf7b19f883ab6b5d9ad67fcc4ee76c405d155244db2094fbcc6f057fe9dfd`
- 聚合器 SHA256：`14c9a7c130d8758766b66e9f18eecb405834babee7d99317e16f4359c3481fcc`
- 远端目录：`/root/infraswe-copy/results/graph-lora-ops-sm120-20260902`

## 结论与下一步

候选在 SM120 上通过冻结核心门槛和扩展多 Draft contract 矩阵。`slots >= 4` 的目标 workload 获得稳定、可重复的收益，低 slot 与梯度 fallback 未出现有意义的回退。当前证据足以支持继续进入 SGLang 项目内 registered test 和 E2 server 集成阶段，但不足以签发官方 ProjectFit 或宣称其他项目的原生兼容性。

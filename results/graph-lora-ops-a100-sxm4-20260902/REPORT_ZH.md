# graph LoRA 多 Draft A100-SXM4 实测报告（2026-09-02）

## 结论

当前 revised candidate 在冻结的 A100 SM80 Cell 上通过本次 benchmark：

- `overall_passed = true`；
- 7/7 个 fresh process 完整结束；
- 21/21 个 case 的 baseline/candidate correctness 均通过；
- candidate 的 7/7 项特殊合同在全部 replay 中通过；
- 21/21 个 case 的 median-of-process paired ratio 均满足
  `candidate / baseline <= 1.03`；
- SGLang 主 Draft 的几何平均加速为 **1.2086×**；
- 全部 case 中最差为 `sglang-slots3-boundary-bf16 = 0.9931×`，即 candidate
  慢约 0.70%，仍在预注册保留门内；
- 全部 case 中最好为 `megatron-tp-slices-fp16 = 1.5977×`。

这是一份真实 SM80 deployment-cell evidence，不改写此前 H100 或 SM120 结果。

## 执行身份

```text
host: 38.49.42.120:54270
GPU used: NVIDIA A100-SXM4-40GB, device 0
available GPUs: 2
compute capability: 8.0
GPU0↔GPU1 topology: NODE（不是 NVLink）
torch: 2.8.0+cu128
CUDA runtime: 12.8
Triton: 3.4.0
```

本 benchmark 只使用 GPU0；GPU1 不参与 LoRA 单卡 kernel timing。两卡拓扑证据保存在
`preflight/nvidia-smi-topology.txt`。

## 冻结协议

正式运行前已冻结：

```text
fresh processes: 7
Draft profiles: 5
cases: 21
paired samples per case per process: 5
CUDA Graph replays per timed sample: 300
paired observations per case: 35
total paired observations: 735
sample/capture order: counterbalanced
retention gate: every case candidate/baseline <= 1.03
```

先运行的 smoke 只验证环境与合同，不进入 7 次正式聚合。Smoke 单样本曾出现明显噪声，
但没有据此修改 case、阈值、sample 数或运行顺序。

## Source identity

| 对象 | SHA-256 | 用途 |
|---|---|---|
| baseline | `390676cc38975095ba124d1932aa8c27dc2b4605e18c495c095ae852f4c96790` | 正式 baseline |
| revised candidate | `8842d8b62f766265142e341e5acc59492ccb41756237bcd9a217568311a76402` | 本次正式 candidate |
| 原始附件 `graph_lora_ops(1).py` | `e4c9b3bdf5a5488f85a17abcb2ad3089d4d7b2b2024845bd2752f97067bc0b76` | 明确不在本次测试中 |

`multidraft/summary.json` SHA-256：

```text
e384167f22545d00ec4f4bcdf04db60f2c1688ea856316814d56e5005e33fc6d
```

## Draft 结果

| Draft profile | 角色 | cases | 几何平均加速 | 最差加速 | 最好加速 | correctness / retention |
|---|---|---:|---:|---:|---:|---|
| `sglang-runtime-kernel-v1` | native target | 5 | 1.2086× | 0.9931× | 1.5964× | PASS / PASS |
| `cutlass-cute-kernel-library-v1` | contract proxy | 4 | 1.1370× | 1.0540× | 1.2331× | PASS / PASS |
| `vllm-kernel-integration-v1` | contract proxy | 4 | 1.2165× | 1.0080× | 1.4975× | PASS / PASS |
| `megatron-core-training-kernel-host-v1` | contract proxy | 4 | 1.2153× | 1.0002× | 1.5977× | PASS / PASS |
| `deepgemm-moe-gemm-kernel-v1` | contract proxy | 4 | 1.2216× | 1.0965× | 1.4939× | PASS / PASS |

只有 SGLang 是 native target contract。其余四个 Draft 是同一 LoRA kernel 的 contract
proxy，不构成这些项目的 native 集成、workload 或 ProjectFit 证据。

## 合同覆盖

Candidate 在全部 7 次 fresh replay 中通过：

```text
empty_weights
mixed_output_dtype
gradient
gradient_slots4_fallback
base_output_none
zero_tokens
concurrent_streams
```

每个 performance case 同时检查 eager、CUDA Graph first replay 与修改输入后的 dynamic
replay；baseline 与 candidate 都必须 finite、数值一致，并保持 `base_output` alias 合同。

## InfraSWE 权威边界

本次可以正式声明的是：

```text
SM80 multi-Draft kernel benchmark gate = PASS
```

本次不能声明：

```text
official ProjectFit-100
cross-project absolute ranking
proxy project native integration pass
2-GPU communication / topology certification
```

原因是预注册已明确 `official_projectfit_allowed=false`，且没有为各 proxy project 运行
完整的 target profile、真实集成 workload、C/R/O 与 sealed acceptance contract。为满足
“真正合并必须 >=85”的政策，不能把本 microbenchmark 的加速率伪装成 ProjectFit 分数。

## 产品回归与证据

远端同一环境在 benchmark 前完成：

```text
ruff: All checks passed
pytest: 241 passed, 1 non-fatal NumPy warning
schema: 66 fresh
```

原始 evidence 位于：

```text
pre-registration/multidraft-plan.json
preflight/
smoke/
multidraft/raw/replay-1..7.json
multidraft/summary.json
verification/
inputs/
```

Vast `/workspace` 不是持久卷；以上文件已在实例仍在线时同步回本地，本报告不依赖远端
继续存活。

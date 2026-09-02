# InfraSWE v0.5 Draft Loop 与训练远端迭代报告

日期：2026-09-01  
远端：`root@118.163.199.123:31409`  
状态：实现与远端诊断通过；Draft、InfraCert、BenchmarkTrust 和官方分数未决

## 结论

本轮已把 v0.5 Draft 准入层、ProjectFit 评分层、默认项目 Catalog 和 Draft 来源回退策略落成
可执行实现，并继续完成 native PyTorch 训练适配的两张 L40S 实跑。代码在本地和远端均为
`155 passed`，Ruff 全绿，28 份协议 schema fresh。远端 7 个 fresh process、两卡交替执行、
2-rank NCCL 和 G2 PyTorch profiler trace 均成功。

这不是官方 ProjectFit 或 Deployability 跑分。当前默认 vLLM profile 是机器提炼的
`proposed` 版本，Draft 停在 D3，没有项目维护者 Review 和 Seal；远端没有 NSYS/G3 system
trace，NCU 又被宿主的 GPU counter 权限挡住。因此：

```text
InfraCert                         = UNRESOLVED
ProjectFit-100                    = not issued
Training Deployability-100 v0.4  = not issued
BenchmarkTrust                    = UNRESOLVED
BenchmarkCost                     = partial
MergeabilityDecision              = UNRESOLVED
```

minimum-suite 中的 `92.4081328330388` 只是合成 G3 scorer fixture，用于证明公式边界，明确
不可发布为硬件分数或排行榜分数。

## 默认 Draft 规则

来源优先级已经固定为：

```text
显式本地 Draft > 显式远程 Git Draft > 内置默认 Catalog
```

本地与远程同时设置时，本地优先并记录 `REMOTE_GIT_DRAFT_SHADOWED_BY_LOCAL`。远程 Git
Draft 必须指定 `repository + revision + repository-relative path`，只 fetch 指定 revision，
不 checkout 工作树，并拒绝路径穿越。

内置 Catalog 是四个独立的 project profile，不会混成一个跨项目分数：

| 默认项目 | Catalog profile | 固定上游 revision | 主要契约重心 |
|---|---|---|---|
| vLLM | `vllm-kernel-integration-v1` | `40824284…15b0b` | CustomOp/schema/meta/opcheck、native fallback、engine/graph 生命周期、TTFT/TPOT |
| SGLang | `sglang-runtime-kernel-v1` | `4c2c169e…8d9b` | SRT/sglang-kernel 包边界、registered tests、server/scheduler/distributed 生命周期 |
| FlashAttention | `flash-attention-kernel-v1` | `ce088ab9…a820` | exact attention fwd/bwd、varlen/causal/MQA/GQA、CUDA/ROCm generation 与 reference oracle |
| FlashInfer | `flashinfer-kernel-library-v1` | `9d0e6f82…07ea` | include/csrc/python 分层、AOT/JIT/cubin/cache、prefill/decode/KV/MoE/graph |

每个 profile 都物化了 API/ABI、lifecycle、build-test matrix、dependency、fallback、workload、
performance target 和 maintainability probes 八类 artifact。所有上游 URL 和 revision 都已固定；
运行时不会抓取最新 `main` 静默修改规则。

当候选路径或 target hint 精确命中项目 alias 时选择对应 profile；完全没有命中时按用户给出的
冻结顺序 `vllm → sglang → flash-attention → flashinfer` 选择 vLLM，并写入
`DEFAULT_TARGET_SELECTED_BY_PRIORITY`。本轮没有显式本地/远程 Draft，也没有项目 alias，故
解析结果是 `vllm-kernel-integration-v1`，报告中没有把它称为通用最优。

## v0.5 实现边界

- Draft D0–D9 状态模型、一次只前进一步、Candidate Loop 与 Contract Loop 的 revision 隔离。
- D4+ acceptance contract 必须绑定 human-review digest；D6+ 必须为 sealed。
- Seal 只接受 D5、`project-maintainer` 权限、approve 决策和与 Draft 完全匹配的 review；Seal
  采用 canonical SHA-256，任何字段篡改都会失败。
- Project Comparison Cell 固定 target/profile/baseline/contract/probe/workload/target/cells/
  formula/evidence policy/season，并硬编码禁止跨项目排名。
- 普通 ProjectFit 公式固定为 `100*M^0.40*P^0.30*R^0.20*O^0.10`；纯 Triton 使用独立公式，
  隐藏 CUDA/HIP/CANN 路径直接资格失败，不给低 X 后继续上榜。
- component floor、至少 5 次 fresh replay、E2/G3、hidden probes、manifest 和 Seal 都是正式
  分数前置条件；缺项不补权、不填 0。
- BenchmarkTrust、BenchmarkCost、edge objective 和 CellEfficiency 与 candidate 主分分层。
- Provisional 只允许 D5；official/not-acceptable 结果只允许 D8 且必须绑定 Seal。
- affected-case selector 强制保留 positive、negative-control、fallback/unsupported、
  hidden-adjacent、build/import/load 五类，并把环境与 collector 纳入 cache identity。

## 远端环境

| 项目 | 实测值 |
|---|---|
| GPU | 2 × NVIDIA L40S，46,068 MiB/卡 |
| Compute capability | 8.9 (`sm_89`) |
| Driver | 570.211.01 |
| CUDA toolkit | 12.8，nvcc 12.8.93 |
| PyTorch | 2.11.0+cu128 |
| Triton | 3.6.0 |
| Python | 3.12.14，`/venv/main` |
| Jupyter tunnel | 本地 `https://127.0.0.1:8080/` 返回 HTTP 302 |

远端是 Vast 容器，`/workspace` 不是持久 volume。本轮遵循实例指南，未安装或修改 NVIDIA
driver；依赖安装在 `/venv/main`，项目同步到 `/workspace/infraswe`。

## 训练与两卡证据

### 语义与 replay

- 7/7 fresh-process replay 通过，PID 全部不同，GPU 0/1 交替覆盖。
- 每个 replay 都重新构建 reference/candidate；loss、logits、gradients、parameters 的最大绝对
  误差均为 `0.0`，fallback calls 为 `0`。
- 7 次最终结果 digest 完全一致：
  `sha256:0199eb5fafb9b12ea3dee61ee1357d397a40e71a65ed56a06005d0d2b403a802`。
- 每次 10 个权威 unprofiled step，共 70 个样本；中位数 `4.299227 ms`，p95
  `6.431733 ms`，min `2.952476 ms`，max `6.701056 ms`。
- 同一固定 batch 的 loss 从 `4.7267022` 降到 `3.7791831`。
- 每次 replay 最大 allocated memory `18,289,152 bytes`，最大 reserved memory
  `23,068,672 bytes`。

这些时间只描述 tiny FP32 eager reference adapter 的当前 cell，不代表真实模型吞吐、BF16、
FSDP、线上训练或跨硬件排名。

### 两卡与 profiler

- 2-rank NCCL all-reduce 通过：rank 0/1 均得到 `3.0`。
- PyTorch profiler trace 已导出，作为 G2 framework evidence；计时结果来自独立 unprofiled run。
- 远端不存在 NSYS，因此 G3 system timeline 为 `unresolved`。
- NCU 目标 GEMM/SiLU/backward 脚本本身成功，但返回 `ERR_NVGPUCTRPERM`；G4 SOL/DRAM counters
  为 `unresolved`，不是 0，也没有据此生成 CellEfficiency。

DDP 证据只证明 NCCL 两 rank 生命周期和 collective 可执行，不等于完整 DDP/FSDP 训练认证。

## BenchmarkTrust 与 BenchmarkCost

7 次 replay 的 bitwise-stable evidence 支持 `T_repro=1.0`，但缺少 G3 trace、预注册统计 CI、
受控 clock/背景负载以及可读取的 NCU counters。因此 BenchmarkTrust 不计算数值；不能只用
reproducibility 分量重分配权重。

本次成功重跑的 BenchmarkCost 为 partial：wall `37.2116 s`、估算 accelerator allocation
`43.5518 accelerator-s`、profiler `0.07885 s`、8 个执行 case、0 skip。eager 路径的 compile
cache ratio 不适用，serialization/config compatibility 和 Draft 分级计时未完成，均用 null 与
failure code 表示。

## Harness incident

首次 GPU replay 调度时，父进程漏传子进程必需的 `--output-dir`。7 个子进程均被 argparse
拒绝，harness 正确返回 fail；该事件按 RFC 归为 harness failure，不归因候选。补参后从头执行
完整 7 replay，最终全部通过。机器记录见 `harness-incidents.json`。

## 当前未覆盖

- 默认四个 profile 尚未得到任何上游项目维护者 Review，不能发布成 human-reviewed catalog。
- 未实现或认证 Transformers、TRL、verl、torchtune、Axolotl、Megatron Core 的真实 adapter。
- 未跑完整 checkpoint/resume、RNG continuity、BF16/FP16、长序列、gradient accumulation、
  DDP/FSDP2/ZeRO、rollout worker 或真实模型工作负载。
- 未获得 NSYS G3 和 NCU G4 counter 权限，不发布 Deployability、ProjectFit 或 CellEfficiency。
- `/workspace` 非持久卷；本地 evidence pack 是交付依据。

## 关键证据

- `draft-source-resolution.json`：默认来源、vLLM 选择原因、D3 Draft 与 proposed profile。
- `gpu-replays/remote-training-replays.json`：7 replay、两卡、step 分布、DDP 和成本。
- `gpu-replays/torch-profiler-replay-0.json`：G2 framework profiler trace。
- `minimum-suite.json`：4 个正例和 11 个负例；fixture-only 标记保留。
- `training-capabilities.json`：native PyTorch ready；其他框架只到 protocol-supported。
- `ncu-training-kernel.csv`：目标 kernel 通过和 `ERR_NVGPUCTRPERM` 原始记录。
- `benchmark-trust.json` / `benchmark-cost.json` / `evaluation-status.json`：不补权的裁决。
- `evidence-manifest.json` / `SHA256SUMS`：交付文件摘要与复验入口。

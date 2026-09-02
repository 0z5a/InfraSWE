# 架构适配源码导航

## 公共评分与证据层

- `source/src/infraswe/kernel/`：kernel 结果、role graph、统计、评分和 sealing。
- `source/src/infraswe/verifier/`：正确性、replay、fallback 与 workload 验证；
  `native_sm100.py` 是 SM100 原生证据验证器。
- `source/schemas/`：通用 kernel 与 Blackwell capability/dynamic/native/result schema。
- `source/benchmarks/kernel_frontier/score_results.py`：通用 v0.3 scorer。
- `source/benchmarks/kernel_frontier/attention_bench.py`、`classic_bench.py`：
  attention 和经典 kernel 正式测量。
- `source/benchmarks/kernel_frontier/garbage_kernels.py`、`sweep_mediocre.py`：
  负控与分数梯度。

## SM80 / A100

- Profile：`source/profiles/gpu-1x-sm80.toml`、`gpu-2x-sm80-pcie.toml`、
  `gpu-4x-sm80-pcie.toml`、`gpu-4x-sm80-nvlink.toml`。
- Runner：`remote_run_a100.sh`、`remote_run_fa4_a100.sh`、
  `remote_run_negative_controls_a100.sh`、`remote_run_score_gradient_a100.sh`、
  `remote_run_dense_score_gradient_a100.sh`。

## SM90 / H200

- Feature workload：`h200_feature_bench.py`。
- Runner：`remote_prepare_h200.sh`、`remote_run_frontier_h200.sh`、
  `remote_run_h200_features.sh`。
- Summarizer：`summarize_h200_features.py`。
- Supervisor：`source/benchmarks/kernel_frontier/supervisor/kernel-h200-*`。

## SM100 / B200

- Profile：`source/profiles/gpu-1x-sm100-b200-cuda133.toml`。
- Contract：`source/src/infraswe/kernel/blackwell.py`。
- Native verifier：`source/src/infraswe/verifier/native_sm100.py`。
- Workloads：`b200_feature_bench.py`、`b200_tma_gather_scatter.cu`。
- Runner：`remote_prepare_b200_cuda133.sh`、`remote_run_b200_compiler_features.sh`、
  `remote_run_b200_feature_scores.sh`。
- Scorer：`summarize_b200_compiler_features.py`、
  `summarize_b200_feature_scores.py`。

## SM120 / RTX PRO 5000

- 当前 profile：`source/profiles/gpu-2x-sm120-pcie.toml`；正式结果实际来自单卡，
  profile 补齐项见 `ITERATION_BACKLOG.md`。
- Runner：`remote_run_frontier_sm120.sh`、`remote_run_score_gradient_sm120.sh`、
  `remote_pilot_score_gradient_sm120.sh`、`remote_extend_score_gradient_sm120.sh`。
- Probe/构建：`remote_probe_sm120.sh`、`remote_build_fa.sh` 和对应 supervisor。

## gfx942 / MI300X

- Profile：`source/profiles/gpu-1x-gfx942-mi300x-rocm61.toml`。
- 说明：`source/benchmarks/kernel_frontier/MI300X_ROCM61.md`。
- Lease：`source/benchmarks/kernel_frontier/rocm_lease_guard.py`。
- Runner：`remote_prepare_mi300x_rocm61.sh`、
  `remote_run_frontier_mi300x_rocm61.sh`。
- Supervisor：`source/benchmarks/kernel_frontier/supervisor/kernel-mi300x-*`。


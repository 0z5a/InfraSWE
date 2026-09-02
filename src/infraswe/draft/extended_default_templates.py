"""Pinned target-profile templates added after the initial four-project catalog."""

from __future__ import annotations

from typing import Any


def _contract(requirement: str, *probes: str) -> dict[str, Any]:
    return {"requirements": [requirement], "probes": list(probes)}


ADDITIONAL_DEFAULT_PROJECT_TEMPLATES: dict[str, dict[str, Any]] = {
    "cutlass-cute": {
        "catalog_profile": "cutlass-cute-kernel-library-v1",
        "repository": "https://github.com/NVIDIA/cutlass",
        "revision": "dc45f979ae336a235da1676b311f35efeb30149a",
        "aliases": ["cutlass-cute", "nvidia/cutlass", "cutlass"],
        "sources": [
            "https://github.com/NVIDIA/cutlass/blob/"
            "dc45f979ae336a235da1676b311f35efeb30149a/README.md",
            "https://github.com/NVIDIA/cutlass/tree/dc45f979ae336a235da1676b311f35efeb30149a/docs",
            "https://github.com/NVIDIA/cutlass/tree/"
            "dc45f979ae336a235da1676b311f35efeb30149a/test/unit",
        ],
        "integration_points": [
            "include/cutlass",
            "include/cute",
            "examples",
            "tools/library",
            "test/unit",
            "python/CuTeDSL",
        ],
        "component_ownership": {
            "cutlass-cuda-kernels": ["upstream-maintainer-review-required"],
            "cute-and-cutedsl": ["upstream-maintainer-review-required"],
        },
        "required_cells": ["cuda-sm80", "cuda-sm90", "cuda-sm100"],
        "edge_profiles": ["cuda-sm120", "windows", "python-cutedsl"],
        "contracts": {
            "api-abi": _contract(
                "Preserve CUTLASS/CuTe template, layout, datatype, collective, epilogue, and "
                "library operation contracts across the declared architecture family.",
                "template-instantiation",
                "layout-contract",
                "epilogue-contract",
                "library-registration",
            ),
            "lifecycle": _contract(
                "Preserve configure, code generation, compile, link, profiler discovery, first "
                "launch, repeated launch, and stream behavior.",
                "cmake-configure",
                "code-generation",
                "compile-link",
                "repeat-launch",
            ),
            "build-test-matrix": _contract(
                "Run affected architecture unit tests plus examples and profiler verification "
                "for the changed operation, datatype, layout, and epilogue.",
                "unit-architecture",
                "example-build",
                "profiler-verification",
                "compile-negative",
            ),
            "dependency-policy": _contract(
                "Pin CUDA toolkit, host compiler, CMake, Python, submodule, and generated-source "
                "requirements without importing build-only dependencies at selection time.",
                "toolkit-compiler-matrix",
                "submodule-state",
                "clean-build",
                "generated-source-identity",
            ),
            "fallback-policy": _contract(
                "Unsupported architectures, alignments, layouts, datatypes, and schedules must "
                "be rejected or routed to a declared library/vendor fallback.",
                "unsupported-architecture",
                "alignment-boundary",
                "can-implement",
                "vendor-fallback",
            ),
            "deployment-workload-portfolio": _contract(
                "Cover dense and grouped GEMM, tiny-M and skinny shapes, non-aligned K, mixed "
                "epilogues, grouped expert tails, and concurrent streams.",
                "dense-gemm",
                "grouped-gemm",
                "tiny-m",
                "non-aligned-k",
                "concurrent-streams",
            ),
            "performance-acceptance-targets": _contract(
                "Report local-cell latency, throughput, workspace, occupancy evidence, compile "
                "cost, and retention against pinned cuBLASLt and prior CUTLASS baselines.",
                "cutlass-profiler",
                "cublaslt-local-baseline",
                "workspace-peak",
                "compile-cost",
            ),
            "maintainability-probes": _contract(
                "Localize generation-specific code, keep shared CuTe abstractions explicit, and "
                "bind new schedules to focused tests and documentation.",
                "architecture-locality",
                "shared-abstraction",
                "test-affinity",
                "documentation-impact",
            ),
        },
    },
    "liger-kernel": {
        "catalog_profile": "liger-training-fused-kernel-v1",
        "repository": "https://github.com/linkedin/Liger-Kernel",
        "revision": "e6a81bb0c34f31ca7806d0c2b72f6d66b0542694",
        "aliases": ["liger-kernel", "linkedin/liger-kernel", "liger_kernel"],
        "sources": [
            "https://github.com/linkedin/Liger-Kernel/blob/"
            "e6a81bb0c34f31ca7806d0c2b72f6d66b0542694/README.md",
            "https://github.com/linkedin/Liger-Kernel/tree/"
            "e6a81bb0c34f31ca7806d0c2b72f6d66b0542694/test",
            "https://github.com/linkedin/Liger-Kernel/blob/"
            "e6a81bb0c34f31ca7806d0c2b72f6d66b0542694/benchmark/"
            "BENCHMARK_GUIDELINES.md",
        ],
        "integration_points": [
            "src/liger_kernel/ops",
            "src/liger_kernel/chunked_loss",
            "src/liger_kernel/transformers",
            "test",
            "benchmark",
        ],
        "component_ownership": {
            "triton-training-ops": ["upstream-kernel-maintainer-review-required"],
            "framework-integration": ["upstream-maintainer-review-required"],
        },
        "required_cells": ["cuda-sm80", "cuda-sm90"],
        "edge_profiles": ["cuda-sm100", "rocm", "cutedsl"],
        "contracts": {
            "api-abi": _contract(
                "Preserve forward, backward, autograd, reduction, ignore-index, dtype, shape, "
                "and framework replacement semantics for fused training operators.",
                "forward-api",
                "backward-api",
                "autograd-contract",
                "framework-replacement",
            ),
            "lifecycle": _contract(
                "Preserve import, Triton JIT, torch.compile, first-call cache, repeated-call, "
                "autocast, checkpointing, and distributed training behavior.",
                "clean-import",
                "jit-first-call",
                "cache-reuse",
                "torch-compile",
                "checkpointing",
            ),
            "build-test-matrix": _contract(
                "Compare forward and backward results and gradients with PyTorch across boundary "
                "shapes, dtypes, reductions, and supported framework integrations.",
                "reference-forward",
                "reference-backward",
                "gradient-error",
                "dtype-shape-matrix",
                "integration-smoke",
            ),
            "dependency-policy": _contract(
                "Declare PyTorch, Triton, transformers, optional framework, and architecture "
                "requirements while keeping optional integrations capability-gated.",
                "dependency-matrix",
                "optional-import-gate",
                "clean-install",
                "package-boundary",
            ),
            "fallback-policy": _contract(
                "Unsupported shape, dtype, backend, compile mode, or framework versions must use "
                "the declared eager composition or fail explicitly.",
                "unsupported-shape",
                "unsupported-dtype",
                "compile-fallback",
                "eager-fallback",
            ),
            "deployment-workload-portfolio": _contract(
                "Cover RMSNorm, RoPE, SwiGLU, cross entropy, fused linear loss, chunked loss, "
                "long sequences, and SFT/RL loss forward-backward paths.",
                "normalization",
                "activation",
                "cross-entropy",
                "chunked-loss",
                "long-sequence",
            ),
            "performance-acceptance-targets": _contract(
                "Report training-step latency, peak memory, compile cost, cache reuse, and "
                "forward-backward speed against pinned eager and prior Liger baselines.",
                "step-latency",
                "peak-memory",
                "compile-cost",
                "cache-reuse",
            ),
            "maintainability-probes": _contract(
                "Keep math, kernel, autograd wrapper, framework integration, tests, and benchmark "
                "changes synchronized under focused owner review.",
                "layer-parity",
                "test-affinity",
                "benchmark-update",
                "owner-review",
            ),
        },
    },
    "deepgemm": {
        "catalog_profile": "deepgemm-moe-gemm-kernel-v1",
        "repository": "https://github.com/deepseek-ai/DeepGEMM",
        "revision": "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
        "aliases": ["deepgemm", "deepseek-ai/deepgemm", "deep_gemm"],
        "sources": [
            "https://github.com/deepseek-ai/DeepGEMM/blob/"
            "559d79fb6994a58b8a15b4b93bf13ccc16edf247/README.md",
            "https://github.com/deepseek-ai/DeepGEMM/tree/"
            "559d79fb6994a58b8a15b4b93bf13ccc16edf247/deep_gemm",
            "https://github.com/deepseek-ai/DeepGEMM/tree/"
            "559d79fb6994a58b8a15b4b93bf13ccc16edf247/tests",
        ],
        "integration_points": ["deep_gemm", "deep_gemm/jit", "tests", "bench.py"],
        "component_ownership": {
            "jit-kernel-generation": ["upstream-maintainer-review-required"],
            "layout-and-runtime": ["upstream-maintainer-review-required"],
        },
        "required_cells": ["cuda-sm90"],
        "edge_profiles": ["cuda-sm100", "cuda-sm120", "multi-node"],
        "contracts": {
            "api-abi": _contract(
                "Preserve dense/grouped GEMM tensor, scale, layout, alignment, accumulation, "
                "output, expert indexing, and tuning-key semantics.",
                "dense-api",
                "grouped-api",
                "scale-layout",
                "expert-indexing",
            ),
            "lifecycle": _contract(
                "Preserve JIT source generation, compile, cache identity, module load, first "
                "launch, repeated launch, stream behavior, and teardown.",
                "jit-generation",
                "compile-cache",
                "module-load",
                "repeat-launch",
            ),
            "build-test-matrix": _contract(
                "Run correctness and tuning coverage for dense, masked grouped, contiguous "
                "grouped, alignment-boundary, and architecture-specific variants.",
                "dense-correctness",
                "grouped-masked",
                "grouped-contiguous",
                "alignment-boundary",
                "architecture-matrix",
            ),
            "dependency-policy": _contract(
                "Pin toolkit, compiler, CUDA driver/runtime, Python extension, submodule, and JIT "
                "cache requirements as part of artifact identity.",
                "toolchain-identity",
                "submodule-state",
                "clean-build",
                "cache-identity",
            ),
            "fallback-policy": _contract(
                "Unsupported architecture, dtype, alignment, scale layout, or expert shape must "
                "select a declared project/vendor path or fail explicitly.",
                "architecture-gate",
                "layout-gate",
                "shape-gate",
                "declared-fallback",
            ),
            "deployment-workload-portfolio": _contract(
                "Cover FP8/BF16 dense and grouped MoE GEMM, balanced and long-tail experts, "
                "tiny-M, non-aligned K, empty experts, and concurrent streams.",
                "dense-fp8",
                "grouped-moe",
                "long-tail-experts",
                "tiny-m",
                "concurrent-streams",
            ),
            "performance-acceptance-targets": _contract(
                "Report local-cell latency, throughput, workspace, compile cost, cache reuse, and "
                "retention against pinned project/vendor implementations.",
                "latency",
                "throughput",
                "workspace",
                "compile-cost",
                "cache-reuse",
            ),
            "maintainability-probes": _contract(
                "Keep generated source, tuning keys, runtime dispatch, tests, and supported "
                "architecture declarations synchronized.",
                "generator-runtime-parity",
                "tuning-key-contract",
                "test-affinity",
                "architecture-declaration",
            ),
        },
    },
    "megatron-core": {
        "catalog_profile": "megatron-core-training-kernel-host-v1",
        "repository": "https://github.com/NVIDIA/Megatron-LM",
        "revision": "3c04d2bd2255c9652a687c3d5a5b9636467696db",
        "aliases": ["megatron-core", "nvidia/megatron-lm", "megatron.core"],
        "sources": [
            "https://github.com/NVIDIA/Megatron-LM/blob/"
            "3c04d2bd2255c9652a687c3d5a5b9636467696db/README.md",
            "https://github.com/NVIDIA/Megatron-LM/tree/"
            "3c04d2bd2255c9652a687c3d5a5b9636467696db/megatron/core",
            "https://github.com/NVIDIA/Megatron-LM/tree/"
            "3c04d2bd2255c9652a687c3d5a5b9636467696db/tests/unit_tests",
        ],
        "integration_points": [
            "megatron/core",
            "megatron/core/tensor_parallel",
            "megatron/core/transformer",
            "megatron/core/distributed",
            "tests/unit_tests",
        ],
        "component_ownership": {
            "parallel-state-and-distributed": ["upstream-maintainer-review-required"],
            "transformer-and-fusions": ["upstream-maintainer-review-required"],
        },
        "required_cells": ["cuda-sm80", "cuda-sm90"],
        "edge_profiles": ["cuda-sm100", "rocm", "multi-node-rdma"],
        "contracts": {
            "api-abi": _contract(
                "Preserve Megatron-Core module, tensor/sequence/context/expert parallel, "
                "distributed checkpoint, config, and extension interfaces.",
                "module-api",
                "parallel-layout",
                "config-contract",
                "checkpoint-schema",
            ),
            "lifecycle": _contract(
                "Preserve process-group initialization, model construction, forward/backward, "
                "optimizer, checkpoint, restart, and process-group teardown.",
                "parallel-init",
                "train-step",
                "checkpoint-restart",
                "group-destroy",
            ),
            "build-test-matrix": _contract(
                "Run focused unit tests and affected TP/PP/CP/EP distributed lanes with forward, "
                "backward, determinism, checkpoint, and restart checks.",
                "unit-focused",
                "distributed-parallelism",
                "forward-backward",
                "checkpoint-restart",
            ),
            "dependency-policy": _contract(
                "Declare PyTorch, CUDA, Transformer Engine, communication, fused-kernel, and "
                "optional framework dependencies with capability-gated imports.",
                "dependency-matrix",
                "optional-import-gate",
                "clean-install",
                "extension-version",
            ),
            "fallback-policy": _contract(
                "Unavailable fused, low-precision, communication, or architecture paths must use "
                "the declared unfused/native path or fail before distributed divergence.",
                "fused-op-fallback",
                "precision-fallback",
                "rank-consistency",
                "explicit-failure",
            ),
            "deployment-workload-portfolio": _contract(
                "Cover representative TP, PP, CP, EP, distributed optimizer, mixed precision, "
                "checkpointing, microbatch, and long-sequence training cells.",
                "tensor-parallel",
                "pipeline-parallel",
                "context-parallel",
                "expert-parallel",
                "checkpointing",
            ),
            "performance-acceptance-targets": _contract(
                "Report step time, model FLOP utilization, scaling efficiency, communication "
                "overlap, memory, compile/startup cost, and checkpoint overhead.",
                "step-time",
                "mfu",
                "scaling-efficiency",
                "communication-overlap",
                "peak-memory",
            ),
            "maintainability-probes": _contract(
                "Keep parallel-state ownership, configuration, kernel integration, distributed "
                "tests, and checkpoint compatibility explicit under owner review.",
                "canonical-owner",
                "config-locality",
                "distributed-test-affinity",
                "checkpoint-compatibility",
            ),
        },
    },
    "torchtitan": {
        "catalog_profile": "torchtitan-pytorch-native-training-host-v1",
        "repository": "https://github.com/pytorch/torchtitan",
        "revision": "496b11d43860bb8d27b54568c76db6310ae7f55e",
        "aliases": ["torchtitan", "pytorch/torchtitan", "torch-titan"],
        "sources": [
            "https://github.com/pytorch/torchtitan/blob/"
            "496b11d43860bb8d27b54568c76db6310ae7f55e/README.md",
            "https://github.com/pytorch/torchtitan/tree/"
            "496b11d43860bb8d27b54568c76db6310ae7f55e/docs",
            "https://github.com/pytorch/torchtitan/tree/"
            "496b11d43860bb8d27b54568c76db6310ae7f55e/tests",
        ],
        "integration_points": ["torchtitan", "torchtitan/parallelisms", "tests", "docs"],
        "component_ownership": {
            "training-runtime": ["upstream-maintainer-review-required"],
            "parallelism-and-kernels": ["upstream-maintainer-review-required"],
        },
        "required_cells": ["cuda-sm80", "cuda-sm90"],
        "edge_profiles": ["cuda-sm100", "rocm", "multi-node"],
        "contracts": {
            "api-abi": _contract(
                "Preserve job configuration, model protocol, parallelism, optimizer, checkpoint, "
                "dataset, metric, and extension interfaces.",
                "job-config",
                "model-protocol",
                "parallelism-api",
                "checkpoint-api",
            ),
            "lifecycle": _contract(
                "Preserve launch, device mesh, model initialization, parallelization, compile, "
                "forward/backward, optimizer, checkpoint/restart, and shutdown.",
                "launch",
                "device-mesh",
                "compile-train-step",
                "checkpoint-restart",
                "shutdown",
            ),
            "build-test-matrix": _contract(
                "Run lint, focused unit tests, single-GPU and distributed FSDP2/TP/PP lanes, "
                "torch.compile, checkpoint, and determinism checks.",
                "lint",
                "unit-focused",
                "fsdp2",
                "torch-compile",
                "checkpoint-restart",
            ),
            "dependency-policy": _contract(
                "Use declared PyTorch-native APIs and capability-gate optional kernels, models, "
                "datasets, profilers, and distributed dependencies.",
                "pytorch-version",
                "optional-import-gate",
                "clean-install",
                "dependency-diff",
            ),
            "fallback-policy": _contract(
                "Unsupported compile, fused-kernel, precision, or parallelism paths must select a "
                "declared PyTorch-native fallback or fail consistently on every rank.",
                "compile-fallback",
                "kernel-fallback",
                "precision-fallback",
                "rank-consistency",
            ),
            "deployment-workload-portfolio": _contract(
                "Cover single-GPU, FSDP2, TP, PP, long sequence, mixed precision, activation "
                "checkpointing, compile, and checkpoint/restart training cells.",
                "single-gpu",
                "fsdp2",
                "tensor-parallel",
                "pipeline-parallel",
                "long-sequence",
            ),
            "performance-acceptance-targets": _contract(
                "Report step latency, throughput, memory, compile time, scaling efficiency, and "
                "checkpoint overhead against a pinned local TorchTitan baseline.",
                "step-latency",
                "throughput",
                "peak-memory",
                "compile-time",
                "scaling-efficiency",
            ),
            "maintainability-probes": _contract(
                "Prefer composable PyTorch-native changes, localize model-specific behavior, and "
                "bind parallelism or kernel changes to focused tests and docs.",
                "pytorch-native-surface",
                "model-locality",
                "test-affinity",
                "docs-impact",
            ),
        },
    },
    "verl": {
        "catalog_profile": "verl-posttraining-rollout-host-v1",
        "repository": "https://github.com/volcengine/verl",
        "revision": "c2429f29a25d573f63d9bcc29e7ceb690817dce9",
        "aliases": ["verl", "volcengine/verl", "verl-trainer"],
        "sources": [
            "https://github.com/volcengine/verl/blob/"
            "c2429f29a25d573f63d9bcc29e7ceb690817dce9/README.md",
            "https://github.com/volcengine/verl/tree/c2429f29a25d573f63d9bcc29e7ceb690817dce9/docs",
            "https://github.com/volcengine/verl/tree/"
            "c2429f29a25d573f63d9bcc29e7ceb690817dce9/tests",
        ],
        "integration_points": [
            "verl",
            "verl/trainer",
            "verl/workers",
            "verl/workers/rollout",
            "tests",
        ],
        "component_ownership": {
            "trainer-and-worker-protocol": ["upstream-maintainer-review-required"],
            "rollout-and-inference-adapter": ["upstream-maintainer-review-required"],
        },
        "required_cells": ["cuda-sm80", "cuda-sm90"],
        "edge_profiles": ["cuda-sm100", "rocm", "vllm-rollout", "sglang-rollout"],
        "contracts": {
            "api-abi": _contract(
                "Preserve trainer, worker, rollout, reward, data, configuration, checkpoint, and "
                "distributed protocol semantics across supported backends.",
                "trainer-config",
                "worker-protocol",
                "rollout-interface",
                "checkpoint-schema",
            ),
            "lifecycle": _contract(
                "Preserve cluster startup, resource placement, worker initialization, rollout, "
                "training update, sleep/wake, checkpoint/restart, and shutdown.",
                "cluster-start",
                "worker-init",
                "rollout-update-cycle",
                "sleep-wake",
                "checkpoint-restart",
            ),
            "build-test-matrix": _contract(
                "Run focused unit and integration tests for SFT and RL algorithms, rollout "
                "backends, distributed workers, fault paths, checkpointing, and supported models.",
                "unit-focused",
                "sft-smoke",
                "rl-smoke",
                "rollout-backend-matrix",
                "fault-recovery",
            ),
            "dependency-policy": _contract(
                "Capability-gate training, rollout, serving, distributed, data, and optional model "
                "dependencies and keep backend versions explicit.",
                "optional-import-gate",
                "backend-version",
                "clean-install",
                "dependency-diff",
            ),
            "fallback-policy": _contract(
                "Unavailable rollout engines, kernels, sleep/wake capabilities, or distributed "
                "features must use a declared backend or fail before partial worker progress.",
                "rollout-backend-fallback",
                "capability-gate",
                "worker-consistency",
                "explicit-failure",
            ),
            "deployment-workload-portfolio": _contract(
                "Cover SFT, GRPO/DAPO-style RL, rollout-training cycles, variable sequence length, "
                "multi-turn generation, sleep/wake, and checkpoint/restart.",
                "sft",
                "rl-update",
                "rollout-training-cycle",
                "variable-sequence",
                "sleep-wake",
            ),
            "performance-acceptance-targets": _contract(
                "Report end-to-end step and rollout throughput, generation latency, GPU memory, "
                "worker utilization, restart overhead, and kernel-local gains separately.",
                "step-throughput",
                "rollout-throughput",
                "generation-latency",
                "peak-memory",
                "worker-utilization",
            ),
            "maintainability-probes": _contract(
                "Keep policy, worker ownership, rollout adapter, backend capability, tests, and "
                "configuration migrations explicit and synchronized.",
                "policy-versioning",
                "canonical-owner",
                "adapter-capability",
                "test-affinity",
                "config-migration",
            ),
        },
    },
}

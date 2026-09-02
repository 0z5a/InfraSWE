# InfraSWE

InfraSWE is an executable benchmark harness for AI infrastructure agents. A run is not
considered complete when an agent merely emits a patch: the patch is collected as a declared
artifact, the agent environment is destroyed, and the artifact is applied in fresh verifier
environments where functional, SLO, fault-recovery, policy, and evidence checks run.

This repository implements the v0.1 protocol core and the complete 16-package release matrix.

## Implemented task packages

| Task | Track | Execution scope |
|---|---|---|
| `cuda-artifact-capability-selection` | Build & packaging | 1×SM80 capability/ABI selection with real CUDA evidence |
| `container-image-dependency-lock` | Build & packaging | Hermetic CPU image/dependency lock resolution |
| `gpu-resource-request-contract` | Deploy & configuration | Hermetic CPU GPU admission contract |
| `gpu-service-rollout-regression` | Deploy & configuration | Hermetic CPU control-plane model |
| `health-probe-drain-regression` | Deploy & configuration | Hermetic CPU readiness/drain rollout contract |
| `telemetry-root-cause-correlation` | Observability | Hermetic CPU multi-signal root-cause correlation |
| `cuda-extension-arch-target` | Build & packaging | 1×SM80 native extension architecture targeting |
| `dynamic-batching-slo-collapse` | Inference performance | 1×SM80 deadline-aware batching with real CUDA probes |
| `numa-cpu-affinity-regression` | Inference performance | 1×SM80 PCI/NUMA CPU affinity with a pinned CUDA probe |
| `gpu-oom-worker-recovery` | Reliability | 1×SM80 real CUDA OOM injection and worker recovery |
| `nccl-topology-silent-fallback-2gpu` | Distributed communication | Experimental 2×SM80 task with a real NCCL collective |
| `kv-aware-routing-cache-collapse-2gpu` | Inference performance | Experimental 2×SM80 task with real CUDA probes and a deterministic routing workload |
| `tensor-parallel-shard-contract` | Inference performance | Experimental 2×SM120 NCCL shard reconstruction |
| `collective-order-rank-divergence` | Distributed communication | Experimental 2×SM120 canonical collective scheduling and NCCL execution |
| `collective-compute-overlap-regression` | Distributed communication | Experimental 2×SM120 serial/async-stream NCCL comparison |
| `rank-exit-collective-recovery` | Reliability | Experimental 2×SM120 real rank-exit injection and group rebuild |

Every package has a noop base case, a trusted solution, hidden behavioral checks, fault evidence,
and three fresh verifier replays. GPU packages require a host satisfying their declared hardware
profile and pinned agent/verifier images.

The rollout task remains a hermetic control-plane model rather than live k3s, the original NCCL
task uses the available 2×A100 profile rather than the design draft's formal 4×A100 topology, and
the KV-routing task uses a deterministic semantic workload rather than a deployed inference
service. Production agent adapters and provider-backed lease lifecycle certification remain
outside this checkpoint.

The kernel-frontier evaluator also includes an experimental one-device AMD profile for
MI300X (`gfx942`) with PyTorch 2.4.0 / ROCm 6.1. Its initial AOTriton and portable Triton
adapter is documented in
[`benchmarks/kernel_frontier/MI300X_ROCM61.md`](benchmarks/kernel_frontier/MI300X_ROCM61.md);
it remains outside the default leaderboard until real-hardware evidence is collected.

A second experimental adapter registers NVIDIA B200 (`sm100`, compute capability 10.0)
with a CUDA 13.3 / PTX 9.3 compiler contract. It separates generic, family-specific,
and architecture-specific targets and fail-closes TMEM/TCGen05, Cluster Launch Control,
and Blackwell TMA evidence. See
[`benchmarks/kernel_frontier/BLACKWELL_B200.md`](benchmarks/kernel_frontier/BLACKWELL_B200.md).
Its initial release reports native evidence coverage only; performance scores remain N/A.

## Quick start

Python 3.12 is required.

```bash
uv sync --extra dev
uv run infraswe task validate tasks/gpu-service-rollout-regression
uv run infraswe task certify tasks/gpu-service-rollout-regression --executor docker
uv run infraswe report runs/<run-id>
```

The `oracle` baseline should pass three fresh replays. The `noop` baseline should fail the target
assertions and is useful for checking that the task is not vacuous.

## Security boundary

The sample runner uses separate temporary workspaces for the agent and every verifier replay. The
hidden `tests/` and `solution/` directories are never copied into the agent workspace. For release
certification, use `--executor docker`; Docker runs commands with no network, dropped Linux
capabilities, and `no-new-privileges`.

The local executor exists for SDK development and tests. It is logical isolation, not a production
sandbox.

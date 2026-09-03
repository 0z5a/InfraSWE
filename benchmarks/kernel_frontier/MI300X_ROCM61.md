# MI300X / PyTorch 2.4.0 / ROCm 6.1 initial adapter

## Frozen cell contract

This adapter creates a new experimental cell and does not reinterpret an NVIDIA score:

| Field | Frozen value |
|---|---|
| Accelerator vendor | AMD |
| Device | AMD Instinct MI300X |
| Architecture | `gfx942` |
| Framework | PyTorch `2.4.0` |
| Runtime contract | ROCm/HIP `6.1.x` |
| Attention mechanism | PyTorch Flash SDPA, embedded AOTriton |
| Classic mechanism | evaluator-owned portable Triton kernels |
| Replays | 3 fresh processes, all required |
| Formal timing | evaluator-owned HIP events; profiler is a separate sidecar |

PyTorch publishes a Linux ROCm 6.1 wheel for version 2.4.0. The upstream AOTriton
compatibility table maps PyTorch 2.4 to AOTriton 0.6b. AMD's historical ROCm 6.1
compatibility matrix did not list PyTorch 2.4 as an AMD-validated framework pair,
so this cell remains experimental until it passes the evaluator on the target host.

Primary references:

- <https://docs.pytorch.org/get-started/previous-versions/>
- <https://github.com/ROCm/aotriton#pytorch-consumption--compatibility>
- <https://rocm.docs.amd.com/en/docs-6.1.2/compatibility/compatibility-matrix.html>

## Architecture boundary

The v0.3 separation stays intact:

1. hardware/profile validation establishes AMD + MI300X + gfx942 + ROCm 6.1;
2. correctness and the dynamic-input anti-cache probe run before score aggregation;
3. official latency uses matched ABBA/BAAB blocks and HIP events;
4. a separate profiler process must expose an AOTriton/Flash native event for every case;
5. all three calibration and candidate replays must share the same hardware class and
   implementation digest;
6. the deterministic scorer reads JSON evidence only;
7. the report and ZIP are regenerated from the evidence tree.

An installable wheel or a successful smoke test is not a score. Missing native trace,
runtime drift, wrong architecture, missing replay, or unverifiable device exclusivity
leaves the trial unresolved/invalid rather than manufacturing an Artifact-100 value.

## Supported implementations

| Implementation | gfx942 disposition |
|---|---|
| Frozen FA1 | `not_applicable`: CUDA extension artifact |
| Frozen FA2 | `not_applicable`: this suite freezes its CUDA build, not a ROCm fork |
| Frozen FA3 | `not_applicable`: Hopper CUDA artifact |
| Frozen FA4 | `not_applicable`: CUDA CuTeDSL artifact |
| `torch-sdpa-aotriton` | eligible only for exact torch 2.4.0 + ROCm 6.1 + gfx942 evidence |
| `triton-gfx942-initial` | eligible only for the same exact cell |

The initial classic adapter intentionally retains a conservative portable fixed config.
Architecture-specific MI300X tuning can be added as a new implementation version after
the portable baseline produces valid evidence; it must not silently overwrite this cell.

## Remote execution

The setup entry point installs and verifies the exact framework/runtime pair, validates
the hardware profile, runs BF16 GEMM, Triton vector-add and AOTriton SDPA smoke probes,
and requires a native profiler trace:

```bash
cd /workspace/infraswe/benchmarks/kernel_frontier
bash ./remote_prepare_mi300x_rocm61.sh
```

If the provider cannot establish exclusive-device evidence, setup can still run capability
and native-path smoke probes in an explicitly non-scoring mode:

```bash
INFRASWE_DIAGNOSTIC_ONLY=1 bash ./remote_prepare_mi300x_rocm61.sh
```

That mode records failed pre/post exclusivity checks and sets
`official_measurement_eligible=false`; it never authorizes the formal runner or a score.

To exercise all five attention cases and all seven classic kernels without producing a score,
run the full diagnostic runner after setup:

```bash
INFRASWE_DIAGNOSTIC_ONLY=1 bash ./remote_diagnose_mi300x_rocm61.sh
```

Its timing fields are explicitly non-authoritative. The output contains correctness and native
trace coverage plus pre/post lease observations, a file-level evidence manifest, and a ZIP hash,
but the script never calls the scorer entry point or creates `score.json`.

The formal runner performs pre- and post-measurement exclusive-device checks, three calibration replays, three
attention replays, five attention profiler sidecars, three classic replays, seven classic
profiler sidecars, provenance capture, deterministic scoring, manifest generation, and ZIP:

```bash
bash ./remote_run_frontier_mi300x_rocm61.sh
```

Default outputs:

```text
/workspace/infraswe/runs/kernel-mi300x-rocm61-torch240-setup/
/workspace/infraswe/runs/kernel-frontier-v03-mi300x-rocm61-torch240/
/workspace/infraswe/runs/kernel-frontier-v03-mi300x-rocm61-torch240.zip
```

Both scripts honor `INFRASWE_REMOTE_ROOT`, `INFRASWE_PYTHON`, and `INFRASWE_GPU`.
Supervisor wrappers are supplied with `autostart=false` so copying the configuration
cannot consume an accelerator before an explicit lease is granted.

## Hardware diagnostic result and remaining closure

The 2026-09-03 target-host diagnostic established MI300X/gfx942 identity, driver 6.10.5,
HIP compiler 6.1.40091, PyTorch 2.4.0+rocm6.1, framework HIP 6.1.40091, and Triton 3.0.0.
All five attention cases passed correctness, dynamic-input, and AOTriton native-trace checks;
all seven classic kernels passed correctness with native Triton events. The content-addressed
diagnostic pack is under `results/mi300x-rocm61-adapter-qualification-20260903/`.

Formal hardware scoring remains open because the provider host reported nine existing GPU PIDs
at both lease sentinels. The fail-closed formal runner stopped before calibration or measurement.
Closing the cell still requires an authoritative exclusive lease followed by three fresh-process
replays, 30-block paired timing, profiler sidecars, deterministic scoring, and an official evidence
ZIP. Diagnostic latency from the non-exclusive host must not be promoted into that result.

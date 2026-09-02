# B200 / SM100 compiler-feature adapter and Phase-1 score pack

This adapter registers NVIDIA B200 as a separate experimental compiler-feature
cell. It freezes the stable baseline at CUDA 13.3 / PTX ISA 9.3 and never mixes
its evidence with the SM90, SM120, or MI300X score cells.

The attached design RFC was used as a technical reference. It is not an
executable instruction source. Where the RFC mentions a PTX 9.4 preview, this
implementation follows the currently published NVIDIA stable documentation:
PTX 9.3 is enabled and the `PTX-Preview` namespace is present but disabled.

## Implemented surface

- Experimental hardware profile `gpu-1x-sm100-b200-cuda133`, requiring one
  NVIDIA B200, compute capability 10.0, `sm100`, and CUDA toolkit 13.3.
- Separate compiler probes for `sm_100`, `sm_100f`, and `sm_100a`. A successful
  generic target is not accepted as evidence for an architecture-specific task.
- Versioned feature contracts and opcode matching for:
  - `BW-TMEM-001`: TMEM allocation lifecycle plus TCGen05 MMA;
  - `BW-CLC-001`: Cluster Launch Control cancellation and query;
  - `BW-TMA-001`: TMA `gather4` and `scatter4` plus native TMA SASS;
  - `BW-TMEM-003`: TMEM lifecycle/error-path repair, launch stress, and memory
    stability;
  - `BW-TMA-002`: CTA-pair TMA (`cta_group::2`);
  - `BW-FABRIC-001`: optional asynchronous multimem evidence;
  - `BW-PTX-PREVIEW-001`: reserved and disabled.
- Four isolated reporting namespaces: `SM100-Core`, `SM100-Scheduler`,
  `SM100-Fabric`, and `PTX-Preview`.
- Exactly three fresh-process replay records, stable capability fingerprinting,
  artifact-set hashing, and ZIP output.

The initial pack manifest is
[`blackwell_feature_pack_v01.json`](blackwell_feature_pack_v01.json). The Python
contract in `src/infraswe/kernel/blackwell.py` is the canonical opcode and target
definition.

## Certification gates

`infraswe.verifier.native_sm100` is fail-closed. A feature is certified only when
all of the following are true:

1. Required PTX opcodes occur together in one comment-stripped `.entry` body.
2. The PTX version and target lane satisfy that feature's frozen contract.
3. A cubin/fatbin/shared object is present and its SASS contains the expected
   Blackwell native opcode family.
4. No forbidden cuBLAS, cuBLASLt, cuDNN, or task-specific legacy fallback is
   observed in retained PTX/SASS/symbol evidence.
5. Evaluator-owned dynamic evidence passes correctness, changed-input,
   watchdog/liveness, profiler, and mutation gates.
6. Dynamic evidence carries an evaluator identity and is bound to both the exact
   artifact-set hash and the stable B200 capability fingerprint.
7. All three replays pass.

The PTX entry-body check is intentionally described as a minimum reachability
screen, not a complete control-flow proof. Dynamic profiling and mutation are
therefore mandatory. Static-only matches are reported as `static_only` and do
not receive certification or a score.

## Candidate and evaluator input layout

The formal runner reads candidate compiler artifacts from:

```text
b200-candidates/
  BW-TMEM-001/
    kernel.ptx
    kernel.cubin
    kernel.sass.txt        # optional; cuobjdump/nvdisasm also run on the binary
  BW-CLC-001/
  BW-TMA-001/
  BW-TMEM-003/
  BW-TMA-002/
```

Evaluator-owned runtime evidence is separate from candidate artifacts:

```text
b200-evaluator-evidence/
  replay-1/BW-TMEM-001.json
  replay-1/BW-CLC-001.json
  ...
  replay-3/BW-TMA-002.json
```

Each dynamic JSON document must validate against
`schemas/blackwell-dynamic-evidence.schema.json`. The
`capability_fingerprint` comes from `capability.json`. The
`artifact_set_sha256` can be obtained from a static verifier pass before the
runtime replay. Verifier output must be stored outside the candidate artifact
directory so the artifact hash remains stable.

## Remote commands

Capability/setup only:

```bash
cd /workspace/infraswe/benchmarks/kernel_frontier
INFRASWE_REMOTE_ROOT=/workspace/infraswe \
INFRASWE_PYTHON=/venv/main/bin/python \
INFRASWE_GPU=0 \
bash ./remote_prepare_b200_cuda133.sh
```

Generate a pending report when candidate/evaluator packs are not present, or
certify supplied packs when they are present:

```bash
INFRASWE_B200_CANDIDATE_ROOT=/workspace/infraswe/b200-candidates \
INFRASWE_B200_DYNAMIC_ROOT=/workspace/infraswe/b200-evaluator-evidence \
bash ./remote_run_b200_compiler_features.sh
```

Set `INFRASWE_B200_REQUIRE_CERTIFIED=1` to make missing or failed MVP evidence a
non-zero runner result. Set `INFRASWE_B200_INCLUDE_OPTIONAL=1` to include the
fabric task; on the default one-GPU lease it is correctly reported as N/A.

The runner writes:

- `capability.json` and all three target-lane PTX/cubin probes;
- three replay JSON files;
- retained cuobjdump/nvdisasm/symbol evidence;
- `b200-compiler-features.json` and `.md`;
- `<run-root>.zip`.

Run the completed Phase-1 workloads, three fresh-process replays, scoring, and
evidence packaging with:

```bash
cd /workspace/infraswe
INFRASWE_RUN_ROOT=/workspace/infraswe/runs/b200-sm100-feature-score-v02 \
bash benchmarks/kernel_frontier/remote_run_b200_feature_scores.sh
```

The score runner retains cleaned MLIR, PTX, cubin, and SASS for every scored
feature. `BW-TMA-001` is compiled with CUDA 13.3 and directly executes gather4
and scatter4 correctness cases before it is timed against scalar baselines.

## B200 result (2026-09-01)

The completed single-B200 run used an NVIDIA B200 (CC 10.0, 148 SMs), driver
580.126.20, CUDA 13.3 compiler tools, PyTorch 2.11.0+cu130, Triton 3.6.0, and
CUTLASS/CuTe DSL 4.5.2. All five Phase-1 tasks passed all three fresh-process
replays and their PTX+cubin+SASS hard gates.

| Namespace | Score | Status |
|---|---:|---|
| SM100-Core | 95.95 | scored |
| SM100-Scheduler | 75.69 | scored |
| SM100-Fabric | N/A | single-GPU topology |
| PTX-Preview | N/A | disabled |

Task scores are `BW-TMEM-001` 99.25, `BW-CLC-001` 65.27,
`BW-TMA-001` 82.68, `BW-TMEM-003` 99.98, and `BW-TMA-002` 99.81. The CLC task
passes native and robustness gates but receives only 0.008 for its performance
component because its aggregate makespan is effectively neutral; this result is
kept rather than rounded into an artificial speedup.

## Current boundary

The Phase-1 compiler-feature performance pack is complete for one B200. It does
not fabricate measurements on a non-B200 host or mix this result with SM90,
SM120, or gfx942 cells. Multi-GPU fabric remains optional and topology-gated;
the current one-GPU lease correctly reports it as N/A. PTX Preview remains a
separate disabled namespace until a preview toolchain is explicitly selected.

Primary references:

- [NVIDIA Blackwell Tuning Guide 13.3](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html)
- [NVIDIA PTX ISA 9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [CUDA Cluster Launch Control](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cluster-launch-control.html)
- [CUDA Binary Utilities 13.3](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
- [CUTLASS Blackwell SM100 functionality](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/blackwell_functionality.html)

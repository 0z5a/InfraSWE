# NVIDIA L40S / L20 Ada SM89 platform adapter

This initial adapter implements one shared Ada `sm_89` architecture core and two board-local
cells: `l40s-48gb-pcie` and `l20-48gb-pcie`. Product names identify a cell, but never select a
separate kernel source path. Codegen cache identity is shared; board autotune fingerprints are
isolated.

The compile gate requires both a native `sm_89` cubin and a `compute_89` PTX fallback. The native
verifier requires task-reachable PTX, disassembled SASS where the task declares a native opcode,
runtime/capability binding, seven fresh-process replays, and zero silent fallback. It rejects TMA,
WGMMA, CTA clusters, TMEM, TCGen05, multimem/fabric operations, and native FP4 claims.
CUDA 11.8 can satisfy the base Ada target gate; task-level gates remain separate. In particular,
the packed SM89 FP8 conversion path requires PTX 8.1+, while this initial `m16n8k32` FP8 MMA
smoke requires PTX 8.7+. Older toolchains therefore report those features as unresolved rather
than silently substituting an FP16 path.

Run the capability probe from the repository root:

```bash
PYTHONPATH=src python benchmarks/ada_sm89/capability_probe.py \
  --artifact-root evidence/ada-sm89/capability-artifacts \
  --runtime-probe-source platforms/nvidia-ada-sm89/capability_probe/cuda_probe.cc \
  --output evidence/ada-sm89/capability.json \
  --require-platform --require-toolchain
```

Run the executable minimum suite after a passing platform and compile gate:

```bash
PYTHONPATH=src python benchmarks/ada_sm89/run_minimum_suite.py \
  --capability evidence/ada-sm89/capability.json \
  --platform-root platforms/nvidia-ada-sm89 \
  --output-root evidence/ada-sm89/minimum-suite
```

Architecture-RFC weights are emitted only as a local diagnostic scorecard. Global
`Deployability-100` remains scoring RFC v0.4's frozen `C/U/M` geometric formula; absent L40S,
L20, E2, or E3 evidence is `unresolved`, never a numeric zero.

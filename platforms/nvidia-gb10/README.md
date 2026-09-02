# NVIDIA GB10 / SM121 platform adapter

This adapter treats DGX Spark as an AArch64 unified-memory SoC, not as a smaller B200.
Its fail-closed platform gate requires an AArch64 host, a GB10 device reporting compute
capability 12.1, CUDA 13.0 or newer, and compiler acceptance for `sm_121`, `sm_121f`, and
`sm_121a`. CUDA 13.3 / PTX 9.3 remains the canonical feature-evidence cell; a CUDA 13.0
cell can certify the P0 build/dispatch gate but must report PTX 9.3-only low-precision
features as unresolved.

The adapter explicitly forbids TMEM and `tcgen05.*`, does not assume TMA `scatter4` or
`.cta_group::2`, and separates optional RoCE scale-out from the single-node result.

Run the platform probe from the repository root:

```bash
PYTHONPATH=src python benchmarks/gb10_sm121/capability_probe.py \
  --artifact-root evidence/gb10/capability-artifacts \
  --runtime-probe-source platforms/nvidia-gb10/capability_probe/cuda_probe.cc \
  --output evidence/gb10/capability.json \
  --require-platform --require-toolchain
```

The generated JSON conforms to `schemas/gb10-capability.schema.json`. Per-task native
evidence must be added before the native-feature, correctness/liveness, or performance
gates can pass.

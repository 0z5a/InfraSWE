# Make CUDA artifact selection capability- and ABI-aware

The package launcher currently chooses the first configured artifact. That can select SASS for the
wrong SM, load a binary built for an incompatible CUDA runtime or C++ ABI, or silently claim a CPU
fallback when no CUDA artifact is safe.

Repair `selector.py` and `build_policy.json`. Preserve the
`select_artifact(request, artifacts, policy) -> dict` API and implement these behaviors:

- prefer native SASS only when the device SM, CUDA driver/runtime, and C++11 ABI are compatible;
- use PTX JIT only when its virtual compute target and minimum driver capability are compatible;
- report PTX JIT as an explicit fallback;
- fail closed with an explicit blocked plan when no CUDA artifact is safe;
- never silently substitute a CPU artifact;
- reject malformed requests and produce deterministic results independent of artifact list order;
- do not mutate request, artifact, or policy inputs.

CUDA versions are encoded as `major * 10 + minor` (for example, CUDA 12.6 is `126`). Device SM is
encoded the same way (SM 8.0 is `80`). Native artifacts use `kind="sass"`, `sms`, `built_cuda`, and
`cxx11_abi`; PTX artifacts use `kind="ptx"`, `compute`, `requires_driver_cuda`, and `cxx11_abi`.
Every artifact has a non-empty string `id`.

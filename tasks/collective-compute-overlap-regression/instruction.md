# Restore collective/compute overlap

The current plan launches compute and all-reduce work serially on the default stream. Correctness
still passes, but pipeline latency regresses. Repair `overlap_policy.py` and `overlap_config.json`,
preserving `build_overlap_plan(stages, topology, config) -> dict`.

On a two-GPU topology that supports concurrent kernels, use a dedicated communication stream,
asynchronous collectives, explicit CUDA event fencing, and overlap each collective with the next
compute stage. Emit stages in sequence order and keep output deterministic. Block unsupported
topologies and reject unsafe or malformed configuration rather than silently serializing. Inputs
must remain immutable and collective results must remain exact.

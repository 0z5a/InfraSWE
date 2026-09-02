# Restore GPU-local CPU affinity

The worker pins every GPU process to the lowest-numbered CPUs, ignoring the GPU's PCIe NUMA node
and reserved host cores. Repair `affinity_policy.py` and `affinity_config.json`, preserving
`select_affinity(gpu, topology, config) -> dict`.

Select the configured number of sorted CPUs from the GPU-local NUMA node, intersected with allowed
CPUs and excluding reserved CPUs. If the local node cannot satisfy the request, block explicitly;
cross-NUMA fallback is forbidden. Results must be deterministic, malformed topology must fail
closed, and inputs must not be mutated.

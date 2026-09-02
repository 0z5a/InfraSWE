# Enforce the GPU workload admission contract

The deployment declares a GPU limit but omits the matching request, NVIDIA runtime class, and
compute-capability selector. The current admission helper accepts it and silently assigns a CPU
path when GPU placement fails.

Repair `admission.py` and `workload.json`, preserving
`admit_workload(spec, node, policy) -> dict`. The fixed workload must preserve its name, image,
port, and replica count while requiring one GPU. Admission must require equal positive GPU requests
and limits, an available NVIDIA runtime class, a matching compute-capability selector, and adequate
node capacity. Incompatible workloads must be rejected explicitly; CPU fallback is forbidden.
Malformed inputs must fail conservatively and inputs must not be mutated.

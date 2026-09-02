# Security

InfraSWE tasks must run only in disposable containers, VMs, or namespaces. Do not point the runner
at production infrastructure. Provider credentials belong to the lease broker and must never be
mounted into an agent environment.

Report suspected sandbox escapes, credential exposure, verifier leakage, or unbounded resource
cleanup privately to the repository maintainers. Include the task ID, image digest, run ID, and the
smallest safe reproduction; do not attach live credentials.

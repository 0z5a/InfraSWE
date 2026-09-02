# Repair readiness and termination draining

The service uses its liveness endpoint as readiness, allows an old replica to disappear during
rollout, and terminates before in-flight requests drain. Repair `probe_policy.py` and
`deployment.json` while preserving `build_rollout_plan(deployment, signal) -> dict` and the public
replica count, image, port, and response semantics.

The fixed plan must use `/readyz` for readiness, `/drainz` for drain observation, preserve capacity
during rollout, wait for the supplied maximum in-flight duration before termination, leave enough
termination grace, and roll back on sustained readiness failure. Malformed configuration must fail
closed and inputs must remain unchanged.

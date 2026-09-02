# Repair the rollout regression

The inference service itself is healthy, but a rolling deployment causes a burst of failed
requests and occasionally terminates an old replica before in-flight requests drain. Repair the
deployment configuration while preserving the public port, replica count, image, and response
semantics.

The finished rollout must:

- route traffic only to ready replicas;
- stay within a zero-corruption error budget during rollout;
- survive an injected termination and complete rollback safely;
- leave no orphan replicas or requests;
- retain enough configuration and event evidence to diagnose the rollout.

Do not modify tests or add a bypass around readiness checks.


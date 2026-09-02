# Recover one worker after a CUDA OOM

The current recovery policy restarts every worker and retries the original batch unchanged. One
CUDA OOM therefore causes collateral disruption and often repeats immediately. Repair `recovery.py`
and `recovery_config.json`, preserving
`plan_recovery(event, state, config) -> dict`.

For a valid `cuda_oom` event, retry only the failed request IDs with a batch size reduced by half
(bounded by the configured minimum), clear and restart only the failed worker, and leave healthy
workers running. Once that worker reaches the configured OOM quarantine threshold, quarantine it
and retry the same failed requests on the lexicographically first healthy worker instead. Every OOM
recovery must be reported explicitly; malformed state, configuration, events, or an absence of a
healthy failover worker must fail closed. The result must be deterministic and inputs must not be
mutated.

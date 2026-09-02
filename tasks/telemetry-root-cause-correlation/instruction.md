# Correlate telemetry before declaring a root cause

The diagnoser currently promotes the first log message to a certain root cause, even when metrics,
traces, and profiles contradict it. Repair `diagnoser.py` and `signal_policy.json` while preserving
`diagnose(signals, policy) -> dict`.

Correlate evidence by `correlation_id`, proposed `cause`, and bounded `at_ms` timestamps. A diagnosis
requires at least three distinct modalities among logs, metrics, traces, and profiles. Return stable,
sorted evidence IDs and calibrated confidence. If signals are missing, skewed, malformed, or do not
reach the evidence threshold, return an explicit inconclusive result instead of guessing. Results
must be deterministic under signal reordering and inputs must not be mutated.

# Recover a collective group after one rank exits

The current policy restarts only the exited rank and resumes from the in-flight step. The surviving
rank keeps a poisoned NCCL communicator, so the next collective hangs or diverges. Repair
`failure_policy.py` and `recovery_config.json`, preserving
`plan_rank_failure(event, state, config) -> dict`.

On the first rank exit, abort every rank in the old group, restart the full fixed-size world,
reinitialize the process group, resume from the last committed checkpoint, and replay every pending
request deterministically. When the restart budget is exhausted, emit an explicit abort plan. Never
shrink the world or silently continue. Reject malformed or inconsistent event/state/config inputs
and do not mutate them.

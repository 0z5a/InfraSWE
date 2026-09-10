"""Shared release gates for historical PR classification campaigns."""

from __future__ import annotations

import math

EXACT_ACCURACY_MINIMUM = 0.95
MERGED_ACCEPT_RECALL_MINIMUM = 0.99


def minimum_successes(total: int, minimum_ratio: float) -> int:
    """Return the integer success count required to meet a ratio floor."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be between zero and one")
    return math.ceil(total * minimum_ratio)


def exact_accuracy_gate_satisfied(*, exact_matches: int, eligible_cases: int) -> bool:
    """Require evidence and at least 95% exact three-class accuracy."""

    return eligible_cases > 0 and exact_matches >= minimum_successes(
        eligible_cases, EXACT_ACCURACY_MINIMUM
    )


def merged_accept_recall_gate_satisfied(*, merged_accepts: int, merged_cases: int) -> bool:
    """Require Accept examples and at least 99% recall on them."""

    return merged_cases > 0 and merged_accepts >= minimum_successes(
        merged_cases, MERGED_ACCEPT_RECALL_MINIMUM
    )


def release_quality_gate_satisfied(
    *,
    exact_matches: int,
    eligible_cases: int,
    merged_accepts: int,
    merged_cases: int,
) -> bool:
    """Require both release gates; neither metric can compensate for the other."""

    return exact_accuracy_gate_satisfied(
        exact_matches=exact_matches,
        eligible_cases=eligible_cases,
    ) and merged_accept_recall_gate_satisfied(
        merged_accepts=merged_accepts,
        merged_cases=merged_cases,
    )

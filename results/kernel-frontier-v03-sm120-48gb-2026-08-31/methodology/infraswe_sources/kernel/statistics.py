from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot compute percentile of empty values")
    if not 0 <= probability <= 1:
        raise ValueError("probability must be inside [0, 1]")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def summarize_samples(values: Sequence[float]) -> dict[str, float | int]:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("samples must be non-empty, finite, and positive")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    median = statistics.median(values)
    deviations = [abs(value - median) for value in values]
    return {
        "n": len(values),
        "median": median,
        "mean": mean,
        "p10": percentile(values, 0.10),
        "p90": percentile(values, 0.90),
        "cv": standard_deviation / mean if mean else 0.0,
        "mad": statistics.median(deviations),
    }


def paired_log_speedup(reference: Sequence[float], candidate: Sequence[float]) -> float:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("paired samples must have the same non-zero length")
    if any(value <= 0 for value in (*reference, *candidate)):
        raise ValueError("paired latencies must be positive")
    estimates = [math.log(ref / cand) for ref, cand in zip(reference, candidate, strict=True)]
    return math.exp(statistics.median(estimates))


def paired_log_speedup_ci(
    reference: Sequence[float],
    candidate: Sequence[float],
    *,
    confidence: float = 0.95,
    resamples: int = 10_000,
    seed: int = 0,
) -> tuple[float, float]:
    if len(reference) != len(candidate) or not reference:
        raise ValueError("paired samples must have the same non-zero length")
    if not 0 < confidence < 1 or resamples <= 0:
        raise ValueError("invalid bootstrap configuration")
    logs = [math.log(ref / cand) for ref, cand in zip(reference, candidate, strict=True)]
    if any(not math.isfinite(value) for value in logs):
        raise ValueError("paired samples must be finite and positive")
    generator = random.Random(seed)
    estimates = []
    for _ in range(resamples):
        sample = [logs[generator.randrange(len(logs))] for _ in logs]
        estimates.append(math.exp(statistics.median(sample)))
    tail = (1 - confidence) / 2
    return percentile(estimates, tail), percentile(estimates, 1 - tail)

from __future__ import annotations

import random
from collections.abc import Sequence


def cluster_bootstrap_means(
    clusters: Sequence[Sequence[float]], *, samples: int = 1000, seed: int = 0
) -> list[float]:
    if not clusters:
        return []
    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(samples):
        draw = [generator.choice(clusters) for _ in clusters]
        values = [value for cluster in draw for value in cluster]
        means.append(sum(values) / len(values))
    return means

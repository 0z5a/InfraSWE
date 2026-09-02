from __future__ import annotations

import math


def rasch_probability(ability: float, difficulty: float) -> float:
    return 1.0 / (1.0 + math.exp(-(ability - difficulty)))


def swe_ability(ability: float) -> float:
    return 50.0 + 10.0 * ability

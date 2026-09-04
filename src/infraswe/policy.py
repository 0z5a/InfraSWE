"""Versioned cross-cutting decision-policy constants."""

from __future__ import annotations

import math
from typing import Literal

POLARIZED_DECISION_POLICY_ID = "project-mergeability-score-bands-v0.1"
HISTORICAL_POLARIZED_ORACLE_POLICY_ID = "historical-score-band-oracle-v0.1"
HISTORICAL_SCORE_BAND_PREDICTION_POLICY_ID = "historical-merge-prediction-v0.1-score-bands"
LEGACY_POLARIZED_DECISION_POLICY_ID = "project-mergeability-polarized-v0.5.1"
LEGACY_HISTORICAL_POLARIZED_ORACLE_POLICY_ID = "historical-polarized-oracle-v0.5.1"
LEGACY_HISTORICAL_POLARIZED_PREDICTION_POLICY_ID = "historical-merge-prediction-v0.5-r2-polarized"
OVERALL_SCORE_REJECT_BELOW_100 = 50.0
OVERALL_SCORE_ACCEPT_ABOVE_100 = 65.0
# Retained for already published historical protocols and audit replays.
MERGE_ACCEPT_SCORE_FLOOR_100 = 85.0
DEFAULT_SEAL_ENABLED = True
DEFAULT_EVALUATION_ENGINE = "infraswe"
DEFAULT_EVALUATION_SCOPE = "full"
CHECK_NEW_PR_MAX_AGE_DAYS = 30
CHECK_ACTIVITY_MAX_IDLE_DAYS = 14
STALE_REVIEWED_OPEN_MIN_AGE_DAYS = 90


def overall_score_decision_band(
    score_100: float,
) -> Literal["accept", "check", "reject"]:
    """Map the sole InfraSWE overall score to its three disposition classes."""

    score = float(score_100)
    if not math.isfinite(score) or not 0 <= score <= 100:
        raise ValueError("InfraSWE overall score must be finite and in [0, 100]")
    if score < OVERALL_SCORE_REJECT_BELOW_100:
        return "reject"
    if score <= OVERALL_SCORE_ACCEPT_ABOVE_100:
        return "check"
    return "accept"


# Backward-compatible aliases for already published historical protocols.
PROJECT_FIT_REJECT_BELOW_100 = OVERALL_SCORE_REJECT_BELOW_100
PROJECT_FIT_ACCEPT_ABOVE_100 = OVERALL_SCORE_ACCEPT_ABOVE_100
project_fit_decision_band = overall_score_decision_band
REVISE_NEW_PR_MAX_AGE_DAYS = CHECK_NEW_PR_MAX_AGE_DAYS
REVISE_ACTIVITY_MAX_IDLE_DAYS = CHECK_ACTIVITY_MAX_IDLE_DAYS

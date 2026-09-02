from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from infraswe.models.draft import DraftPrecompilePolicy


class DraftPrecompileDecision(BaseModel):
    """Deterministic execution decision; no learned or weighted policy is involved."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    compilation_required: bool
    cache_hit: bool
    action: Literal[
        "skip-no-compilation",
        "reuse-precompiled-artifact",
        "precompile-before-timed-cases",
        "compile-inline-with-warning",
    ]
    timed_phases: list[Literal["precompile", "cold-start", "steady-state"]]
    steady_state_compile_allowed: Literal[False] = False
    rationale_codes: list[str]


def decide_precompile(
    policy: DraftPrecompilePolicy,
    *,
    compilation_required: bool,
    cache_hit: bool,
) -> DraftPrecompileDecision:
    """Resolve the Draft switch before any benchmark timing begins."""

    if not compilation_required:
        action = "skip-no-compilation"
        rationale = ["COMPILATION_NOT_REQUIRED"]
    elif cache_hit:
        action = "reuse-precompiled-artifact"
        rationale = ["PRECOMPILE_CACHE_HIT"]
    elif policy.mode == "auto":
        action = "precompile-before-timed-cases"
        rationale = ["UNAVOIDABLE_COMPILE_MOVED_BEFORE_TIMED_CASES"]
    else:
        action = "compile-inline-with-warning"
        rationale = ["PRECOMPILE_DISABLED_FOR_REQUIRED_BUILD"]

    return DraftPrecompileDecision(
        compilation_required=compilation_required,
        cache_hit=cache_hit,
        action=action,
        timed_phases=list(policy.timing_phases),
        rationale_codes=rationale,
    )

from __future__ import annotations

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.draft import (
    AffectedCase,
    AffectedCaseDecision,
    AffectedCasePlan,
    EvidenceCacheIdentity,
)

REQUIRED_CASE_CATEGORIES = {
    "positive",
    "negative-control",
    "fallback-or-unsupported",
    "hidden-adjacent",
    "build-import-load",
}


def select_affected_cases(
    *, changed_symbols: list[str], cases: list[AffectedCase]
) -> AffectedCasePlan:
    if not changed_symbols:
        raise ValueError("affected-case selection requires changed symbols")
    if not cases:
        raise ValueError("affected-case selection requires a case catalog")
    changed = set(changed_symbols)
    selected: dict[str, list[str]] = {}
    for case in cases:
        reasons: list[str] = []
        if case.required:
            reasons.append("project-profile-required")
        overlap = sorted(changed & set(case.symbols))
        if overlap:
            reasons.append("changed-symbol:" + ",".join(overlap))
        if reasons:
            selected[case.case_id] = reasons

    observed_catalog_categories = {category for case in cases for category in case.categories}
    failures = []
    missing_from_catalog = sorted(REQUIRED_CASE_CATEGORIES - observed_catalog_categories)
    if missing_from_catalog:
        failures.append("AFFECTED_CASE_CATEGORY_MISSING")
    for category in sorted(REQUIRED_CASE_CATEGORIES):
        if any(case.case_id in selected and category in case.categories for case in cases):
            continue
        candidate = next((case for case in cases if category in case.categories), None)
        if candidate is not None:
            selected.setdefault(candidate.case_id, []).append(f"mandatory-category:{category}")

    decisions = [
        AffectedCaseDecision(
            case_id=case.case_id,
            selected=case.case_id in selected,
            reasons=selected.get(case.case_id, ["not-affected-and-not-mandatory"]),
        )
        for case in cases
    ]
    selected_cases = {decision.case_id for decision in decisions if decision.selected}
    covered_symbols = {
        symbol
        for case in cases
        if case.case_id in selected_cases
        for symbol in case.symbols
        if symbol in changed
    }
    all_categories_present = not missing_from_catalog and all(
        any(case.case_id in selected_cases and category in case.categories for case in cases)
        for category in REQUIRED_CASE_CATEGORIES
    )
    confidence = (
        "high"
        if all_categories_present and covered_symbols == changed
        else ("medium" if all_categories_present else "low")
    )
    return AffectedCasePlan(
        changed_symbols=changed_symbols,
        decisions=decisions,
        coverage_confidence=confidence,
        required_categories_present=all_categories_present,
        failure_codes=failures,
    )


def evidence_cache_key(identity: EvidenceCacheIdentity) -> str:
    return canonical_sha256(identity)

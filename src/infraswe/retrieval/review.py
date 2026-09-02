from __future__ import annotations

from collections.abc import Sequence

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.retrieval import HumanRuleDecision, RuleCandidate

_ACTION_STATUS = {
    "accept": "accepted",
    "advisory-only": "advisory-only",
    "reject": "rejected",
    "conflict-unresolved": "conflicted",
}


def apply_human_rule_decisions(
    rules: Sequence[RuleCandidate],
    decisions: Sequence[HumanRuleDecision],
    *,
    edited_rules: Sequence[RuleCandidate] = (),
) -> list[RuleCandidate]:
    rule_map = {rule.rule_id: rule for rule in rules}
    if len(rule_map) != len(rules):
        raise ValueError("rule candidates must have unique ids")
    decision_map = {decision.rule_id: decision for decision in decisions}
    if len(decision_map) != len(decisions):
        raise ValueError("human rule decisions must be unique per rule")
    unknown = sorted(set(decision_map) - set(rule_map))
    if unknown:
        raise ValueError("human rule decisions reference unknown rules: " + ", ".join(unknown))
    edited_map = {rule.rule_id: rule for rule in edited_rules}
    if len(edited_map) != len(edited_rules):
        raise ValueError("edited rule candidates must have unique ids")

    reviewed: list[RuleCandidate] = []
    for rule_id in sorted(rule_map):
        rule = rule_map[rule_id]
        decision = decision_map.get(rule_id)
        if decision is None:
            reviewed.append(rule)
            continue
        if rule.status != "proposed":
            raise ValueError("human review can only consume proposed rule candidates")
        if decision.before_sha256 != canonical_sha256(rule):
            raise ValueError(f"human rule review before digest mismatch: {rule_id}")
        if decision.action == "accept-with-edits":
            updated = edited_map.get(rule_id)
            if updated is None:
                raise ValueError(f"accept-with-edits requires an edited rule: {rule_id}")
            if updated.status != "edited":
                raise ValueError("human-edited rules must use status=edited")
            if (
                updated.source_precedents != rule.source_precedents
                or updated.authority != rule.authority
            ):
                raise ValueError("human edits cannot change rule provenance or authority")
        else:
            if rule_id in edited_map:
                raise ValueError("edited rule supplied for a non-edit review action")
            updated = rule.model_copy(update={"status": _ACTION_STATUS[decision.action]})
        if decision.after_sha256 != canonical_sha256(updated):
            raise ValueError(f"human rule review after digest mismatch: {rule_id}")
        reviewed.append(updated)
    unused_edits = sorted(set(edited_map) - set(decision_map))
    if unused_edits:
        raise ValueError("edited rules lack human decisions: " + ", ".join(unused_edits))
    return reviewed


def contract_executable_rules(rules: Sequence[RuleCandidate]) -> list[RuleCandidate]:
    """Return only human-accepted rules that a D3 contract may consume."""

    return sorted(
        (rule for rule in rules if rule.status in {"accepted", "edited"}),
        key=lambda rule: rule.rule_id,
    )

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from infraswe.agentic.protocol import build_sealed
from infraswe.models.agentic import LegacyExperienceManifest, LegacyExperienceRecord

_GROUP_ARTIFACTS = (
    "input-lock.json",
    "exact-head-evidence.json",
    "exact-head-infra-rerun.json",
    "judgment-locks.json",
    "outcome-reveal.json",
    "oracle-audit.json",
    "next-policy.json",
)


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"legacy artifact must contain an object: {path}")
    return value


def _file_sha256(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _by_case(values: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in values:
        material = item.get("material")
        nested_case_id = material.get("case_id") if isinstance(material, dict) else None
        case_id = item.get("case_id", nested_case_id)
        if case_id is None:
            raise ValueError("legacy case-indexed artifact entry has no case_id")
        normalized = str(case_id)
        if normalized in indexed:
            raise ValueError(f"legacy artifact repeats case_id: {normalized}")
        indexed[normalized] = item
    return indexed


def _decision(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    if normalized == "accept_with_scope":
        return "accept"
    if normalized == "revise":
        return "check"
    if normalized in {"accept", "check", "reject", "unresolved"}:
        return normalized
    return None


def _domain(profile: str, root: Path) -> str:
    if profile in {"training", "inference", "communication"}:
        return profile
    name = root.name.lower()
    for candidate in ("training", "inference", "communication"):
        if candidate in name:
            return candidate
    return "other"


def _group_records(root: Path, group: Path) -> list[LegacyExperienceRecord]:
    input_lock = _read(group / "input-lock.json")
    exact = _read(group / "exact-head-evidence.json")
    judgment = _read(group / "judgment-locks.json")
    reveal = _read(group / "outcome-reveal.json")
    audit = _read(group / "oracle-audit.json")

    exact_by_id = _by_case(exact.get("records", []))
    rerun_path = group / "exact-head-infra-rerun.json"
    if rerun_path.is_file():
        exact_by_id.update(_by_case(_read(rerun_path).get("records", [])))
    locks_by_id = _by_case(judgment.get("locks", []))
    reveal_by_id = _by_case(reveal.get("cases", []))
    audit_by_id = _by_case(audit.get("cases", []))
    source_artifacts = {
        name: _file_sha256(group / name) for name in _GROUP_ARTIFACTS if (group / name).is_file()
    }
    profile = str(input_lock.get("profile", ""))
    domain = _domain(profile, root)
    group_id = f"{root.name}/{group.name}"
    records: list[LegacyExperienceRecord] = []
    for case in input_lock.get("cases", []):
        case_id = str(case["case_id"])
        locked = locks_by_id.get(case_id, {})
        material = locked.get("material", {})
        revealed = reveal_by_id.get(case_id, {})
        audited = audit_by_id.get(case_id, {})
        acquisition_status = str(case.get("acquisition_status", "acquired"))
        outcome = revealed.get("outcome", {})
        oracle_availability = str(
            outcome.get(
                "availability",
                "invalid" if acquisition_status == "invalid" else "available",
            )
        )
        machine = _decision(
            revealed.get("machine_decision")
            or audited.get("machine_decision")
            or material.get("decision")
        )
        oracle = _decision(revealed.get("oracle_decision") or audited.get("oracle_decision"))
        if acquisition_status == "invalid":
            machine = "unresolved"
            oracle = "unresolved"
        invalid_reason = None
        if acquisition_status == "invalid":
            invalid_reason = str(
                case.get("acquisition_failure_code") or "LEGACY_ACQUISITION_INVALID"
            )
        elif oracle_availability != "available":
            invalid_reason = str(outcome.get("failure_code") or "LEGACY_ORACLE_UNAVAILABLE")
        elif oracle in {None, "unresolved"}:
            invalid_reason = "LEGACY_ORACLE_UNRESOLVED"
        experience_status = "invalid" if invalid_reason is not None else "valid"
        record = build_sealed(
            LegacyExperienceRecord,
            case_id=case_id,
            domain=domain,
            project=str(case.get("project", "unknown")),
            repository=str(case.get("repository", "unknown/unknown")),
            pull_number=int(case.get("pull_number", 1)),
            group_id=group_id,
            source_artifact_sha256s=source_artifacts,
            acquisition_status=acquisition_status,
            oracle_availability=oracle_availability,
            experience_status=experience_status,
            invalid_reason=invalid_reason,
            execution_status=exact_by_id.get(case_id, {}).get("status"),
            machine_decision=machine,
            oracle_decision=oracle,
            trajectory_fidelity="reconstructed",
            harness_fidelity="transcript-only",
            exact_token_ids_available=False,
            policy_gradient_eligible=False,
            reward_qualification="not-qualified",
            reward_pack_sha256=None,
            allowed_uses=[
                "external-policy",
                "curriculum",
                "offline-retrieval",
                "qualitative-audit",
            ],
        )
        records.append(record)
    return records


def build_legacy_experience_manifest(
    source_roots: list[Path],
    *,
    manifest_id: str,
    generated_at: datetime | None = None,
) -> LegacyExperienceManifest:
    """Wrap only completed historical groups without fabricating tokens or rewards."""

    if not source_roots:
        raise ValueError("legacy migration requires at least one source root")
    records: list[LegacyExperienceRecord] = []
    completed_groups = 0
    for source_root in source_roots:
        root = source_root.resolve()
        if not root.is_dir():
            raise ValueError(f"legacy source root does not exist: {root}")
        for group in sorted((root / "groups").glob("group-*")):
            required = [
                group / "input-lock.json",
                group / "exact-head-evidence.json",
                group / "judgment-locks.json",
                group / "outcome-reveal.json",
                group / "oracle-audit.json",
            ]
            if not all(path.is_file() for path in required):
                continue
            records.extend(_group_records(root, group))
            completed_groups += 1
    if not completed_groups or not records:
        raise ValueError("legacy migration found no completed historical groups")
    identifiers = [item.case_id for item in records]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("legacy migration found duplicate case ids across source roots")
    invalid = sum(item.experience_status == "invalid" for item in records)
    return build_sealed(
        LegacyExperienceManifest,
        manifest_id=manifest_id,
        generated_at=generated_at or datetime.now(UTC),
        source_roots=[str(path) for path in source_roots],
        records=records,
        attempted_records=len(records),
        valid_records=len(records) - invalid,
        invalid_records=invalid,
        policy_gradient_eligible_records=0,
        fabricated_token_records=0,
        qualified_reward_records=0,
    )

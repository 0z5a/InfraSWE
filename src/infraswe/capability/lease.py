from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

from infraswe.capability.resolver import audit_capability_resolution
from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.capability import (
    CapabilityResolution,
    ResourceLease,
    RunnerSnapshot,
)

_ZERO_DIGEST = "sha256:" + "0" * 64


def _audit_snapshot(snapshot: RunnerSnapshot) -> bool:
    material = snapshot.model_dump(mode="json", exclude={"snapshot_sha256"})
    return snapshot.snapshot_sha256 == canonical_sha256(material)


def _critical_identity(snapshot: RunnerSnapshot) -> dict[str, Any]:
    return {
        "runner_manifest_sha256": snapshot.runner_manifest_sha256,
        "topology_sha256": snapshot.topology_sha256,
        "devices": [
            {
                "device_id": item.device_id,
                "kind": item.kind,
                "pci_bdf": item.pci_bdf,
                "numa_node": item.numa_node,
                "partition_mode": item.partition_mode,
            }
            for item in sorted(snapshot.devices, key=lambda value: value.device_id)
        ],
        "resource_totals": {
            name: value.total for name, value in sorted(snapshot.resources.items())
        },
    }


def build_resource_lease(
    *,
    resolution: CapabilityResolution,
    pre_lease_snapshot: RunnerSnapshot,
    post_lease_snapshot: RunnerSnapshot,
    allocations: dict[str, Any],
    isolation: dict[str, Any],
    lease_id: str,
    acquired_at: datetime | None = None,
    expires_at: datetime | None = None,
    heartbeat_interval_s: int = 5,
) -> ResourceLease:
    if audit_capability_resolution(resolution) or resolution.status != "eligible":
        raise ValueError("resource lease requires an eligible sealed resolution")
    if not _audit_snapshot(pre_lease_snapshot) or not _audit_snapshot(post_lease_snapshot):
        raise ValueError("resource lease snapshots have invalid digests")
    if resolution.selected_runner_snapshot_sha256 != pre_lease_snapshot.snapshot_sha256:
        raise ValueError("pre-lease snapshot does not match capability resolution")

    failures: list[str] = []
    if _critical_identity(pre_lease_snapshot) != _critical_identity(post_lease_snapshot):
        failures.append("CELL_DRIFT_BEFORE_RUN")
    if isolation.get("device_visibility") != "exact":
        failures.append("RESOURCE_DEVICE_VISIBILITY_NOT_EXACT")
    if isolation.get("process_namespace") != "isolated":
        failures.append("RESOURCE_PROCESS_NAMESPACE_NOT_ISOLATED")
    if not allocations:
        failures.append("RESOURCE_ALLOCATIONS_EMPTY")
    timestamp = acquired_at or datetime.now(UTC)
    expiry = expires_at or timestamp + timedelta(hours=1)
    preliminary = ResourceLease(
        lease_id=lease_id,
        resolution_sha256=resolution.resolution_sha256,
        status="broken" if failures else "active",
        allocations=allocations,
        isolation=isolation,
        acquired_at=timestamp,
        expires_at=expiry,
        heartbeat_interval_s=heartbeat_interval_s,
        pre_lease_snapshot_sha256=pre_lease_snapshot.snapshot_sha256,
        post_lease_snapshot_sha256=post_lease_snapshot.snapshot_sha256,
        failure_codes=failures,
        lease_sha256=_ZERO_DIGEST,
    )
    material = preliminary.model_dump(mode="json", exclude={"lease_sha256"})
    return preliminary.model_copy(update={"lease_sha256": canonical_sha256(material)})


def audit_resource_lease(lease: ResourceLease) -> list[str]:
    material = lease.model_dump(mode="json", exclude={"lease_sha256"})
    return (
        []
        if lease.lease_sha256 == canonical_sha256(material)
        else ["RESOURCE_LEASE_DIGEST_MISMATCH"]
    )

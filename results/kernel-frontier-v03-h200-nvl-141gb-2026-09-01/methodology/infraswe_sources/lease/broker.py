from __future__ import annotations

import secrets
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from infraswe.io import atomic_write_json


@dataclass(frozen=True)
class Lease:
    lease_id: str
    provider: str
    profile: str
    acquired_at: str
    expires_at: str
    max_cost_usd: float
    state: str = "ACTIVE"


class LocalLeaseBroker:
    """Records a TTL-bounded lease for an already-provisioned development host."""

    def acquire(
        self,
        *,
        profile: str,
        ttl_minutes: int,
        max_cost_usd: float,
        output: Path,
    ) -> Lease:
        now = datetime.now(UTC)
        lease = Lease(
            lease_id=f"local-{secrets.token_hex(6)}",
            provider="local-existing-host",
            profile=profile,
            acquired_at=now.isoformat(),
            expires_at=(now + timedelta(minutes=ttl_minutes)).isoformat(),
            max_cost_usd=max_cost_usd,
        )
        atomic_write_json(output, asdict(lease))
        return lease

    def release(self, lease: Lease, output: Path) -> Lease:
        released = Lease(**{**asdict(lease), "state": "RELEASED"})
        atomic_write_json(output, asdict(released))
        return released

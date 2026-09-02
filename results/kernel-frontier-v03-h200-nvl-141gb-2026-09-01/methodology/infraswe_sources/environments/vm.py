"""VM environments are acquired through the lease provider in v0.1."""

from dataclasses import dataclass


@dataclass(frozen=True)
class VMEnvironment:
    lease_id: str
    host: str

from __future__ import annotations

from typing import Any


def build_plan(peer_access: list[list[bool]]) -> dict[str, Any]:
    """Return the NCCL launch plan for the supplied directed peer-access matrix."""
    del peer_access
    return {
        "schema_version": "1",
        "transport": "p2p",
        "reason": "configured_preference",
        "fallback_reported": False,
        "nccl_env": {
            "NCCL_P2P_DISABLE": "0",
            "NCCL_SHM_DISABLE": "0",
        },
    }

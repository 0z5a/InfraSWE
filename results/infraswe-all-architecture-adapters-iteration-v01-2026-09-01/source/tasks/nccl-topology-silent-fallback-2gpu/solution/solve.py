from __future__ import annotations

from pathlib import Path

source = """from __future__ import annotations

from typing import Any


def _complete_peer_access(peer_access: list[list[bool]]) -> bool:
    size = len(peer_access)
    if size < 2 or any(len(row) != size for row in peer_access):
        return False
    return all(
        bool(peer_access[source][target])
        for source in range(size)
        for target in range(size)
        if source != target
    )


def build_plan(peer_access: list[list[bool]]) -> dict[str, Any]:
    if _complete_peer_access(peer_access):
        return {
            "schema_version": "1",
            "transport": "p2p",
            "reason": "complete_peer_access",
            "fallback_reported": False,
            "nccl_env": {
                "NCCL_P2P_DISABLE": "0",
                "NCCL_SHM_DISABLE": "0",
            },
        }
    return {
        "schema_version": "1",
        "transport": "shm",
        "reason": "p2p_unavailable_or_asymmetric",
        "fallback_reported": True,
        "nccl_env": {
            "NCCL_P2P_DISABLE": "1",
            "NCCL_SHM_DISABLE": "0",
        },
    }
"""

Path("launch_policy.py").write_text(source, encoding="utf-8")

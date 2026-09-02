from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

CAPABILITY_SCHEMA_VERSION = "0.1"
MINIMUM_CUDA_SERIES = "13.0"
CANONICAL_CUDA_SERIES = "13.3"
CANONICAL_PTX_ISA = "9.3"
MATCHER_VERSION = "cuda-binary-utilities-sm121-v1"

TargetLane = Literal["sm_121", "sm_121f", "sm_121a"]
PackNamespace = Literal[
    "GB10-SM121-Codegen",
    "GB10-SM121-LowP",
    "GB10-SM121-TMA",
    "GB10-UMA",
    "GB10-Arm64",
    "GB10-Runtime",
    "GB10-RoCE",
]

TARGET_LANES: dict[TargetLane, dict[str, object]] = {
    "sm_121": {
        "kind": "baseline",
        "forward_compatible": True,
        "description": "SM121 baseline deployment target",
    },
    "sm_121f": {
        "kind": "family",
        "forward_compatible": True,
        "description": "SM12x family-specific target",
    },
    "sm_121a": {
        "kind": "architecture",
        "forward_compatible": False,
        "description": "SM121 architecture-specific target",
    },
}

PACK_NAMESPACES: tuple[PackNamespace, ...] = (
    "GB10-SM121-Codegen",
    "GB10-SM121-LowP",
    "GB10-SM121-TMA",
    "GB10-UMA",
    "GB10-Arm64",
    "GB10-Runtime",
    "GB10-RoCE",
)


@dataclass(frozen=True)
class GB10FeatureContract:
    feature_id: str
    title: str
    namespace: PackNamespace
    phase: str
    required_target: TargetLane | None
    minimum_ptx_isa: str | None
    ptx_require_all: tuple[str, ...] = ()
    ptx_require_any: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()
    host_isa_require_any: tuple[str, ...] = ()
    runtime_requirements: tuple[str, ...] = ()
    leaderboard_eligible: bool = True

    def as_manifest(self) -> dict[str, object]:
        return asdict(self)


_GB10_FORBIDDEN = (
    r"\btcgen05\.",
    r"\btmem\b",
    r"\btarget\s+sm_100a\b",
)
_OPAQUE_GEMM_FALLBACKS = (r"\bcublas(?:lt)?\b", r"\bcutlass_profiler\b")

FEATURE_CONTRACTS: dict[str, GB10FeatureContract] = {
    "GB10-TARGET-001": GB10FeatureContract(
        feature_id="GB10-TARGET-001",
        title="SM121 native build and runtime dispatch",
        namespace="GB10-SM121-Codegen",
        phase="p0",
        required_target="sm_121",
        minimum_ptx_isa=None,
        forbidden_patterns=_GB10_FORBIDDEN,
        runtime_requirements=("aarch64", "compute_capability_12_1"),
    ),
    "GB10-MMA-001": GB10FeatureContract(
        feature_id="GB10-MMA-001",
        title="SM121 block-scaled warp MMA",
        namespace="GB10-SM121-LowP",
        phase="p1",
        required_target="sm_121a",
        minimum_ptx_isa="9.3",
        ptx_require_all=(r"\bmma\.sync\b", r"\.block_scale\b"),
        ptx_require_any=(r"\.kind::mxf4nvf4\b", r"\.kind::mxf4\b"),
        forbidden_patterns=(*_GB10_FORBIDDEN, *_OPAQUE_GEMM_FALLBACKS),
    ),
    "GB10-MATRIX-IO-001": GB10FeatureContract(
        feature_id="GB10-MATRIX-IO-001",
        title="SM121 low-bit matrix I/O conversion",
        namespace="GB10-SM121-LowP",
        phase="p2",
        required_target="sm_121f",
        minimum_ptx_isa="9.3",
        ptx_require_all=(r"\bldmatrix\.(?:m16n16|m8n16)\b",),
        ptx_require_any=(r"\.b6x16_p32\b", r"\.b4x16_p64\b"),
        forbidden_patterns=(*_GB10_FORBIDDEN, r"global[_ -]?temporary[_ -]?unpack"),
    ),
    "GB10-TMA-001": GB10FeatureContract(
        feature_id="GB10-TMA-001",
        title="SM121 CTA-local TMA gather4",
        namespace="GB10-SM121-TMA",
        phase="p2",
        required_target="sm_121",
        minimum_ptx_isa="9.0",
        ptx_require_all=(
            r"\bcp\.async\.bulk\.tensor\b",
            r"\.tile::gather4\b",
            r"\.shared::cta\b",
        ),
        forbidden_patterns=(
            *_GB10_FORBIDDEN,
            r"\.tile::scatter4\b",
            r"\.cta_group::2\b",
        ),
        runtime_requirements=("tensor_map_access_supported",),
    ),
    "GB10-UMA-001": GB10FeatureContract(
        feature_id="GB10-UMA-001",
        title="GB10 CPU/GPU unified-memory pipeline",
        namespace="GB10-UMA",
        phase="p1",
        required_target=None,
        minimum_ptx_isa=None,
        runtime_requirements=(
            "unified_addressing",
            "pageable_memory_access",
            "host_page_table_coherence",
        ),
    ),
    "GB10-ARM-ORDER-001": GB10FeatureContract(
        feature_id="GB10-ARM-ORDER-001",
        title="GB10 Arm weak-memory-ordering ring buffer",
        namespace="GB10-Arm64",
        phase="p2",
        required_target=None,
        minimum_ptx_isa=None,
        host_isa_require_any=(r"\bcas(?:p)?\b", r"\bswp\b", r"\bldadd\b", r"\bldset\b"),
        runtime_requirements=("aarch64", "host_native_atomics"),
    ),
    "GB10-ROCE-001": GB10FeatureContract(
        feature_id="GB10-ROCE-001",
        title="Pinned-host staged RoCE transport",
        namespace="GB10-RoCE",
        phase="p3-optional",
        required_target=None,
        minimum_ptx_isa=None,
        runtime_requirements=("minimum_nodes_2", "gpudirect_rdma_not_assumed"),
        leaderboard_eligible=False,
    ),
}

MINIMUM_RELEASE_FEATURE_IDS = (
    "GB10-TARGET-001",
    "GB10-MMA-001",
    "GB10-UMA-001",
    "GB10-MATRIX-IO-001",
    "GB10-ARM-ORDER-001",
)


def version_tuple(value: str) -> tuple[int, ...]:
    parts = value.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid dotted version: {value}")
    return tuple(int(part) for part in parts)


def target_satisfies(observed: str, required: TargetLane) -> bool:
    compatibility: dict[TargetLane, set[str]] = {
        "sm_121": {"sm_121", "sm_121f", "sm_121a"},
        "sm_121f": {"sm_121f", "sm_121a"},
        "sm_121a": {"sm_121a"},
    }
    return observed in compatibility[required]


def capability_contract_manifest() -> dict[str, object]:
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "minimum_cuda_series": MINIMUM_CUDA_SERIES,
        "canonical_cuda_series": CANONICAL_CUDA_SERIES,
        "canonical_ptx_isa": CANONICAL_PTX_ISA,
        "matcher_version": MATCHER_VERSION,
        "target_lanes": TARGET_LANES,
        "pack_namespaces": PACK_NAMESPACES,
        "minimum_release_feature_ids": MINIMUM_RELEASE_FEATURE_IDS,
        "features": {
            feature_id: contract.as_manifest() for feature_id, contract in FEATURE_CONTRACTS.items()
        },
    }

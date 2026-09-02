from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

CAPABILITY_SCHEMA_VERSION = "0.1"
NATIVE_EVIDENCE_SCHEMA_VERSION = "0.1"
STABLE_CUDA_SERIES = "13.3"
STABLE_PTX_ISA = "9.3"
NATIVE_MATCHER_VERSION = "cuda-binary-utilities-13.3-sm100-v2"

TargetLane = Literal["sm_100", "sm_100f", "sm_100a"]
ScoreNamespace = Literal["SM100-Core", "SM100-Scheduler", "SM100-Fabric", "PTX-Preview"]

TARGET_LANES: dict[TargetLane, dict[str, object]] = {
    "sm_100": {
        "kind": "generic",
        "forward_compatible": True,
        "description": "portable compute capability 10.0 target",
    },
    "sm_100f": {
        "kind": "family",
        "forward_compatible": True,
        "description": "Blackwell family-specific target",
    },
    "sm_100a": {
        "kind": "architecture",
        "forward_compatible": False,
        "description": "B200/SM100 architecture-specific target",
    },
}

SCORE_NAMESPACES: tuple[ScoreNamespace, ...] = (
    "SM100-Core",
    "SM100-Scheduler",
    "SM100-Fabric",
    "PTX-Preview",
)


@dataclass(frozen=True)
class FeatureContract:
    feature_id: str
    title: str
    namespace: ScoreNamespace
    phase: str
    required_target: TargetLane
    minimum_ptx_isa: str
    ptx_require_all: tuple[str, ...]
    ptx_require_any: tuple[str, ...]
    sass_require_all: tuple[str, ...]
    sass_require_any: tuple[str, ...]
    forbidden_patterns: tuple[str, ...]
    runtime_scope: Literal["single_gpu", "multi_gpu", "compile_only"]
    leaderboard_eligible: bool

    def as_manifest(self) -> dict[str, object]:
        return asdict(self)


_COMMON_FALLBACKS = (
    r"\bcublas(?:lt)?\b",
    r"\bcudnn\b",
)

FEATURE_CONTRACTS: dict[str, FeatureContract] = {
    "BW-TMEM-001": FeatureContract(
        feature_id="BW-TMEM-001",
        title="TMEM lifecycle and TCGen05 MMA",
        namespace="SM100-Core",
        phase="mvp",
        required_target="sm_100a",
        minimum_ptx_isa="8.6",
        ptx_require_all=(
            r"\btcgen05\.alloc(?:\.|\s)",
            r"\btcgen05\.mma(?:\.|\s)",
            r"\btcgen05\.dealloc(?:\.|\s)",
            r"\btcgen05\.relinquish_alloc_permit(?:\.|\s|;)",
        ),
        ptx_require_any=(),
        sass_require_all=(),
        sass_require_any=(
            r"\bUTCHMMA\b",
            r"\bUTCIMMA\b",
            r"\bUTCOMMA\b",
            r"\bUTCQMMA\b",
        ),
        forbidden_patterns=(*_COMMON_FALLBACKS, r"\bwgmma\.mma_async\b"),
        runtime_scope="single_gpu",
        leaderboard_eligible=True,
    ),
    "BW-CLC-001": FeatureContract(
        feature_id="BW-CLC-001",
        title="Cluster Launch Control work stealing",
        namespace="SM100-Scheduler",
        phase="mvp",
        required_target="sm_100",
        minimum_ptx_isa="8.6",
        ptx_require_all=(
            r"\bclusterlaunchcontrol\.try_cancel(?:\.|\s)",
            r"\bclusterlaunchcontrol\.query_cancel\.is_canceled(?:\.|\s)",
        ),
        ptx_require_any=(),
        sass_require_all=(r"\bUGETNEXTWORKID\b",),
        sass_require_any=(),
        forbidden_patterns=_COMMON_FALLBACKS,
        runtime_scope="single_gpu",
        leaderboard_eligible=True,
    ),
    "BW-TMA-001": FeatureContract(
        feature_id="BW-TMA-001",
        title="Irregular TMA gather4/scatter4",
        namespace="SM100-Core",
        phase="mvp",
        required_target="sm_100a",
        minimum_ptx_isa="8.6",
        # Candidate packs commonly expose gather and scatter as two entry
        # points.  The entry-local reachability gate certifies either path;
        # the v0.2 evaluator additionally requires both across the artifact.
        ptx_require_all=(r"\bcp\.async\.bulk\.tensor(?:\.|\s)",),
        ptx_require_any=(r"\.tile::gather4\b", r"\.tile::scatter4\b"),
        sass_require_all=(r"\bUTMALDG\b", r"\bUTMASTG\b"),
        sass_require_any=(),
        forbidden_patterns=_COMMON_FALLBACKS,
        runtime_scope="single_gpu",
        leaderboard_eligible=True,
    ),
    "BW-TMEM-003": FeatureContract(
        feature_id="BW-TMEM-003",
        title="TMEM lifecycle and error-path repair",
        namespace="SM100-Core",
        phase="mvp",
        required_target="sm_100a",
        minimum_ptx_isa="8.6",
        ptx_require_all=(
            r"\btcgen05\.alloc(?:\.|\s)",
            r"\btcgen05\.dealloc(?:\.|\s)",
            r"\btcgen05\.relinquish_alloc_permit(?:\.|\s|;)",
        ),
        ptx_require_any=(),
        sass_require_all=(),
        sass_require_any=(r"\bUTCHMMA\b", r"\bUTCIMMA\b", r"\bUTCOMMA\b", r"\bUTCQMMA\b"),
        forbidden_patterns=(*_COMMON_FALLBACKS, r"\bwgmma\.mma_async\b"),
        runtime_scope="single_gpu",
        leaderboard_eligible=True,
    ),
    "BW-TMA-002": FeatureContract(
        feature_id="BW-TMA-002",
        title="CTA-pair TMA",
        namespace="SM100-Core",
        phase="mvp",
        required_target="sm_100a",
        minimum_ptx_isa="8.6",
        ptx_require_all=(
            r"\bcp\.async\.bulk\.tensor(?:\.|\s)",
            r"\.cta_group::2\b",
        ),
        ptx_require_any=(),
        sass_require_all=(),
        sass_require_any=(r"\bUTMALDG\b", r"\bUTMASTG\b"),
        forbidden_patterns=_COMMON_FALLBACKS,
        runtime_scope="single_gpu",
        leaderboard_eligible=True,
    ),
    "BW-FABRIC-001": FeatureContract(
        feature_id="BW-FABRIC-001",
        title="Asynchronous multimem operation",
        namespace="SM100-Fabric",
        phase="optional",
        required_target="sm_100f",
        minimum_ptx_isa=STABLE_PTX_ISA,
        ptx_require_all=(r"\bmultimem\.(?:st|red)\.async(?:\.|\s)",),
        ptx_require_any=(),
        sass_require_all=(),
        sass_require_any=(r"\bLDGMC\b", r"\bSTGMC\b", r"\bREDGMC\b"),
        forbidden_patterns=_COMMON_FALLBACKS,
        runtime_scope="multi_gpu",
        leaderboard_eligible=False,
    ),
    "BW-PTX-PREVIEW-001": FeatureContract(
        feature_id="BW-PTX-PREVIEW-001",
        title="Reserved PTX post-9.3 preview lane",
        namespace="PTX-Preview",
        phase="preview-disabled",
        required_target="sm_100a",
        minimum_ptx_isa="9.4",
        ptx_require_all=(),
        ptx_require_any=(),
        sass_require_all=(),
        sass_require_any=(),
        forbidden_patterns=_COMMON_FALLBACKS,
        runtime_scope="compile_only",
        leaderboard_eligible=False,
    ),
}

MVP_FEATURE_IDS: tuple[str, ...] = (
    "BW-TMEM-001",
    "BW-CLC-001",
    "BW-TMA-001",
    "BW-TMEM-003",
    "BW-TMA-002",
)


def version_tuple(value: str) -> tuple[int, ...]:
    parts = value.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid dotted version: {value}")
    return tuple(int(part) for part in parts)


def target_satisfies(observed: str, required: TargetLane) -> bool:
    compatibility: dict[TargetLane, set[str]] = {
        "sm_100": {"sm_100", "sm_100f", "sm_100a"},
        "sm_100f": {"sm_100f", "sm_100a"},
        "sm_100a": {"sm_100a"},
    }
    return observed in compatibility[required]


def feature_contract_manifest() -> dict[str, object]:
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "stable_cuda_series": STABLE_CUDA_SERIES,
        "stable_ptx_isa": STABLE_PTX_ISA,
        "matcher_version": NATIVE_MATCHER_VERSION,
        "target_lanes": TARGET_LANES,
        "score_namespaces": SCORE_NAMESPACES,
        "mvp_feature_ids": MVP_FEATURE_IDS,
        "features": {
            feature_id: contract.as_manifest() for feature_id, contract in FEATURE_CONTRACTS.items()
        },
    }

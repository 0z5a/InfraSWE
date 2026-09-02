from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

CAPABILITY_SCHEMA_VERSION = "0.1"
MINIMUM_CUDA_SERIES = "11.8"
CANONICAL_CUDA_SERIES = "13.3"
CANONICAL_PTX_ISA = "9.3"
MATCHER_VERSION = "cuda-binary-utilities-ada-sm89-v1"
NATIVE_TARGET = "sm_89"
PTX_FALLBACK_TARGET = "compute_89"

PlatformCell = Literal["l40s-48gb-pcie", "l20-48gb-pcie", "generic-sm89"]
PackNamespace = Literal[
    "Ada-SM89-Core",
    "Ada-Stability",
    "Ada-Compile-Runtime",
    "Ada-Cross-SKU-Reuse",
    "Ada-Production-Meso",
    "Ada-PCIe-MultiGPU",
    "Ada-vGPU-TimeSlice",
]

PLATFORM_CELLS: dict[PlatformCell, dict[str, object]] = {
    "l40s-48gb-pcie": {
        "product_pattern": r"(?:^|\s)L40S(?:[-\s]|$)",
        "canonical": True,
        "description": "NVIDIA L40S 48 GB PCIe local calibration cell",
    },
    "l20-48gb-pcie": {
        "product_pattern": r"(?:^|\s)L20(?:[-\s]|$)",
        "canonical": True,
        "description": "NVIDIA L20 48 GB PCIe local calibration cell",
    },
    "generic-sm89": {
        "product_pattern": r".*",
        "canonical": False,
        "description": "Explicit opt-in compile/compatibility cell; never a board score cell",
    },
}


@dataclass(frozen=True)
class AdaSM89FeatureContract:
    feature_id: str
    title: str
    namespace: PackNamespace
    phase: str
    required_target: Literal["sm_89"] | None
    minimum_ptx_isa: str | None
    ptx_require_all: tuple[str, ...] = ()
    ptx_require_any: tuple[str, ...] = ()
    sass_require_any: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()
    runtime_requirements: tuple[str, ...] = ()
    supported_cells: tuple[PlatformCell, ...] = (
        "l40s-48gb-pcie",
        "l20-48gb-pcie",
    )
    external_evidence: bool = False
    leaderboard_eligible: bool = True

    def as_manifest(self) -> dict[str, object]:
        return asdict(self)


_FORBIDDEN_NON_ADA = (
    r"\bwgmma(?:\.|\b)",
    r"\bcp\.async\.bulk(?:\.tensor)?(?:\.|\b)",
    r"\btensormap(?:\.|\b)",
    r"\bclusterlaunchcontrol(?:\.|\b)",
    r"%cluster_(?:ctarank|nctarank)\b",
    r"\b(?:mapa|getctarank)(?:\.|\b)",
    r"\btcgen05(?:\.|\b)",
    r"\btmem(?:\.|\s|$)",
    r"\bmultimem(?:\.|\b)",
    r"\bfabric\.try_(?:\.|\b)",
    r"\.(?:e2m1|e2m3|e3m2)\b",
    r"\btarget\s+sm_(?:90a|100a|120a)\b",
)
_OPAQUE_GEMM_FALLBACKS = (
    r"\bcublas(?:lt)?(?:64)?\b",
    r"\bcutlass_profiler\b",
)

FEATURE_CONTRACTS: dict[str, AdaSM89FeatureContract] = {
    "SM89-TARGET-001": AdaSM89FeatureContract(
        feature_id="SM89-TARGET-001",
        title="SM89 native cubin, compute_89 PTX, and two-path dispatch",
        namespace="Ada-SM89-Core",
        phase="p0",
        required_target="sm_89",
        minimum_ptx_isa=None,
        forbidden_patterns=_FORBIDDEN_NON_ADA,
        runtime_requirements=("compute_capability_8_9", "native_and_ptx_jit_dispatch"),
    ),
    "SM89-FP8-MMA-001": AdaSM89FeatureContract(
        feature_id="SM89-FP8-MMA-001",
        title="SM89 warp-level E4M3/E5M2 MMA",
        namespace="Ada-SM89-Core",
        phase="p1",
        required_target="sm_89",
        minimum_ptx_isa="8.7",
        ptx_require_all=(
            r"\bmma\.sync\.aligned\.m16n8k(?:16|32)\b",
            r"\.f32\.e4m3\.(?:e4m3|e5m2)\.f32\b",
            r"\.f32\.e5m2\.(?:e4m3|e5m2)\.f32\b",
        ),
        sass_require_any=(r"\bHMMA\b",),
        forbidden_patterns=(*_FORBIDDEN_NON_ADA, *_OPAQUE_GEMM_FALLBACKS),
        runtime_requirements=("fp32_accumulator", "no_full_size_fp16_temporary"),
    ),
    "SM89-FP8-CVT-001": AdaSM89FeatureContract(
        feature_id="SM89-FP8-CVT-001",
        title="SM89 E4M3/E5M2 convert and pack",
        namespace="Ada-SM89-Core",
        phase="p1",
        required_target="sm_89",
        minimum_ptx_isa="8.1",
        ptx_require_all=(
            r"\bcvt\.rn\.satfinite\b",
            r"\.e4m3x2\.",
            r"\.e5m2x2\.",
        ),
        forbidden_patterns=_FORBIDDEN_NON_ADA,
        runtime_requirements=("odd_tail_masked", "deterministic_replay"),
    ),
    "SM89-CPASYNC-001": AdaSM89FeatureContract(
        feature_id="SM89-CPASYNC-001",
        title="SM89 CTA shared-memory cp.async pipeline",
        namespace="Ada-SM89-Core",
        phase="p1",
        required_target="sm_89",
        minimum_ptx_isa="7.0",
        ptx_require_all=(
            r"\bcp\.async\.(?:ca|cg)\.shared\.global\b",
            r"\bcp\.async\.commit_group\b",
            r"\bcp\.async\.wait_group\b",
        ),
        sass_require_any=(r"\bLDGSTS\b",),
        forbidden_patterns=_FORBIDDEN_NON_ADA,
        runtime_requirements=("tail_zero_fill", "alignment_guard"),
    ),
    "SM89-L2-001": AdaSM89FeatureContract(
        feature_id="SM89-L2-001",
        title="SM89 L2 residency and reuse",
        namespace="Ada-SM89-Core",
        phase="p2",
        required_target=None,
        minimum_ptx_isa=None,
        forbidden_patterns=_FORBIDDEN_NON_ADA,
        runtime_requirements=("runtime_l2_size", "allocation_audit", "cold_warm_samples"),
    ),
    "ADA-CONCURRENCY-001": AdaSM89FeatureContract(
        feature_id="ADA-CONCURRENCY-001",
        title="Ada concurrent mixed-kernel stability",
        namespace="Ada-Stability",
        phase="p2",
        required_target=None,
        minimum_ptx_isa=None,
        forbidden_patterns=_FORBIDDEN_NON_ADA,
        runtime_requirements=("v04_load_ladder", "fresh_process_replays_7"),
        external_evidence=True,
    ),
    "ADA-CROSS-SKU-001": AdaSM89FeatureContract(
        feature_id="ADA-CROSS-SKU-001",
        title="Ada shared-kernel cross-SKU reuse",
        namespace="Ada-Cross-SKU-Reuse",
        phase="p2",
        required_target=None,
        minimum_ptx_isa=None,
        forbidden_patterns=_FORBIDDEN_NON_ADA,
        runtime_requirements=(
            "same_semantic_artifact",
            "independent_board_tuning",
            "both_canonical_cells",
        ),
        external_evidence=True,
    ),
    "ADA-TORCHCOMPILE-001": AdaSM89FeatureContract(
        feature_id="ADA-TORCHCOMPILE-001",
        title="Ada torch.compile specialization and cache stability",
        namespace="Ada-Compile-Runtime",
        phase="p3",
        required_target=None,
        minimum_ptx_isa=None,
        forbidden_patterns=_FORBIDDEN_NON_ADA,
        runtime_requirements=("bounded_specialization", "compile_cache_provenance"),
        external_evidence=True,
    ),
}

MINIMUM_RELEASE_FEATURE_IDS = tuple(FEATURE_CONTRACTS)
CANONICAL_PLATFORM_CELLS = ("l40s-48gb-pcie", "l20-48gb-pcie")


def version_tuple(value: str) -> tuple[int, ...]:
    parts = value.split(".")
    if not parts or any(not part.isdigit() for part in parts):
        raise ValueError(f"invalid dotted version: {value}")
    return tuple(int(part) for part in parts)


def target_satisfies(observed: str, required: Literal["sm_89"]) -> bool:
    """Ada has no family/architecture lane: only an exact sm_89 PTX target is native."""

    return observed == required


def capability_contract_manifest() -> dict[str, object]:
    return {
        "schema_version": CAPABILITY_SCHEMA_VERSION,
        "minimum_cuda_series": MINIMUM_CUDA_SERIES,
        "canonical_cuda_series": CANONICAL_CUDA_SERIES,
        "canonical_ptx_isa": CANONICAL_PTX_ISA,
        "matcher_version": MATCHER_VERSION,
        "native_target": NATIVE_TARGET,
        "ptx_fallback_target": PTX_FALLBACK_TARGET,
        "platform_cells": PLATFORM_CELLS,
        "minimum_release_feature_ids": MINIMUM_RELEASE_FEATURE_IDS,
        "features": {
            feature_id: contract.as_manifest() for feature_id, contract in FEATURE_CONTRACTS.items()
        },
        "scoring_authority": {
            "global": "infraswe-scoring-v0.4",
            "deployability_formula": "100*C^0.45*U^0.30*M^0.25",
            "architecture_overlay": "diagnostic-cell-scorecard-only",
            "missing_evidence_policy": "unresolved-not-zero",
        },
    }

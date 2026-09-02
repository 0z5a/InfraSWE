from __future__ import annotations

from functools import lru_cache

from infraswe.models.system_paths import (
    MemoryObjectKind,
    SystemDraftProfile,
    SystemDraftProfileCatalog,
)

SYSTEM_PROFILE_CATALOG_VERSION = "system-path-profiles-v0.5.2-proposed.1"

COMMUNICATION_PROFILE_ORDER = (
    "nccl-collective-runtime-v1",
    "rccl-collective-runtime-v1",
    "nvshmem-one-sided-runtime-v1",
    "uccl-collective-runtime-v1",
    "ucx-transport-runtime-v1",
    "ucc-collective-integration-v1",
    "pytorch-processgroup-nccl-integration-v1",
    "vllm-communication-runtime-v1",
    "sglang-communication-runtime-v1",
    "megatron-collective-integration-v1",
)

MEMORY_PROFILE_ORDER = (
    "memory-tiering-offload-runtime-v1",
    "kv-cache-cpu-offload-v1",
    "weight-cpu-offload-v1",
    "training-state-cpu-offload-v1",
    "activation-cpu-offload-v1",
    "checkpoint-staging-cpu-offload-v1",
)

_COMMUNICATION_LAYERS = {
    "nccl-collective-runtime-v1": "collective-library",
    "rccl-collective-runtime-v1": "collective-library",
    "nvshmem-one-sided-runtime-v1": "one-sided-runtime",
    "uccl-collective-runtime-v1": "collective-library",
    "ucx-transport-runtime-v1": "transport-runtime",
    "ucc-collective-integration-v1": "collective-scheduling",
    "pytorch-processgroup-nccl-integration-v1": "framework-process-group",
    "vllm-communication-runtime-v1": "collective-scheduling",
    "sglang-communication-runtime-v1": "collective-scheduling",
    "megatron-collective-integration-v1": "collective-scheduling",
}

_MEMORY_PROFILE_DATA: dict[
    str,
    tuple[MemoryObjectKind, list[str], list[str], list[str]],
] = {
    "kv-cache-cpu-offload-v1": (
        "kv-cache",
        ["context-and-concurrency-capacity", "ttft-tpot", "slo-goodput"],
        ["request-isolation", "block-version-mapping", "decode-visibility"],
        ["request-cancel", "prefix-cache-reuse", "low-host-budget", "repeated-load-unload"],
    ),
    "weight-cpu-offload-v1": (
        "weight",
        ["model-fit", "cold-start", "steady-throughput", "expert-miss-stall"],
        ["immutable-version", "layer-or-expert-prefetch-order", "no-mixed-weight-version"],
        ["add-layer", "add-expert", "change-prefetch-window", "model-hot-reload"],
    ),
    "training-state-cpu-offload-v1": (
        "training-state",
        ["step-time", "tokens-per-second", "max-microbatch", "gpu-memory-relief"],
        ["step-version", "optimizer-update-order", "checkpoint-consistency"],
        ["change-microbatch", "change-accumulation-steps", "checkpoint-resume", "rank-restart"],
    ),
    "activation-cpu-offload-v1": (
        "activation",
        ["step-time", "max-batch-or-sequence", "backward-prefetch-stall"],
        ["backward-consumer-binding", "recompute-policy-explicit", "release-after-last-consumer"],
        [
            "change-sequence-length",
            "enable-recompute",
            "disable-recompute",
            "repeated-forward-backward",
        ],
    ),
    "checkpoint-staging-cpu-offload-v1": (
        "checkpoint-staging",
        ["checkpoint-pause", "durable-completion-time", "restart-time"],
        ["durability-after-flush", "manifest-version-binding", "partial-write-rejection"],
        ["checkpoint-resume", "partial-write", "flush-failure", "repeated-checkpoint"],
    ),
}

_OBJECT_PROFILE = {
    object_kind: profile_id for profile_id, (object_kind, _, _, _) in _MEMORY_PROFILE_DATA.items()
}


def _communication_profile(profile_id: str) -> SystemDraftProfile:
    return SystemDraftProfile(
        profile_id=profile_id,
        domain="distributed-communication",
        template_id="communication-path-integration-v1",
        sealable=True,
        layer=_COMMUNICATION_LAYERS[profile_id],
        anchor_plugin="communication-v1",
        performance_objectives=[
            "slo-goodput",
            "tail-latency",
            "progress-and-overlap",
            "rank-fairness",
            "resource-stability",
        ],
        correctness_invariants=[
            "correct-result",
            "collective-order-consistency",
            "structured-error-propagation",
            "no-silent-fallback",
            "bounded-lifecycle-resources",
        ],
        required_probes=[
            "concurrent-communicators",
            "message-size-boundaries",
            "repeated-init-destroy",
            "transport-or-provider-failure",
        ],
    )


def _memory_profile(profile_id: str) -> SystemDraftProfile:
    if profile_id == "memory-tiering-offload-runtime-v1":
        return SystemDraftProfile(
            profile_id=profile_id,
            domain="memory-tiering",
            template_id="memory-tier-integration-v1",
            sealable=False,
            anchor_plugin="memory-tier-v1",
            performance_objectives=["shared-service-residency-transfer-evidence"],
            correctness_invariants=[
                "residency-state-machine",
                "consumer-waits-for-ready-event",
                "teardown-quiescence",
            ],
            required_probes=["choose-concrete-offload-object-profile"],
        )
    object_kind, objectives, invariants, probes = _MEMORY_PROFILE_DATA[profile_id]
    return SystemDraftProfile(
        profile_id=profile_id,
        domain="memory-tiering",
        template_id="memory-tier-integration-v1",
        sealable=True,
        abstract_parent_profile="memory-tiering-offload-runtime-v1",
        object_kind=object_kind,
        anchor_plugin="memory-tier-v1",
        performance_objectives=objectives,
        correctness_invariants=invariants,
        required_probes=probes,
    )


@lru_cache(maxsize=1)
def build_system_profile_catalog() -> SystemDraftProfileCatalog:
    order = [*COMMUNICATION_PROFILE_ORDER, *MEMORY_PROFILE_ORDER]
    profiles = {
        profile.profile_id: profile
        for profile in [
            *(_communication_profile(profile_id) for profile_id in COMMUNICATION_PROFILE_ORDER),
            *(_memory_profile(profile_id) for profile_id in MEMORY_PROFILE_ORDER),
        ]
    }
    return SystemDraftProfileCatalog(
        catalog_version=SYSTEM_PROFILE_CATALOG_VERSION,
        profile_order=order,
        profiles=profiles,
    )


def select_memory_profile(object_kind: MemoryObjectKind) -> SystemDraftProfile:
    try:
        profile_id = _OBJECT_PROFILE[object_kind]
    except KeyError as error:
        raise ValueError(f"unsupported or ambiguous offload object kind: {object_kind}") from error
    return build_system_profile_catalog().profiles[profile_id]

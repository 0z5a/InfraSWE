from __future__ import annotations

from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

from infraswe.draft.candidate_registry import (
    infer_candidate_request,
    resolve_default_candidates,
)
from infraswe.draft.extended_default_templates import (
    ADDITIONAL_DEFAULT_PROJECT_TEMPLATES,
)
from infraswe.draft.lifecycle import canonical_sha256
from infraswe.models.draft import (
    DefaultDraftCatalog,
    DefaultDraftCatalogEntry,
    DefaultDraftProject,
    DefaultProjectContractArtifact,
    DraftAcceptanceContract,
    DraftBaseline,
    DraftBenchmarkLoop,
    DraftCandidate,
    DraftDeployment,
    DraftMetadata,
    DraftObjectiveBinding,
    DraftRetrieval,
    DraftScoringPolicy,
    DraftSpec,
    DraftTarget,
    EdgeEcosystemObjective,
    EdgeProfileObjective,
    ProjectObjectives,
    ProjectOwnership,
    TargetProjectProfile,
    VersionedArtifactRef,
)

CATALOG_VERSION = "default-projects-v0.5-proposed.2"
SOURCE_CAPTURE_DATE = "2026-09-02"
DEFAULT_PROJECT_ORDER: tuple[DefaultDraftProject, ...] = (
    "vllm",
    "sglang",
    "flash-attention",
    "flashinfer",
    "cutlass-cute",
    "liger-kernel",
    "deepgemm",
    "megatron-core",
    "torchtitan",
    "verl",
)
CONTRACT_KINDS = (
    "api-abi",
    "lifecycle",
    "build-test-matrix",
    "dependency-policy",
    "fallback-policy",
    "deployment-workload-portfolio",
    "performance-acceptance-targets",
    "maintainability-probes",
)


def _contract(requirement: str, *probes: str) -> dict[str, Any]:
    return {"requirements": [requirement], "probes": list(probes)}


_TEMPLATES: dict[DefaultDraftProject, dict[str, Any]] = {
    "vllm": {
        "catalog_profile": "vllm-kernel-integration-v1",
        "repository": "https://github.com/vllm-project/vllm",
        "revision": "40824284bcb2f50047a48307ed39ce441bb15b0b",
        "aliases": ["vllm", "vllm-project/vllm"],
        "sources": [
            "https://github.com/vllm-project/vllm/blob/"
            "40824284bcb2f50047a48307ed39ce441bb15b0b/docs/contributing/README.md",
            "https://github.com/vllm-project/vllm/blob/"
            "40824284bcb2f50047a48307ed39ce441bb15b0b/docs/design/custom_op.md",
            "https://github.com/vllm-project/vllm/blob/"
            "40824284bcb2f50047a48307ed39ce441bb15b0b/"
            "docs/contributing/deprecation_policy.md",
        ],
        "integration_points": [
            "vllm/kernels",
            "vllm/model_executor/layers",
            "vllm/model_executor/hw_agnostic/custom_op.py",
            "csrc",
            "tests/kernels",
        ],
        "component_ownership": {
            "kernel-and-custom-op": ["upstream-codeowner-review-required"],
            "engine-integration": ["upstream-codeowner-review-required"],
        },
        "required_cells": ["cuda-sm80", "cuda-sm89", "cuda-sm90"],
        "edge_profiles": ["cuda-sm100", "rocm", "cpu"],
        "contracts": {
            "api-abi": _contract(
                "Register schemas, implementations, tensor meta functions, and public API changes "
                "through supported vLLM/PyTorch integration and deprecation paths.",
                "operator-schema",
                "torch-library-opcheck",
                "public-api-regression",
            ),
            "lifecycle": _contract(
                "Preserve eager, compiled, warmup, CUDA-graph capture/replay, native fallback, "
                "and shutdown behavior.",
                "engine-init",
                "warmup",
                "cuda-graph-replay",
                "shutdown",
            ),
            "build-test-matrix": _contract(
                "Pass lint plus focused unit, kernel, integration, and affected-model tests; "
                "signature changes update registered schemas.",
                "lint",
                "pytest-focused",
                "kernel-correctness",
                "affected-model-smoke",
            ),
            "dependency-policy": _contract(
                "Declare dependencies in project manifests and capability-gate platform-specific "
                "imports and build requirements.",
                "clean-build",
                "dependency-diff",
                "cpu-import",
            ),
            "fallback-policy": _contract(
                "Unsupported or disabled custom ops use the declared native fallback or fail "
                "explicitly without changing semantics.",
                "custom-op-disabled",
                "unsupported-shape",
                "dispatch-trace",
            ),
            "deployment-workload-portfolio": _contract(
                "Cover prefill, decode, mixed batching, concurrency, representative tensor "
                "parallelism, and separate cold-start behavior.",
                "prefill",
                "decode",
                "mixed-batch",
                "tp-serving",
                "cold-start",
            ),
            "performance-acceptance-targets": _contract(
                "Report throughput, TTFT, TPOT, memory, and regressions against a pinned local "
                "baseline with exact workload and hardware identity.",
                "throughput",
                "ttft",
                "tpot",
                "peak-memory",
            ),
            "maintainability-probes": _contract(
                "Keep diffs concise, require an RFC for major architecture changes, document "
                "user behavior, and retain human review of AI-assisted code.",
                "diff-locality",
                "rfc-threshold",
                "docs-impact",
                "human-review-record",
            ),
        },
    },
    "sglang": {
        "catalog_profile": "sglang-runtime-kernel-v1",
        "repository": "https://github.com/sgl-project/sglang",
        "revision": "4c2c169e6ba15aee5408b250ce25ff7e73388d9b",
        "aliases": ["sglang", "sgl-project/sglang", "sgl-kernel", "sglang/srt"],
        "sources": [
            "https://github.com/sgl-project/sglang/blob/"
            "4c2c169e6ba15aee5408b250ce25ff7e73388d9b/"
            "docs/developer_guide/contribution_guide.md",
            "https://github.com/sgl-project/sglang/blob/"
            "4c2c169e6ba15aee5408b250ce25ff7e73388d9b/test/registered/README.md",
            "https://github.com/sgl-project/sglang/blob/"
            "4c2c169e6ba15aee5408b250ce25ff7e73388d9b/.github/MAINTAINER.md",
        ],
        "integration_points": [
            "python/sglang/srt",
            "python/sglang/kernels",
            "sgl-kernel",
            "test/registered/unit",
            "test/registered/kernels",
        ],
        "component_ownership": {
            "runtime": ["upstream-codeowner-review-required"],
            "kernel": ["upstream-kernel-maintainer-review-required"],
        },
        "required_cells": ["cuda-sm80", "cuda-sm89", "cuda-sm90"],
        "edge_profiles": ["cuda-sm100", "amd", "ascend"],
        "contracts": {
            "api-abi": _contract(
                "Preserve SRT call signatures, request semantics, registered server behavior, "
                "and project-owned kernel binding points.",
                "runtime-import",
                "request-contract",
                "kernel-binding",
                "server-api-smoke",
            ),
            "lifecycle": _contract(
                "Preserve server launch, scheduler, prefill/decode, graph capture, distributed "
                "execution, shutdown, and isolated hardware-specific paths.",
                "server-start",
                "prefill-decode",
                "cuda-graph",
                "distributed-smoke",
            ),
            "build-test-matrix": _contract(
                "Mirror changed SRT modules in registered unit tests and place CUDA, server, "
                "model, platform, accuracy, and benchmark coverage in designated suites.",
                "pre-commit",
                "registered-unit",
                "registered-kernel",
                "server-e2e",
            ),
            "dependency-policy": _contract(
                "Treat sglang and sglang-kernel as separately released packages and follow the "
                "kernel-source, package-release, runtime-version multi-PR lifecycle.",
                "package-boundary",
                "kernel-version-pin",
                "clean-install",
            ),
            "fallback-policy": _contract(
                "Unsupported hardware and workloads select a declared backend or fail explicitly "
                "without silently changing output semantics.",
                "backend-selection",
                "unsupported-shape",
                "missing-kernel-package",
            ),
            "deployment-workload-portfolio": _contract(
                "Cover server prefill/decode, mixed traffic, continuous batching, graph replay, "
                "and distributed serving using project benchmark tooling.",
                "prefill",
                "decode",
                "mixed-traffic",
                "continuous-batching",
                "tp-serving",
            ),
            "performance-acceptance-targets": _contract(
                "Compare project-local latency, throughput, memory, graph retention, and variance; "
                "never infer speed from the GSM8K accuracy sanity test.",
                "latency",
                "throughput",
                "memory",
                "graph-retention",
                "variance",
            ),
            "maintainability-probes": _contract(
                "Prefer new files for hardware components, keep the common path first, avoid "
                "drastic changes, and include fast focused tests and owner review.",
                "diff-locality",
                "hardware-isolation",
                "test-runtime",
                "owner-review",
            ),
        },
    },
    "flash-attention": {
        "catalog_profile": "flash-attention-kernel-v1",
        "repository": "https://github.com/Dao-AILab/flash-attention",
        "revision": "ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820",
        "aliases": [
            "flash-attention",
            "flashattention",
            "flash_attn",
            "dao-ailab/flash-attention",
        ],
        "sources": [
            "https://github.com/Dao-AILab/flash-attention/blob/"
            "ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/README.md",
            "https://github.com/Dao-AILab/flash-attention/blob/"
            "ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/setup.py",
            "https://github.com/Dao-AILab/flash-attention/tree/"
            "ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820/tests",
        ],
        "integration_points": ["flash_attn", "csrc", "hopper", "tests", "benchmarks"],
        "component_ownership": {
            "cuda-rocm-kernels": ["upstream-maintainer-review-required"],
            "python-interface": ["upstream-maintainer-review-required"],
        },
        "required_cells": ["cuda-sm80", "cuda-sm89", "cuda-sm90"],
        "edge_profiles": ["cuda-sm100", "rocm-cdna", "rocm-rdna"],
        "contracts": {
            "api-abi": _contract(
                "Preserve exact-attention forward/backward plus packed, varlen, causal, MQA/GQA, "
                "dtype, layout, device, and contiguity semantics.",
                "forward",
                "backward",
                "packed-varlen",
                "causal",
                "mqa-gqa",
            ),
            "lifecycle": _contract(
                "Source and wheel builds initialize backend submodules and preserve compile, "
                "import, first-call, repeated-call, and teardown behavior.",
                "source-build",
                "wheel-build",
                "import",
                "first-call",
                "repeat-call",
            ),
            "build-test-matrix": _contract(
                "Run the affected generation/backend suite and compare outputs and gradients with "
                "a high-precision reference across boundary shapes and dtypes.",
                "reference-forward",
                "reference-backward",
                "boundary-shapes",
                "dtype-matrix",
            ),
            "dependency-policy": _contract(
                "Respect declared PyTorch, toolkit, compiler, ninja, platform, and backend-scoped "
                "CUDA, CK, Triton-AMD, Hopper, and CuTeDSL dependencies.",
                "toolkit-version",
                "submodule-state",
                "backend-build-isolation",
            ),
            "fallback-policy": _contract(
                "Unsupported architectures, dtypes, dimensions, and feature combinations are "
                "explicit and never silently substitute different attention semantics.",
                "unsupported-arch",
                "unsupported-dtype",
                "head-dim-boundary",
            ),
            "deployment-workload-portfolio": _contract(
                "Cover forward/backward, causal/non-causal, varlen, long sequence, MQA/GQA, and "
                "representative training and inference shapes per hardware cell.",
                "training-fwd-bwd",
                "inference-fwd",
                "varlen",
                "long-sequence",
            ),
            "performance-acceptance-targets": _contract(
                "Report local-cell latency, memory, exactness retention, and compile time against "
                "the pinned implementation/reference.",
                "latency",
                "memory",
                "correctness-tolerance",
                "compile-time",
            ),
            "maintainability-probes": _contract(
                "Localize backend-generation changes, bind tests to the affected implementation "
                "family, and document supported feature and architecture changes.",
                "backend-locality",
                "test-affinity",
                "support-docs",
                "review-record",
            ),
        },
    },
    "flashinfer": {
        "catalog_profile": "flashinfer-kernel-library-v1",
        "repository": "https://github.com/flashinfer-ai/flashinfer",
        "revision": "9d0e6f82ffa23d4271c08e0e0d4fc638b6b707ea",
        "aliases": ["flashinfer", "flashinfer-ai/flashinfer"],
        "sources": [
            "https://github.com/flashinfer-ai/flashinfer/blob/"
            "9d0e6f82ffa23d4271c08e0e0d4fc638b6b707ea/CONTRIBUTING.md",
            "https://github.com/flashinfer-ai/flashinfer/blob/"
            "9d0e6f82ffa23d4271c08e0e0d4fc638b6b707ea/README.md",
            "https://github.com/flashinfer-ai/flashinfer/tree/"
            "9d0e6f82ffa23d4271c08e0e0d4fc638b6b707ea/docs",
        ],
        "integration_points": ["include", "csrc", "python", "tests", "benchmarks"],
        "component_ownership": {
            "kernel-definition": ["upstream-codeowner-review-required"],
            "framework-registration": ["upstream-codeowner-review-required"],
            "python-api": ["upstream-codeowner-review-required"],
        },
        "required_cells": ["cuda-sm80", "cuda-sm89", "cuda-sm90"],
        "edge_profiles": ["cuda-sm100", "cuda-sm120", "multi-gpu"],
        "contracts": {
            "api-abi": _contract(
                "Keep framework-agnostic kernels under include, PyTorch registration under csrc, "
                "user APIs under python, and released APIs backward-compatible.",
                "raw-pointer-kernel",
                "torch-registration",
                "python-api",
            ),
            "lifecycle": _contract(
                "Preserve AOT, JIT, cubin, JIT-cache, first-use compilation, CUDA graph, and "
                "torch.compile behavior with observable cache/module selection.",
                "aot-build",
                "jit-first-call",
                "cache-hit",
                "cuda-graph",
            ),
            "build-test-matrix": _contract(
                "Add Python unit tests and affected architecture lanes; optimization PRs include "
                "reproducible before/after results with GPU and exact problem sizes.",
                "public-ci",
                "gpu-unit",
                "architecture-matrix",
                "reproducible-benchmark",
            ),
            "dependency-policy": _contract(
                "Keep Torch headers out of include and declare compatible toolkit, cubin, "
                "JIT-cache, "
                "CUTLASS, package, and build-isolation dependencies.",
                "header-boundary",
                "clean-editable-install",
                "submodule-state",
            ),
            "fallback-policy": _contract(
                "Make unsupported backend/feature and missing precompiled/JIT artifact behavior "
                "explicit with a declared recovery or failure path.",
                "backend-selection",
                "missing-cubin",
                "empty-jit-cache",
            ),
            "deployment-workload-portfolio": _contract(
                "Cover prefill, decode, append, mixed batching, paged/ragged KV, GEMM, MoE, "
                "sampling, communication claims, CUDA graph, and torch.compile.",
                "prefill",
                "decode",
                "mixed-batch",
                "paged-kv",
                "moe",
            ),
            "performance-acceptance-targets": _contract(
                "Compare latency, throughput, memory, compile cost, and cache reuse against a "
                "pinned local baseline with GPU identity and exact sizes.",
                "latency",
                "throughput",
                "memory",
                "compile-cost",
                "cache-reuse",
            ),
            "maintainability-probes": _contract(
                "Reuse project utilities and synchronize kernel, registration, Python, tests, "
                "benchmarks, docs, and module manifests under human review.",
                "layer-boundary",
                "utility-reuse",
                "manifest-update",
                "human-review",
            ),
        },
    },
}
_TEMPLATES.update(ADDITIONAL_DEFAULT_PROJECT_TEMPLATES)


def _artifact(
    project: DefaultDraftProject, kind: str, template: dict[str, Any]
) -> DefaultProjectContractArtifact:
    contract = template["contracts"][kind]
    return DefaultProjectContractArtifact(
        project=project,
        catalog_profile=template["catalog_profile"],
        artifact_kind=kind,
        source_repository=template["repository"],
        source_revision=template["revision"],
        source_urls=template["sources"],
        requirements=contract["requirements"],
        probes=contract["probes"],
    )


def _artifact_ref(artifact: DefaultProjectContractArtifact) -> VersionedArtifactRef:
    return VersionedArtifactRef(
        id=f"{artifact.catalog_profile}:{artifact.artifact_kind}",
        sha256=canonical_sha256(artifact),
        path=f"builtin://{CATALOG_VERSION}/{artifact.project}/{artifact.artifact_kind}",
    )


@lru_cache(maxsize=1)
def build_default_catalog() -> DefaultDraftCatalog:
    entries: dict[DefaultDraftProject, DefaultDraftCatalogEntry] = {}
    for project in DEFAULT_PROJECT_ORDER:
        template = _TEMPLATES[project]
        artifacts = {kind: _artifact(project, kind, template) for kind in CONTRACT_KINDS}
        refs = {kind: _artifact_ref(artifact) for kind, artifact in artifacts.items()}
        profile = TargetProjectProfile(
            id=template["catalog_profile"],
            version=CATALOG_VERSION,
            status="proposed",
            repository=template["repository"],
            supported_revision_policy=(
                f"pinned:{template['revision']}; updates require a new "
                "human-reviewed profile version"
            ),
            ownership=ProjectOwnership(
                maintainers=["UNASSIGNED:infraswe-default-profile-maintainer"]
            ),
            component_ownership=template["component_ownership"],
            allowed_integration_points=template["integration_points"],
            api_abi_contract=refs["api-abi"],
            lifecycle_contract=refs["lifecycle"],
            build_test_matrix=refs["build-test-matrix"],
            dependency_policy=refs["dependency-policy"],
            fallback_policy=refs["fallback-policy"],
            deployment_workload_portfolio=refs["deployment-workload-portfolio"],
            performance_acceptance_targets=refs["performance-acceptance-targets"],
            maintainability_probes=refs["maintainability-probes"],
            project_objectives=ProjectObjectives(
                edge_ecosystem=EdgeEcosystemObjective(
                    owner="UNASSIGNED:profile-maintainer",
                    policy="roadmap",
                    profiles=[
                        EdgeProfileObjective(id=item, status="planned")
                        for item in template["edge_profiles"]
                    ],
                )
            ),
            scoring_template_id="project-fit-kernel-v0.5",
        )
        entries[project] = DefaultDraftCatalogEntry(
            project=project,
            aliases=template["aliases"],
            source_revision=template["revision"],
            profile=profile,
            artifacts=artifacts,
        )
    return DefaultDraftCatalog(
        catalog_version=CATALOG_VERSION,
        status="proposed",
        source_capture_date=SOURCE_CAPTURE_DATE,
        default_order=list(DEFAULT_PROJECT_ORDER),
        entries=entries,
    )


def select_default_project(
    *, target_hint: str | None, candidate: DraftCandidate
) -> tuple[DefaultDraftProject, str, list[str]]:
    haystack = " ".join([target_hint or "", *candidate.entrypoints]).casefold()
    catalog = build_default_catalog()
    matched = [
        project
        for project in DEFAULT_PROJECT_ORDER
        if any(alias.casefold() in haystack for alias in catalog.entries[project].aliases)
    ]
    if len(matched) == 1:
        return matched[0], "matched-explicit-project-alias", ["DEFAULT_TARGET_REPORTED"]
    if matched:
        return (
            matched[0],
            "ambiguous-alias-match-resolved-by-frozen-order",
            ["DEFAULT_TARGET_REPORTED", "DEFAULT_TARGET_ALIAS_AMBIGUOUS"],
        )
    return (
        DEFAULT_PROJECT_ORDER[0],
        "no-project-alias-match; selected-by-frozen-catalog-order",
        ["DEFAULT_TARGET_REPORTED", "DEFAULT_TARGET_SELECTED_BY_PRIORITY"],
    )


def build_default_draft(
    *,
    project: DefaultDraftProject,
    candidate: DraftCandidate,
    created_by: str,
) -> tuple[DraftSpec, TargetProjectProfile]:
    catalog = build_default_catalog()
    entry = catalog.entries[project]
    profile = entry.profile
    profile_sha256 = canonical_sha256(profile)
    repository_sha256 = canonical_sha256(
        {"repository": profile.repository, "git_revision": entry.source_revision}
    )
    refs = {kind: _artifact_ref(artifact) for kind, artifact in entry.artifacts.items()}
    acceptance_sha256 = canonical_sha256(
        {
            "catalog_version": catalog.catalog_version,
            "profile_sha256": profile_sha256,
            "artifacts": {kind: reference.sha256 for kind, reference in refs.items()},
        }
    )
    draft = DraftSpec(
        draft=DraftMetadata(
            id=f"default-{project}-draft-v05",
            revision=1,
            state="D3-contract-proposed",
            created_by=created_by,
        ),
        target=DraftTarget(
            mode="catalog",
            catalog_profile=profile.id,
            repository=profile.repository,
            revision=repository_sha256,
            project_profile_sha256=profile_sha256,
        ),
        candidate=candidate,
        default_candidates=resolve_default_candidates(
            infer_candidate_request(candidate, default_target_project=project)
        ),
        baseline=DraftBaseline(mode="target-head", revision=repository_sha256),
        deployment=DraftDeployment(
            workload_portfolio=refs["deployment-workload-portfolio"],
            required_cells=list(_TEMPLATES[project]["required_cells"]),
            optional_cells=list(_TEMPLATES[project]["edge_profiles"]),
            request_or_step_protocol=refs["lifecycle"],
        ),
        retrieval=DraftRetrieval(
            corpus_cutoff=datetime.fromisoformat(SOURCE_CAPTURE_DATE).replace(tzinfo=UTC),
            sources=[
                "target-code",
                "merged-prs",
                "rejected-prs",
                "review-comments",
                "ci-failures",
                "release-notes",
            ],
            precedent_set_sha256=canonical_sha256(
                {
                    "source_revision": entry.source_revision,
                    "source_urls": sorted(
                        {
                            url
                            for artifact in entry.artifacts.values()
                            for url in artifact.source_urls
                        }
                    ),
                    "status": "machine-proposed",
                }
            ),
        ),
        acceptance_contract=DraftAcceptanceContract(
            status="proposed",
            path=f"builtin://{catalog.catalog_version}/{project}/acceptance-contract",
            sha256=acceptance_sha256,
            probe_set_sha256=refs["maintainability-probes"].sha256,
            hidden_probe_policy_sha256=canonical_sha256(
                {"profile": profile.id, "policy": "sealed-probes-required-for-official"}
            ),
        ),
        project_objectives=DraftObjectiveBinding(
            edge_ecosystem_policy=profile.project_objectives.edge_ecosystem.policy,
            profile_set_sha256=canonical_sha256(
                [
                    item.model_dump(mode="json")
                    for item in profile.project_objectives.edge_ecosystem.profiles
                ]
            ),
        ),
        benchmark_loop=DraftBenchmarkLoop(
            official_replays=7,
            benchmark_budget_policy_id="draft-staged-budget-v0.5",
            evidence_policy_id="v0.4-evidence-ladder-plus-seal-v0.5",
        ),
        scoring=DraftScoringPolicy(
            formula_template_id=profile.scoring_template_id,
            project_season="default-catalog-2026q3",
        ),
    )
    return draft, profile

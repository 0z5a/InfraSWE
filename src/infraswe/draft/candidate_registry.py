from __future__ import annotations

from collections import OrderedDict
from functools import lru_cache
from threading import Lock

from infraswe.draft.lifecycle import canonical_sha256
from infraswe.draft.precompile import decide_precompile
from infraswe.models.candidates import (
    CandidateActivationAction,
    CandidateActivationPlan,
    CandidateBackend,
    CandidateBuildPolicy,
    CandidatePhase,
    CandidateSelectionRequest,
    CandidateSelectionTrace,
    CandidateSource,
    CandidateTimingGate,
    DefaultCandidateDefinition,
    DefaultCandidateRegistry,
    DefaultCandidateResolution,
    DefaultCandidateRule,
    OperatorFamily,
)
from infraswe.models.draft import DraftCandidate, DraftPrecompilePolicy

REGISTRY_VERSION = "default-candidates-v0.5-proposed.1"
SOURCE_CAPTURE_DATE = "2026-09-02"

# The registry is immutable, so its canonical digest is safe to memoize by object identity.
# Keeping the cache small avoids retaining arbitrary caller-provided registries indefinitely.
_REGISTRY_DIGEST_CACHE_MAXSIZE = 16
_REGISTRY_DIGEST_CACHE: OrderedDict[int, tuple[DefaultCandidateRegistry, str]] = OrderedDict()
_REGISTRY_DIGEST_CACHE_LOCK = Lock()

_PINS = {
    "flash_attention": (
        "https://github.com/Dao-AILab/flash-attention",
        "ce088ab9ce0fc0434dcd8afa0a791da9fcc3a820",
    ),
    "flashinfer": (
        "https://github.com/flashinfer-ai/flashinfer",
        "9d0e6f82ffa23d4271c08e0e0d4fc638b6b707ea",
    ),
    "vllm": (
        "https://github.com/vllm-project/vllm",
        "cdefd9d4997f00da72dc6245cc60678b50761b7e",
    ),
    "sglang": (
        "https://github.com/sgl-project/sglang",
        "bb3e3cbceb510b05109d3dfdcbbd07a1a2905314",
    ),
    "cutlass_cute": (
        "https://github.com/NVIDIA/cutlass",
        "dc45f979ae336a235da1676b311f35efeb30149a",
    ),
    "liger_kernel": (
        "https://github.com/linkedin/Liger-Kernel",
        "e6a81bb0c34f31ca7806d0c2b72f6d66b0542694",
    ),
    "deepgemm": (
        "https://github.com/deepseek-ai/DeepGEMM",
        "559d79fb6994a58b8a15b4b93bf13ccc16edf247",
    ),
    "transformer_engine": (
        "https://github.com/NVIDIA/TransformerEngine",
        "f307af80275d473b4177415552f492f4797a1e39",
    ),
    "megatron_core": (
        "https://github.com/NVIDIA/Megatron-LM",
        "3c04d2bd2255c9652a687c3d5a5b9636467696db",
    ),
    "torchtitan": (
        "https://github.com/pytorch/torchtitan",
        "496b11d43860bb8d27b54568c76db6310ae7f55e",
    ),
    "verl": (
        "https://github.com/volcengine/verl",
        "c2429f29a25d573f63d9bcc29e7ceb690817dce9",
    ),
    "nccl": (
        "https://github.com/NVIDIA/nccl",
        "fd168324a3dc0c9080fd4881b6c7f4bb252a95a2",
    ),
    "deepep": (
        "https://github.com/deepseek-ai/DeepEP",
        "01dc3aaac82068020353dce2c302e38153c0bfaa",
    ),
    "tritonbench": (
        "https://github.com/pytorch-labs/tritonbench",
        "e5c687f5ec4c3ebb5e64334a64db37b29d399f15",
    ),
    "torchao": (
        "https://github.com/pytorch/ao",
        "86d5d96ebb2d8be1bc13e3a0f5334130e5ad3ad6",
    ),
    "tensorrt_llm": (
        "https://github.com/NVIDIA/TensorRT-LLM",
        "5fb68830c919b8dbcc07cb1095f5e7d369509919",
    ),
    "fbgemm_gpu": (
        "https://github.com/pytorch/FBGEMM",
        "f51f3f45b12d3b4d3e7f5233d52a50cf9dd46adc",
    ),
    "nvshmem": (
        "https://github.com/NVIDIA/nvshmem",
        "b0d9d3dc08fc3ee0840fdb6f3a2c11932d85a2e2",
    ),
    "aiter": (
        "https://github.com/ROCm/aiter",
        "27a85a1f6413a6a26195e089a2311be78245ff0c",
    ),
    "composable_kernel": (
        "https://github.com/ROCm/composable_kernel",
        "798ec4fb2baab96f35a4fbe941645983c1de68a5",
    ),
    "kernelbench": (
        "https://github.com/ScalingIntelligence/KernelBench",
        "423217d9fda91e0c2d67e4a43bf62f96f6d104f1",
    ),
    "llama_cpp": (
        "https://github.com/ggml-org/llama.cpp",
        "c845263f8b7d60113e213a3bd2d5cc6472ccf204",
    ),
    "mlc_llm": (
        "https://github.com/mlc-ai/mlc-llm",
        "9fa644f54b04983adea4d0168f49fc6af4a893ba",
    ),
    "executorch": (
        "https://github.com/pytorch/executorch",
        "1cfcdf8022325d86d7b745ff624f1a83c4d52d60",
    ),
    "xformers": (
        "https://github.com/facebookresearch/xformers",
        "029779dabdf19ce75dcb6c4a648535136d990639",
    ),
}


def _git_source(candidate_id: str) -> CandidateSource:
    repository, revision = _PINS[candidate_id]
    return CandidateSource(kind="pinned-git", repository=repository, revision=revision)


def _build(mode: str, candidate_id: str) -> CandidateBuildPolicy:
    adapter = f"{candidate_id}-adapter-v1" if mode in {"adapter-aot", "adapter-jit"} else None
    return CandidateBuildPolicy(compilation_mode=mode, adapter_id=adapter)


def _git_candidate(
    candidate_id: str,
    display_name: str,
    roles: list[str],
    tier: str,
    families: list[str],
    phases: list[str],
    backends: list[str],
    build_mode: str,
) -> DefaultCandidateDefinition:
    return DefaultCandidateDefinition(
        id=candidate_id,
        display_name=display_name,
        roles=roles,
        tier=tier,
        operator_families=families,
        phases=phases,
        backends=backends,
        source=_git_source(candidate_id),
        build=_build(build_mode, candidate_id),
        license_status="declared",
    )


def _builtin(
    candidate_id: str,
    display_name: str,
    role: str,
    families: list[str],
    *,
    source_kind: str = "runtime-builtin",
) -> DefaultCandidateDefinition:
    return DefaultCandidateDefinition(
        id=candidate_id,
        display_name=display_name,
        roles=[role],
        tier="P0",
        operator_families=families,
        phases=["generic", "training", "inference", "communication"],
        backends=["generic", "cuda", "rocm", "triton"],
        source=CandidateSource(kind=source_kind),
        build=_build(
            "environment-provided" if source_kind == "runtime-builtin" else "none", candidate_id
        ),
        license_status="runtime-provided" if source_kind == "runtime-builtin" else "draft-owned",
    )


def _rule(
    order: int,
    family: str,
    phases: list[str],
    backends: list[str],
    *,
    oracles: list[str],
    primary_peer: str,
    secondary_peers: list[str],
    hosts: list[str],
    workloads: list[str],
    coverage: list[str],
    tests: list[str],
) -> DefaultCandidateRule:
    return DefaultCandidateRule(
        id=f"{order:02d}-{family}-{'-'.join(backends)}",
        order=order,
        operator_family=family,
        phases=phases,
        backends=backends,
        oracles=oracles,
        primary_peer_impl=primary_peer,
        secondary_peer_impls=secondary_peers,
        hosts=hosts,
        workload_sources=workloads,
        coverage_targets=coverage,
        required_tests=tests,
    )


@lru_cache(maxsize=1)
def build_default_candidate_registry() -> DefaultCandidateRegistry:
    all_families: list[OperatorFamily] = [
        "generic",
        "attention-training",
        "attention-inference-prefill",
        "attention-inference-decode",
        "paged-attention",
        "dense-gemm",
        "grouped-moe-gemm",
        "training-fused-ops",
        "quantization",
        "communication-collective",
        "moe-dispatch-combine",
        "gpu-initiated-communication",
    ]
    candidates = [
        _builtin("torch_eager", "PyTorch eager", "oracle", all_families),
        _builtin(
            "project_original",
            "Project original",
            "oracle",
            all_families,
            source_kind="draft-relative",
        ),
        _builtin(
            "python_reference",
            "Minimal Python reference",
            "oracle",
            all_families,
            source_kind="draft-relative",
        ),
        _builtin("cublaslt", "cuBLASLt", "oracle", ["dense-gemm", "grouped-moe-gemm"]),
        _builtin("nccl_runtime", "NCCL runtime oracle", "oracle", ["communication-collective"]),
        _builtin(
            "reference_scatter_gather",
            "Reference scatter/gather",
            "oracle",
            ["moe-dispatch-combine"],
            source_kind="draft-relative",
        ),
        _builtin(
            "host_native",
            "Host-native trace",
            "workload-source",
            all_families,
            source_kind="draft-relative",
        ),
        _builtin(
            "synthetic_boundary",
            "Synthetic boundary generator",
            "workload-source",
            all_families,
            source_kind="draft-relative",
        ),
        _builtin(
            "user_custom",
            "User custom workload",
            "workload-source",
            all_families,
            source_kind="draft-relative",
        ),
        _builtin("cuda_datacenter", "CUDA datacenter", "coverage-target", all_families),
        _builtin("rocm_coverage", "ROCm", "coverage-target", all_families),
        _builtin("training_coverage", "Training", "coverage-target", all_families),
        _builtin("online_inference", "Online inference", "coverage-target", all_families),
        _builtin("edge_coverage", "Edge inference", "coverage-target", all_families),
        _git_candidate(
            "flash_attention",
            "FlashAttention",
            ["peer-impl"],
            "P0",
            ["attention-training", "attention-inference-prefill"],
            ["training", "inference"],
            ["cuda"],
            "adapter-aot",
        ),
        _git_candidate(
            "flashinfer",
            "FlashInfer",
            ["peer-impl"],
            "P0",
            [
                "attention-inference-prefill",
                "attention-inference-decode",
                "paged-attention",
                "quantization",
            ],
            ["inference"],
            ["cuda"],
            "adapter-jit",
        ),
        _git_candidate(
            "vllm",
            "vLLM",
            ["host-project", "peer-impl"],
            "P0",
            [
                "generic",
                "attention-inference-prefill",
                "attention-inference-decode",
                "paged-attention",
                "grouped-moe-gemm",
                "quantization",
                "communication-collective",
                "moe-dispatch-combine",
                "gpu-initiated-communication",
            ],
            ["generic", "inference", "communication"],
            ["cuda", "rocm"],
            "host-owned",
        ),
        _git_candidate(
            "sglang",
            "SGLang",
            ["host-project", "peer-impl"],
            "P0",
            [
                "generic",
                "attention-inference-prefill",
                "attention-inference-decode",
                "paged-attention",
                "grouped-moe-gemm",
                "quantization",
                "communication-collective",
                "moe-dispatch-combine",
            ],
            ["generic", "inference", "communication"],
            ["cuda", "rocm"],
            "host-owned",
        ),
        _git_candidate(
            "cutlass_cute",
            "CUTLASS / CuTe",
            ["peer-impl"],
            "P0",
            ["dense-gemm", "grouped-moe-gemm"],
            ["training", "inference"],
            ["cuda"],
            "adapter-aot",
        ),
        _git_candidate(
            "liger_kernel",
            "Liger-Kernel",
            ["peer-impl"],
            "P0",
            ["training-fused-ops"],
            ["training"],
            ["cuda", "triton"],
            "adapter-jit",
        ),
        _git_candidate(
            "deepgemm",
            "DeepGEMM",
            ["peer-impl"],
            "P1",
            ["dense-gemm", "grouped-moe-gemm"],
            ["training", "inference"],
            ["cuda"],
            "adapter-jit",
        ),
        _git_candidate(
            "transformer_engine",
            "Transformer Engine",
            ["peer-impl"],
            "P1",
            ["attention-training", "dense-gemm", "training-fused-ops", "quantization"],
            ["training"],
            ["cuda"],
            "adapter-aot",
        ),
        _git_candidate(
            "megatron_core",
            "Megatron-Core",
            ["host-project"],
            "P1",
            [
                "generic",
                "attention-training",
                "grouped-moe-gemm",
                "training-fused-ops",
                "quantization",
                "communication-collective",
                "moe-dispatch-combine",
                "gpu-initiated-communication",
            ],
            ["generic", "training", "communication"],
            ["cuda"],
            "host-owned",
        ),
        _git_candidate(
            "torchtitan",
            "TorchTitan",
            ["host-project"],
            "P1",
            ["generic", "attention-training", "training-fused-ops", "quantization"],
            ["generic", "training"],
            ["cuda", "triton"],
            "host-owned",
        ),
        _git_candidate(
            "verl",
            "verl",
            ["host-project"],
            "P1",
            ["training-fused-ops", "generic"],
            ["generic", "training"],
            ["cuda", "triton"],
            "host-owned",
        ),
        _git_candidate(
            "nccl",
            "NCCL",
            ["peer-impl"],
            "P1",
            ["communication-collective"],
            ["communication"],
            ["cuda"],
            "environment-provided",
        ),
        _git_candidate(
            "deepep",
            "DeepEP",
            ["peer-impl"],
            "P1",
            ["moe-dispatch-combine"],
            ["communication", "inference"],
            ["cuda"],
            "adapter-aot",
        ),
        _git_candidate(
            "tritonbench",
            "TritonBench",
            ["workload-source"],
            "P1",
            all_families,
            ["training", "inference"],
            ["cuda", "rocm", "triton"],
            "none",
        ),
        _git_candidate(
            "torchao",
            "TorchAO",
            ["peer-impl"],
            "P1",
            ["quantization"],
            ["training", "inference"],
            ["cuda", "rocm"],
            "adapter-aot",
        ),
        _git_candidate(
            "tensorrt_llm",
            "TensorRT-LLM",
            ["host-project"],
            "P2",
            [
                "attention-inference-prefill",
                "attention-inference-decode",
                "paged-attention",
                "dense-gemm",
                "grouped-moe-gemm",
                "quantization",
            ],
            ["inference"],
            ["cuda"],
            "host-owned",
        ),
        _git_candidate(
            "fbgemm_gpu",
            "FBGEMM_GPU",
            ["peer-impl"],
            "P2",
            ["quantization"],
            ["training", "inference"],
            ["cuda"],
            "adapter-aot",
        ),
        _git_candidate(
            "nvshmem",
            "NVSHMEM",
            ["peer-impl"],
            "P2",
            ["gpu-initiated-communication"],
            ["communication"],
            ["cuda"],
            "adapter-aot",
        ),
        _git_candidate(
            "aiter",
            "AITER",
            ["peer-impl"],
            "P2",
            all_families,
            ["training", "inference", "communication"],
            ["rocm"],
            "adapter-aot",
        ),
        _git_candidate(
            "composable_kernel",
            "Composable Kernel",
            ["peer-impl"],
            "P2",
            [
                "attention-training",
                "attention-inference-prefill",
                "dense-gemm",
                "grouped-moe-gemm",
                "quantization",
            ],
            ["training", "inference"],
            ["rocm"],
            "adapter-aot",
        ),
        _git_candidate(
            "kernelbench",
            "KernelBench",
            ["workload-source"],
            "P2",
            all_families,
            ["training", "inference"],
            ["cuda", "triton"],
            "none",
        ),
        _git_candidate(
            "llama_cpp",
            "llama.cpp",
            ["host-project", "coverage-target"],
            "P2",
            ["generic", "attention-inference-decode", "dense-gemm", "quantization"],
            ["inference"],
            ["generic", "cuda"],
            "host-owned",
        ),
        _git_candidate(
            "mlc_llm",
            "MLC-LLM",
            ["host-project", "coverage-target"],
            "P2",
            ["generic", "attention-inference-decode", "dense-gemm", "quantization"],
            ["inference"],
            ["generic", "cuda"],
            "host-owned",
        ),
        _git_candidate(
            "executorch",
            "ExecuTorch",
            ["host-project", "coverage-target"],
            "P2",
            ["generic", "dense-gemm", "quantization"],
            ["inference"],
            ["generic"],
            "host-owned",
        ),
        _git_candidate(
            "xformers",
            "xFormers",
            ["peer-impl"],
            "P2",
            ["attention-training", "attention-inference-prefill"],
            ["training", "inference"],
            ["cuda", "triton"],
            "adapter-aot",
        ),
    ]
    by_id = {item.id: item for item in candidates}
    common_workloads = ["host_native", "synthetic_boundary", "tritonbench"]
    rules = [
        _rule(
            1,
            "attention-training",
            ["training"],
            ["cuda", "triton"],
            oracles=["torch_eager", "project_original"],
            primary_peer="flash_attention",
            secondary_peers=["transformer_engine", "xformers"],
            hosts=["torchtitan", "megatron_core"],
            workloads=common_workloads,
            coverage=["cuda_datacenter", "training_coverage"],
            tests=[
                "forward-correctness",
                "backward-correctness",
                "gradient-error",
                "long-sequence",
                "compile-cache-reuse",
            ],
        ),
        _rule(
            2,
            "attention-inference-prefill",
            ["inference"],
            ["cuda"],
            oracles=["torch_eager", "project_original"],
            primary_peer="flashinfer",
            secondary_peers=["flash_attention", "vllm"],
            hosts=["vllm", "sglang", "tensorrt_llm"],
            workloads=common_workloads,
            coverage=["cuda_datacenter", "online_inference"],
            tests=[
                "prefill-correctness",
                "varlen",
                "paged-kv",
                "graph-capture",
                "concurrent-streams",
            ],
        ),
        _rule(
            3,
            "attention-inference-decode",
            ["inference"],
            ["cuda"],
            oracles=["torch_eager", "project_original"],
            primary_peer="flashinfer",
            secondary_peers=["vllm", "sglang"],
            hosts=["vllm", "sglang"],
            workloads=common_workloads,
            coverage=["cuda_datacenter", "online_inference"],
            tests=[
                "decode-correctness",
                "paged-kv",
                "batch-variation",
                "graph-capture",
                "concurrent-streams",
            ],
        ),
        _rule(
            4,
            "paged-attention",
            ["inference"],
            ["cuda"],
            oracles=["torch_eager", "project_original"],
            primary_peer="flashinfer",
            secondary_peers=["vllm"],
            hosts=["vllm", "sglang"],
            workloads=common_workloads,
            coverage=["cuda_datacenter", "online_inference"],
            tests=[
                "page-boundary",
                "ragged-kv",
                "decode-correctness",
                "graph-capture",
                "memory-peak",
            ],
        ),
        _rule(
            5,
            "dense-gemm",
            ["training", "inference", "generic"],
            ["cuda"],
            oracles=["torch_eager", "cublaslt"],
            primary_peer="cutlass_cute",
            secondary_peers=["deepgemm"],
            hosts=["vllm", "sglang", "megatron_core"],
            workloads=common_workloads,
            coverage=["cuda_datacenter"],
            tests=[
                "numerical-correctness",
                "tiny-m",
                "skinny-shape",
                "non-aligned-k",
                "concurrent-streams",
            ],
        ),
        _rule(
            6,
            "grouped-moe-gemm",
            ["training", "inference"],
            ["cuda"],
            oracles=["torch_eager", "project_original"],
            primary_peer="deepgemm",
            secondary_peers=["cutlass_cute"],
            hosts=["vllm", "sglang", "megatron_core"],
            workloads=common_workloads,
            coverage=["cuda_datacenter"],
            tests=[
                "balanced-experts",
                "long-tail-experts",
                "tiny-m",
                "non-aligned-k",
                "concurrent-streams",
            ],
        ),
        _rule(
            7,
            "training-fused-ops",
            ["training"],
            ["cuda", "triton"],
            oracles=["torch_eager", "project_original"],
            primary_peer="liger_kernel",
            secondary_peers=["transformer_engine"],
            hosts=["torchtitan", "megatron_core", "verl"],
            workloads=common_workloads,
            coverage=["cuda_datacenter", "training_coverage"],
            tests=[
                "forward-correctness",
                "backward-correctness",
                "gradient-error",
                "memory-peak",
                "fsdp2-compatibility",
            ],
        ),
        _rule(
            8,
            "quantization",
            ["training", "inference"],
            ["cuda"],
            oracles=["torch_eager", "project_original"],
            primary_peer="torchao",
            secondary_peers=["fbgemm_gpu", "transformer_engine", "vllm", "sglang"],
            hosts=["vllm", "sglang", "megatron_core"],
            workloads=common_workloads,
            coverage=["cuda_datacenter"],
            tests=[
                "quant-dequant-correctness",
                "scale-layout",
                "pack-unpack",
                "batch-variation",
                "memory-lifecycle",
            ],
        ),
        _rule(
            9,
            "communication-collective",
            ["communication"],
            ["cuda"],
            oracles=["nccl_runtime"],
            primary_peer="nccl",
            secondary_peers=[],
            hosts=["megatron_core", "vllm", "sglang"],
            workloads=["host_native", "synthetic_boundary"],
            coverage=["cuda_datacenter"],
            tests=[
                "rank-branch-consistency",
                "deadlock-soak",
                "stream-overlap",
                "backpressure",
                "multinode-stability",
            ],
        ),
        _rule(
            10,
            "moe-dispatch-combine",
            ["communication", "inference"],
            ["cuda"],
            oracles=["reference_scatter_gather", "project_original"],
            primary_peer="deepep",
            secondary_peers=["nccl"],
            hosts=["vllm", "sglang", "megatron_core"],
            workloads=["host_native", "synthetic_boundary"],
            coverage=["cuda_datacenter"],
            tests=[
                "dispatch-correctness",
                "variable-size",
                "long-tail-experts",
                "deadlock-soak",
                "backpressure",
            ],
        ),
        _rule(
            11,
            "gpu-initiated-communication",
            ["communication"],
            ["cuda"],
            oracles=["project_original", "nccl_runtime"],
            primary_peer="nvshmem",
            secondary_peers=["nccl"],
            hosts=["megatron_core", "vllm"],
            workloads=["host_native", "synthetic_boundary"],
            coverage=["cuda_datacenter"],
            tests=[
                "ordering",
                "stream-overlap",
                "deadlock-soak",
                "backpressure",
                "multinode-stability",
            ],
        ),
        _rule(
            12,
            "generic",
            ["generic", "training", "inference", "communication"],
            ["rocm"],
            oracles=["torch_eager", "project_original"],
            primary_peer="aiter",
            secondary_peers=["composable_kernel"],
            hosts=["vllm", "sglang"],
            workloads=common_workloads,
            coverage=["rocm_coverage"],
            tests=[
                "numerical-correctness",
                "unsupported-shape",
                "profiler-evidence",
                "concurrent-stability",
                "memory-lifecycle",
            ],
        ),
        _rule(
            13,
            "generic",
            ["generic", "inference", "training", "communication"],
            ["generic", "cuda", "triton"],
            oracles=["torch_eager", "project_original"],
            primary_peer="vllm",
            secondary_peers=["sglang"],
            hosts=["vllm", "sglang", "torchtitan"],
            workloads=common_workloads,
            coverage=["cuda_datacenter"],
            tests=[
                "numerical-correctness",
                "synthetic-boundary",
                "host-integration",
                "concurrent-stability",
                "memory-lifecycle",
            ],
        ),
    ]
    return DefaultCandidateRegistry(
        registry_version=REGISTRY_VERSION,
        source_capture_date=SOURCE_CAPTURE_DATE,
        candidates=by_id,
        rules=rules,
        fallback_chains={
            "generic": ["project_original", "torch_eager", "python_reference"],
            "attention": [
                "flashinfer",
                "flash_attention",
                "vllm",
                "torch_eager",
                "python_reference",
            ],
            "gemm": ["deepgemm", "cutlass_cute", "cublaslt", "torch_eager", "python_reference"],
            "training-fused-ops": [
                "liger_kernel",
                "transformer_engine",
                "project_original",
                "torch_eager",
            ],
            "communication": ["deepep", "nvshmem", "nccl", "project_original", "python_reference"],
        },
    )


def _candidate_compatible(
    registry: DefaultCandidateRegistry,
    candidate_id: str,
    request: CandidateSelectionRequest,
) -> bool:
    candidate = registry.candidates[candidate_id]
    phase_ok = request.phase in candidate.phases or "generic" in candidate.phases
    backend_ok = request.backend in candidate.backends or "generic" in candidate.backends
    family_ok = (
        request.operator_family in candidate.operator_families
        or "generic" in candidate.operator_families
    )
    return phase_ok and backend_ok and family_ok


def _registry_sha256(registry: DefaultCandidateRegistry) -> str:
    """Return a stable registry digest without reserializing it on every hot-path lookup."""

    key = id(registry)
    with _REGISTRY_DIGEST_CACHE_LOCK:
        cached = _REGISTRY_DIGEST_CACHE.get(key)
        if cached is not None and cached[0] is registry:
            _REGISTRY_DIGEST_CACHE.move_to_end(key)
            return cached[1]
        digest = canonical_sha256(registry)
        _REGISTRY_DIGEST_CACHE[key] = (registry, digest)
        _REGISTRY_DIGEST_CACHE.move_to_end(key)
        while len(_REGISTRY_DIGEST_CACHE) > _REGISTRY_DIGEST_CACHE_MAXSIZE:
            _REGISTRY_DIGEST_CACHE.popitem(last=False)
        return digest


def resolve_default_candidates(
    request: CandidateSelectionRequest,
    *,
    registry: DefaultCandidateRegistry | None = None,
) -> DefaultCandidateResolution:
    registry = registry or build_default_candidate_registry()
    trace: list[CandidateSelectionTrace] = []
    matched: DefaultCandidateRule | None = None
    for rule in registry.rules:
        family_ok = rule.operator_family in {request.operator_family, "generic"}
        phase_ok = request.phase in rule.phases or "generic" in rule.phases
        backend_ok = request.backend in rule.backends or "generic" in rule.backends
        is_match = family_ok and phase_ok and backend_ok
        trace.append(
            CandidateSelectionTrace(
                step=len(trace) + 1,
                question=f"Does frozen rule {rule.id} match family/phase/backend?",
                result="matched" if is_match else "not-matched",
                explanation=(
                    f"family={family_ok}, phase={phase_ok}, backend={backend_ok}; "
                    "rules are evaluated in ascending order without a score"
                ),
            )
        )
        if is_match:
            matched = rule
            break
    if matched is None:
        raise ValueError("default candidate registry has no matching fallback rule")

    primary_host = matched.hosts[0]
    if request.requested_primary_host is not None:
        requested = registry.candidates.get(request.requested_primary_host)
        if (
            requested
            and "host-project" in requested.roles
            and requested.default_eligible
            and _candidate_compatible(registry, requested.id, request)
        ):
            primary_host = requested.id
            trace.append(
                CandidateSelectionTrace(
                    step=len(trace) + 1,
                    question="Is the requested primary host role-compatible?",
                    result="preserved",
                    explanation=f"Preserved explicit compatible host {requested.id}.",
                )
            )
        else:
            trace.append(
                CandidateSelectionTrace(
                    step=len(trace) + 1,
                    question="Is the requested primary host role-compatible?",
                    result="fallback",
                    explanation=(
                        f"Requested host {request.requested_primary_host} is unavailable or "
                        f"incompatible; used frozen default {primary_host}."
                    ),
                )
            )
    secondary_hosts = [item for item in matched.hosts if item != primary_host]
    trace.append(
        CandidateSelectionTrace(
            step=len(trace) + 1,
            question="Can selection finish without importing or compiling any candidate?",
            result="selected",
            explanation=(
                "Returned pinned metadata only. Compilation is deferred to explicit activation "
                "of a selected candidate."
            ),
        )
    )
    return DefaultCandidateResolution(
        registry_sha256=_registry_sha256(registry),
        request=request,
        matched_rule_id=matched.id,
        oracles=matched.oracles,
        primary_peer_impl=matched.primary_peer_impl,
        secondary_peer_impls=matched.secondary_peer_impls,
        primary_host=primary_host,
        secondary_hosts=secondary_hosts,
        workload_sources=matched.workload_sources,
        coverage_targets=matched.coverage_targets,
        required_tests=matched.required_tests,
        trace=trace,
    )


def infer_candidate_request(
    candidate: DraftCandidate,
    *,
    default_target_project: str,
) -> CandidateSelectionRequest:
    joined = " ".join(candidate.entrypoints).lower()
    family: OperatorFamily = candidate.operator_family
    if family == "generic":
        project_default_families: dict[str, OperatorFamily] = {
            "flash-attention": "attention-training",
            "flashinfer": "attention-inference-decode",
            "cutlass-cute": "dense-gemm",
            "liger-kernel": "training-fused-ops",
            "deepgemm": "grouped-moe-gemm",
            "megatron-core": "grouped-moe-gemm",
            "torchtitan": "training-fused-ops",
            "verl": "training-fused-ops",
        }
        if default_target_project in project_default_families:
            family = project_default_families[default_target_project]
        elif "moe" in joined and any(token in joined for token in ("gemm", "matmul")):
            family = "grouped-moe-gemm"
        elif "dispatch" in joined or "combine" in joined:
            family = "moe-dispatch-combine"
        elif any(token in joined for token in ("allreduce", "all_reduce", "collective")):
            family = "communication-collective"
        elif "quant" in joined:
            family = "quantization"
        elif any(token in joined for token in ("loss", "rmsnorm", "layernorm", "rope")):
            family = "training-fused-ops"
        elif "attention" in joined or "flashinfer" in joined:
            if "decode" in joined or default_target_project == "flashinfer":
                family = "attention-inference-decode"
            elif default_target_project == "flash-attention":
                family = "attention-training"
            else:
                family = "attention-inference-prefill"

    phase: CandidatePhase = candidate.phase
    if phase == "generic":
        if default_target_project in {
            "liger-kernel",
            "megatron-core",
            "torchtitan",
            "verl",
        } or family in {"attention-training", "training-fused-ops"}:
            phase = "training"
        elif family in {
            "communication-collective",
            "moe-dispatch-combine",
            "gpu-initiated-communication",
        }:
            phase = "communication"
        elif family != "generic":
            phase = "inference"

    backend: CandidateBackend = candidate.backend
    if backend == "generic":
        backend = "triton" if candidate.implementation_kind == "triton-pure" else "cuda"
    host_map = {
        "vllm": "vllm",
        "sglang": "sglang",
        "flash-attention": "torchtitan",
        "flashinfer": "vllm",
        "cutlass-cute": "vllm",
        "liger-kernel": "torchtitan",
        "deepgemm": "vllm",
        "megatron-core": "megatron_core",
        "torchtitan": "torchtitan",
        "verl": "verl",
    }
    return CandidateSelectionRequest(
        operator_family=family,
        phase=phase,
        backend=backend,
        requested_primary_host=candidate.primary_host_candidate or host_map[default_target_project],
    )


def plan_candidate_activation(
    resolution: DefaultCandidateResolution,
    *,
    activated_candidate_ids: list[str] | None = None,
    compilation_required: dict[str, bool] | None = None,
    cache_hits: dict[str, bool] | None = None,
    precompile_policy: DraftPrecompilePolicy | None = None,
    registry: DefaultCandidateRegistry | None = None,
) -> CandidateActivationPlan:
    registry = registry or build_default_candidate_registry()
    if resolution.registry_sha256 != _registry_sha256(registry):
        raise ValueError("candidate resolution and activation registry digests do not match")
    activated = activated_candidate_ids or [resolution.primary_peer_impl]
    if len(activated) != 1:
        raise ValueError(
            "default candidate activation requires exactly one peer; "
            "compare additional peers in separate candidate runs"
        )
    if len(activated) != len(set(activated)):
        raise ValueError("activated_candidate_ids cannot contain duplicates")
    selected = resolution.selected_candidate_ids()
    outside = set(activated) - selected
    if outside:
        raise ValueError(
            "cannot activate unselected default candidates: " + ", ".join(sorted(outside))
        )
    non_peers = [
        candidate_id
        for candidate_id in activated
        if "peer-impl" not in registry.candidates[candidate_id].roles
    ]
    if non_peers:
        raise ValueError(
            "default activation is limited to peer implementations: " + ", ".join(sorted(non_peers))
        )
    requirements = compilation_required or {}
    hits = cache_hits or {}
    undeclared = (set(requirements) | set(hits)) - set(activated)
    if undeclared:
        raise ValueError(
            "compile/cache inputs are only allowed for explicitly activated candidates: "
            + ", ".join(sorted(undeclared))
        )
    policy = precompile_policy or DraftPrecompilePolicy()
    actions: list[CandidateActivationAction] = []
    for candidate_id in activated:
        definition = registry.candidates[candidate_id]
        default_requires_compile = definition.build.compilation_mode in {
            "adapter-aot",
            "adapter-jit",
        }
        decision = decide_precompile(
            policy,
            compilation_required=requirements.get(candidate_id, default_requires_compile),
            cache_hit=hits.get(candidate_id, False),
        )
        actions.append(
            CandidateActivationAction(
                candidate_id=candidate_id,
                compilation_mode=definition.build.compilation_mode,
                compilation_required=decision.compilation_required,
                cache_hit=decision.cache_hit,
                action=decision.action,
                rationale_codes=decision.rationale_codes,
            )
        )
    return CandidateActivationPlan(
        registry_sha256=resolution.registry_sha256,
        resolution_sha256=canonical_sha256(resolution),
        activated_candidate_ids=activated,
        actions=actions,
        registry_candidate_count=len(registry.candidates),
        inactive_candidate_count=len(registry.candidates) - len(activated),
    )


def evaluate_candidate_timing_gate(
    plan: CandidateActivationPlan,
    *,
    prepared_candidate_ids: list[str] | None = None,
) -> CandidateTimingGate:
    """Refuse timed cases until the single activated peer is prepared.

    Preparation means either materializing a verified cache hit or completing the requested
    precompile. With the explicit ``off`` switch, inline compilation remains diagnostic-only and
    is surfaced as a warning; compilation is still forbidden once steady-state timing begins.
    """

    prepared = prepared_candidate_ids or []
    unknown = set(prepared) - set(plan.activated_candidate_ids)
    if unknown:
        raise ValueError(
            "cannot prepare candidates outside the activation plan: " + ", ".join(sorted(unknown))
        )
    blockers: list[str] = []
    warnings: list[str] = []
    for action in plan.actions:
        if (
            action.action
            in {
                "reuse-precompiled-artifact",
                "precompile-before-timed-cases",
            }
            and action.candidate_id not in prepared
        ):
            blockers.append(f"CANDIDATE_NOT_PREPARED:{action.candidate_id}")
        elif action.action == "compile-inline-with-warning":
            warnings.append(f"INLINE_COMPILE_DIAGNOSTIC_ONLY:{action.candidate_id}")
    return CandidateTimingGate(
        activation_plan_sha256=canonical_sha256(plan),
        activated_candidate_ids=list(plan.activated_candidate_ids),
        prepared_candidate_ids=list(prepared),
        timed_benchmark_allowed=not blockers,
        timing_eligibility=(
            "blocked" if blockers else "diagnostic-only" if warnings else "official"
        ),
        blockers=blockers,
        warnings=warnings,
    )

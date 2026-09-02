from __future__ import annotations

from datetime import UTC, datetime

from infraswe.history.heuristics import (
    analyze_integration_preflight,
    analyze_python_changes,
    audit_explainable_judgment_lock,
    compile_explainable_judgment,
    freeze_explainable_judgment,
)

DIGEST = "sha256:" + "a" * 64


def _by_rule(before: dict[str, str], after: dict[str, str], diff: str = ""):
    return {item.rule_id: item for item in analyze_python_changes(before, after, unified_diff=diff)}


def test_retained_but_unused_parameter_is_a_blocking_contract_failure() -> None:
    before = {
        "fused.py": ("def fused_topk(x, renormalize):\n    return x / 2 if renormalize else x\n")
    }
    after = {"fused.py": "def fused_topk(x, renormalize):\n    return x\n"}
    result = _by_rule(before, after)["interface-parameter-use-preserved"]
    assert result.status == "fail"
    assert result.failure_code == "CALLER_CONTRACT_PARAMETER_IGNORED:renormalize"
    assert "silently ignored" in result.conclusion


def test_test_matrix_contraction_and_broad_warning_suppression_are_explicit() -> None:
    before = {
        "python/pkg/test_kernel.py": (
            "DTYPES = [1, 2, 3]\nM = [1, 33, 64, 222, 999]\n"
            "N = [128, 1024, 2048]\nBLOCKS = [1, 2, 3, 4]\n"
        )
    }
    after = {
        "test/runtime/test_kernel.py": (
            "import warnings\n"
            "warnings.filterwarnings('ignore', category=UserWarning)\n"
            "DTYPES = [1, 2]\nM = [1, 64, 222, 999]\n"
            "N = [128, 2048]\nBLOCKS = [1]\n"
        )
    }
    results = _by_rule(before, after)
    matrix = results["test-matrix-not-silently-contracted"]
    warning = results["diagnostics-remain-specific"]
    assert matrix.status == "fail"
    assert "180->16" in matrix.evidence[0]
    assert warning.status == "fail"
    assert warning.failure_code == "DIAGNOSTIC_SUPPRESSION_BROAD_USER_WARNING"


def test_repeated_leaf_fix_forces_central_owner_review_without_a_score() -> None:
    diff = """\
+++ b/a.py
+with torch.cuda.device(device):
+++ b/b.py
+with torch.cuda.device(device):
+++ b/c.py
+with torch.cuda.device(device):
"""
    result = _by_rule({}, {}, diff)["cross-cutting-fix-owned-centrally"]
    assert result.status == "unresolved"
    assert result.failure_code == "CENTRAL_OWNERSHIP_REVIEW_REQUIRED"


def test_default_rules_freeze_a_human_readable_revise_decision() -> None:
    results = analyze_python_changes(
        {"op.py": "def op(x, flag):\n    return x if flag else -x\n"},
        {"op.py": "def op(x, flag):\n    return x\n"},
    )
    material = compile_explainable_judgment(
        case_id="vllm-pr-14027",
        candidate_sha256=DIGEST,
        test_plan_sha256="sha256:" + "b" * 64,
        evidence_sha256="sha256:" + "c" * 64,
        observations=results,
        frozen_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    assert material.policy_id == "historical-explainable-agent-v0.5-r4"
    assert material.decision == "revise"
    assert not hasattr(material, "score")
    assert "silently ignored" in material.narrative
    lock = freeze_explainable_judgment(material)
    assert audit_explainable_judgment_lock(lock)


def test_r2_policy_stays_frozen_at_the_original_four_rules() -> None:
    results = analyze_python_changes(
        {"op.py": "x = 1\n"},
        {"op.py": "x = 2\n"},
        policy_id="historical-explainable-agent-v0.5-r2",
    )
    assert [item.rule_id for item in results] == [
        "interface-parameter-use-preserved",
        "test-matrix-not-silently-contracted",
        "diagnostics-remain-specific",
        "cross-cutting-fix-owned-centrally",
    ]


def test_r4_revise_semantics_remain_replayable_but_r5_is_polarized() -> None:
    observations = analyze_python_changes(
        {"op.py": "def op(x, flag):\n    return x if flag else -x\n"},
        {"op.py": "def op(x, flag):\n    return x\n"},
        policy_id="historical-explainable-agent-v0.5-r4",
    )
    arguments = {
        "case_id": "policy-replay",
        "candidate_sha256": DIGEST,
        "test_plan_sha256": "sha256:" + "b" * 64,
        "evidence_sha256": "sha256:" + "c" * 64,
        "observations": observations,
        "frozen_at": datetime(2026, 9, 1, tzinfo=UTC),
    }
    legacy = compile_explainable_judgment(
        **arguments,
        policy_id="historical-explainable-agent-v0.5-r4",
    )
    polarized = compile_explainable_judgment(
        **arguments,
        policy_id="historical-explainable-agent-v0.5-r5-polarized",
    )
    assert legacy.decision == "revise"
    assert polarized.decision == "reject"


def test_r3_flags_a_low_level_test_cluster_moved_into_a_broad_root() -> None:
    before = {
        "python/pkg/test/test_activation.py": "def test_activation():\n    pass\n",
        "python/pkg/test/test_block_fp8.py": "def test_block_fp8():\n    pass\n",
        "python/pkg/test/test_layernorm.py": "def test_layernorm():\n    pass\n",
    }
    after = {
        "test/srt/test_activation.py": before["python/pkg/test/test_activation.py"],
        "test/srt/test_block_fp8.py": before["python/pkg/test/test_block_fp8.py"],
        "test/srt/test_layernorm.py": before["python/pkg/test/test_layernorm.py"],
    }
    results = {
        item.rule_id: item
        for item in analyze_python_changes(
            before,
            after,
            policy_id="historical-explainable-agent-v0.5-r3",
        )
    }
    placement = results["moved-tests-have-a-named-owner-directory"]
    assert placement.status == "fail"
    assert placement.failure_code == "TEST_DIRECTORY_OWNERSHIP_UNRESOLVED"
    assert "test/srt" in placement.evidence[0]


def test_r3_traces_storage_bias_performance_and_quality_as_separate_contracts() -> None:
    before = {
        "vllm/model_executor/models/deepseek_v2.py": (
            "class MoE:\n"
            "    def __init__(self):\n"
            "        self.gate = ReplicatedLinear(8, 4, bias=False)\n"
            "        self.gate.correction_bias = nn.Parameter(torch.empty(4))\n"
        )
    }
    after = {
        "vllm/model_executor/models/deepseek_v2.py": (
            "class MoE:\n"
            "    def __init__(self):\n"
            "        self.gate = ReplicatedLinear(\n"
            "            8, 4, bias=False, params_dtype=torch.float32\n"
            "        )\n"
            "        self.gate.correction_bias = nn.Parameter(torch.empty(4))\n"
        )
    }
    diff = """\
diff --git a/vllm/model_executor/models/deepseek_v2.py b/vllm/model_executor/models/deepseek_v2.py
--- a/vllm/model_executor/models/deepseek_v2.py
+++ b/vllm/model_executor/models/deepseek_v2.py
@@ -1 +1 @@
+self.gate = ReplicatedLinear(8, 4, params_dtype=torch.float32)
"""
    results = {
        item.rule_id: item
        for item in analyze_python_changes(
            before,
            after,
            unified_diff=diff,
            policy_id="historical-explainable-agent-v0.5-r3",
        )
    }
    assert (
        results["compute-dtype-is-separated-from-parameter-storage"].failure_code
        == "PARAMETER_STORAGE_DTYPE_WIDENED_WITHOUT_COST_EVIDENCE"
    )
    assert (
        results["companion-parameter-dtypes-are-explicit"].failure_code
        == "COMPANION_PARAMETER_DTYPE_NOT_EXPLICIT"
    )
    assert (
        results["routing-steady-state-performance-is-evidenced"].failure_code
        == "ROUTER_STEADY_STATE_PERFORMANCE_MISSING"
    )
    assert (
        results["routing-end-to-end-quality-is-evidenced"].failure_code
        == "ROUTER_END_TO_END_QUALITY_MISSING"
    )


def test_r3_accepts_explicit_dtype_and_external_evidence_without_weights() -> None:
    before = {
        "router.py": (
            "class MoE:\n"
            "    def __init__(self):\n"
            "        self.gate = Linear(8, 4)\n"
            "        self.bias = nn.Parameter(torch.empty(4, dtype=torch.float32))\n"
        )
    }
    after = {
        "router.py": (
            "class MoE:\n"
            "    def __init__(self):\n"
            "        self.gate = Linear(8, 4, params_dtype=torch.float32)\n"
            "        self.bias = nn.Parameter(torch.empty(4, dtype=torch.float32))\n"
        )
    }
    diff = """\
diff --git a/router.py b/router.py
--- a/router.py
+++ b/router.py
@@ -1 +1 @@
+self.gate = Linear(8, 4, params_dtype=torch.float32)
"""
    evidence_codes = frozenset(
        {
            "PARAMETER_STORAGE_COST_ACCOUNTED",
            "ROUTER_STEADY_STATE_BENCHMARK",
            "ROUTER_END_TO_END_QUALITY",
        }
    )
    observations = analyze_python_changes(
        before,
        after,
        unified_diff=diff,
        policy_id="historical-explainable-agent-v0.5-r3",
        evidence_codes=evidence_codes,
    )
    material = compile_explainable_judgment(
        case_id="router-explicit-evidence",
        candidate_sha256=DIGEST,
        test_plan_sha256="sha256:" + "b" * 64,
        evidence_sha256="sha256:" + "c" * 64,
        observations=observations,
        frozen_at=datetime(2026, 9, 2, tzinfo=UTC),
        policy_id="historical-explainable-agent-v0.5-r3",
    )
    assert material.policy_id == "historical-explainable-agent-v0.5-r3"
    assert material.decision == "accept_with_scope"
    assert not hasattr(material, "score")


def test_r3_keeps_a_one_line_barrier_fix_acceptable_but_names_follow_up_debt() -> None:
    diff = """\
diff --git a/csrc/moe/kernel.cu b/csrc/moe/kernel.cu
--- a/csrc/moe/kernel.cu
+++ b/csrc/moe/kernel.cu
@@ -1 +1,2 @@
 init_shared();
+__syncthreads();
 consume_shared();
"""
    observations = analyze_python_changes(
        {},
        {},
        unified_diff=diff,
        policy_id="historical-explainable-agent-v0.5-r3",
    )
    follow_up = {item.rule_id: item for item in observations}[
        "minimal-concurrency-repair-has-follow-up-owner"
    ]
    assert follow_up.status == "unresolved"
    assert follow_up.blocking is False
    material = compile_explainable_judgment(
        case_id="barrier-interim-fix",
        candidate_sha256=DIGEST,
        test_plan_sha256="sha256:" + "b" * 64,
        evidence_sha256="sha256:" + "c" * 64,
        observations=observations,
        frozen_at=datetime(2026, 9, 2, tzinfo=UTC),
        policy_id="historical-explainable-agent-v0.5-r3",
    )
    assert material.decision == "accept_with_scope"
    assert material.rationale_codes == []


def test_r4_preflight_rejects_ungated_optional_backend_without_fallback() -> None:
    results = analyze_integration_preflight(
        capability_contracts={
            "deep_gemm_sm90": (
                False,
                False,
                "environment flag imports and calls backend directly",
            )
        },
        targeted_architecture_generation=None,
        default_architecture_generation=None,
        successor_generation_covered=False,
    )
    capability = {item.rule_id: item for item in results}[
        "runtime-capabilities-are-gated-with-fallbacks"
    ]
    assert capability.status == "fail"
    assert capability.failure_code == "OPTIONAL_CAPABILITY_GATE_OR_FALLBACK_MISSING"


def test_r4_preflight_names_nondefault_generation_and_competing_owner() -> None:
    results = analyze_integration_preflight(
        capability_contracts={},
        targeted_architecture_generation="V0",
        default_architecture_generation="V1",
        successor_generation_covered=False,
        competing_fix_refs=("vllm#12150",),
        canonical_owner_ref="vllm#12150",
        candidate_ref="vllm#12111",
    )
    by_rule = {item.rule_id: item for item in results}
    assert (
        by_rule["optimization-targets-current-architecture-generation"].failure_code
        == "NON_DEFAULT_ARCHITECTURE_ONLY"
    )
    assert (
        by_rule["competing-fixes-have-one-canonical-owner"].failure_code
        == "COMPETING_FIX_HAS_DIFFERENT_CANONICAL_OWNER"
    )


def test_r4_inherits_r3_source_questions_without_a_score() -> None:
    results = analyze_python_changes(
        {"router.py": "def route(x, flag):\n    return x if flag else -x\n"},
        {"router.py": "def route(x, flag):\n    return x\n"},
        policy_id="historical-explainable-agent-v0.5-r4",
    )
    assert len(results) == 10
    assert results[0].failure_code == "CALLER_CONTRACT_PARAMETER_IGNORED:flag"

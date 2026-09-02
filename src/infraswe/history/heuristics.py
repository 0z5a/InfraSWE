from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Mapping
from datetime import UTC, datetime
from math import prod
from pathlib import PurePosixPath
from typing import TypeVar

from infraswe.history.blind import canonical_sha256
from infraswe.models.draft import Digest
from infraswe.models.history import (
    HistoricalExplainableJudgmentLock,
    HistoricalExplainableJudgmentMaterial,
    HistoricalExplainablePolicyId,
    HistoricalHeuristicObservation,
)

Node = TypeVar("Node", ast.FunctionDef, ast.AsyncFunctionDef)

R2_POLICY_ID = "historical-explainable-agent-v0.5-r2"
R3_POLICY_ID = "historical-explainable-agent-v0.5-r3"
R4_POLICY_ID = "historical-explainable-agent-v0.5-r4"
R5_POLICY_ID = "historical-explainable-agent-v0.5-r5-polarized"


def _parse_sources(sources: Mapping[str, str]) -> tuple[dict[str, ast.Module], list[str]]:
    trees: dict[str, ast.Module] = {}
    errors: list[str] = []
    for path, source in sources.items():
        if not path.endswith(".py"):
            continue
        try:
            trees[path] = ast.parse(source, filename=path)
        except SyntaxError as error:
            errors.append(f"{path}:{error.lineno}:{error.msg}")
    return trees, errors


def _matched_python_paths(
    before: Mapping[str, ast.Module], after: Mapping[str, ast.Module]
) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    used_after: set[str] = set()
    for old_path in sorted(before):
        if old_path in after:
            pairs.append((old_path, old_path))
            used_after.add(old_path)
            continue
        basename = PurePosixPath(old_path).name
        candidates = [
            path
            for path in after
            if PurePosixPath(path).name == basename and path not in used_after
        ]
        if len(candidates) == 1:
            pairs.append((old_path, candidates[0]))
            used_after.add(candidates[0])
    return pairs


def _functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    found: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    counts: defaultdict[str, int] = defaultdict(int)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            counts[node.name] += 1
            key = node.name if counts[node.name] == 1 else f"{node.name}#{counts[node.name]}"
            found[key] = node
    return found


def _parameter_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    arguments = node.args
    names = [item.arg for item in arguments.posonlyargs + arguments.args + arguments.kwonlyargs]
    if arguments.vararg:
        names.append(arguments.vararg.arg)
    if arguments.kwarg:
        names.append(arguments.kwarg.arg)
    return names


def _load_count(node: ast.AST, name: str) -> int:
    return sum(
        isinstance(item, ast.Name) and isinstance(item.ctx, ast.Load) and item.id == name
        for item in ast.walk(node)
    )


def detect_ignored_parameters(
    before: Mapping[str, ast.Module], after: Mapping[str, ast.Module]
) -> HistoricalHeuristicObservation:
    losses: list[str] = []
    for old_path, new_path in _matched_python_paths(before, after):
        old_functions = _functions(before[old_path])
        new_functions = _functions(after[new_path])
        for function_name in sorted(old_functions.keys() & new_functions.keys()):
            old_function = old_functions[function_name]
            new_function = new_functions[function_name]
            if ast.dump(old_function) == ast.dump(new_function):
                continue
            common_parameters = set(_parameter_names(old_function)) & set(
                _parameter_names(new_function)
            )
            for parameter in sorted(common_parameters):
                old_uses = _load_count(old_function, parameter)
                new_uses = _load_count(new_function, parameter)
                if old_uses > 0 and new_uses == 0:
                    losses.append(
                        f"{old_path}->{new_path}:{function_name} parameter={parameter} "
                        f"loads={old_uses}->0"
                    )
    if losses:
        parameters = sorted({item.split("parameter=", 1)[1].split()[0] for item in losses})
        return HistoricalHeuristicObservation(
            rule_id="interface-parameter-use-preserved",
            question="Does every retained caller-facing parameter still affect behavior?",
            status="fail",
            blocking=True,
            evidence=losses,
            conclusion=(
                "The patch keeps the public parameter(s) "
                + ", ".join(parameters)
                + " but removes all uses, so callers can request behavior that is silently ignored."
            ),
            failure_code="CALLER_CONTRACT_PARAMETER_IGNORED:" + ",".join(parameters),
        )
    return HistoricalHeuristicObservation(
        rule_id="interface-parameter-use-preserved",
        question="Does every retained caller-facing parameter still affect behavior?",
        status="pass",
        blocking=True,
        conclusion=(
            "No retained parameter changed from used to completely unused in changed Python code."
        ),
    )


def _sequence_constants(tree: ast.Module) -> dict[str, int]:
    constants: dict[str, int] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        value = node.value
        if not isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            continue
        for target in targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                constants[target.id] = len(value.elts)
    return constants


def detect_test_matrix_contraction(
    before: Mapping[str, ast.Module], after: Mapping[str, ast.Module]
) -> HistoricalHeuristicObservation:
    contractions: list[str] = []
    ratios: list[float] = []
    for old_path, new_path in _matched_python_paths(before, after):
        if "test" not in PurePosixPath(old_path).name.lower():
            continue
        old_constants = _sequence_constants(before[old_path])
        new_constants = _sequence_constants(after[new_path])
        shared = sorted(old_constants.keys() & new_constants.keys())
        changed = [name for name in shared if old_constants[name] != new_constants[name]]
        if len(shared) < 2 or not changed:
            continue
        before_cases = prod(old_constants[name] for name in shared)
        after_cases = prod(new_constants[name] for name in shared)
        ratio = after_cases / before_cases if before_cases else 1.0
        if ratio < 0.5:
            ratios.append(ratio)
            details = ", ".join(
                f"{name}:{old_constants[name]}->{new_constants[name]}" for name in changed
            )
            contractions.append(
                f"{old_path}->{new_path} explicit Cartesian upper bound "
                f"{before_cases}->{after_cases} ({ratio:.1%}); {details}"
            )
    if contractions:
        worst = min(ratios)
        return HistoricalHeuristicObservation(
            rule_id="test-matrix-not-silently-contracted",
            question="Does a test move preserve at least half of its explicit input matrix?",
            status="fail",
            blocking=True,
            evidence=contractions,
            conclusion=(
                f"The explicit test matrix falls to {worst:.1%} of its prior size. "
                "No compensating evidence is present in this source-level check."
            ),
            failure_code="TEST_MATRIX_CONTRACTION_GT_50_PERCENT",
        )
    return HistoricalHeuristicObservation(
        rule_id="test-matrix-not-silently-contracted",
        question="Does a test move preserve at least half of its explicit input matrix?",
        status="pass",
        blocking=True,
        conclusion="No changed Python test file loses more than half of its explicit matrix.",
    )


def _broad_user_warning_suppressions(tree: ast.Module) -> list[int]:
    lines: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if not (
            isinstance(function, ast.Attribute)
            and isinstance(function.value, ast.Name)
            and function.value.id == "warnings"
            and function.attr == "filterwarnings"
        ):
            continue
        ignores = bool(
            node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "ignore"
        )
        keywords = {item.arg: item.value for item in node.keywords if item.arg}
        category = keywords.get("category")
        user_warning = isinstance(category, ast.Name) and category.id == "UserWarning"
        scoped = "message" in keywords or "module" in keywords
        if ignores and user_warning and not scoped:
            lines.append(node.lineno)
    return lines


def detect_broad_warning_suppression(
    before: Mapping[str, ast.Module], after: Mapping[str, ast.Module]
) -> HistoricalHeuristicObservation:
    added: list[str] = []
    pairs = _matched_python_paths(before, after)
    for old_path, new_path in pairs:
        old_count = len(_broad_user_warning_suppressions(before[old_path]))
        new_lines = _broad_user_warning_suppressions(after[new_path])
        if len(new_lines) > old_count:
            added.extend(f"{new_path}:{line}" for line in new_lines[old_count:])
    if added:
        return HistoricalHeuristicObservation(
            rule_id="diagnostics-remain-specific",
            question="Does the patch avoid globally suppressing a broad warning category?",
            status="fail",
            blocking=True,
            evidence=added,
            conclusion=(
                "The patch adds an unscoped UserWarning suppression; unrelated diagnostics can "
                "disappear and make the moved suite less informative."
            ),
            failure_code="DIAGNOSTIC_SUPPRESSION_BROAD_USER_WARNING",
        )
    return HistoricalHeuristicObservation(
        rule_id="diagnostics-remain-specific",
        question="Does the patch avoid globally suppressing a broad warning category?",
        status="pass",
        blocking=True,
        conclusion="No new unscoped warnings.filterwarnings(ignore, UserWarning) call was found.",
    )


def detect_repeated_leaf_fix(unified_diff: str) -> HistoricalHeuristicObservation:
    current_path: str | None = None
    occurrences: defaultdict[str, set[str]] = defaultdict(set)
    for raw_line in unified_diff.splitlines():
        if raw_line.startswith("+++ b/"):
            current_path = raw_line[6:]
            continue
        if not current_path or not raw_line.startswith("+") or raw_line.startswith("+++"):
            continue
        normalized = " ".join(raw_line[1:].strip().split())
        if len(normalized) >= 20 and any(character.isalpha() for character in normalized):
            occurrences[normalized].add(current_path)
    repeated = [
        f"{line!r} added in {', '.join(sorted(paths))}"
        for line, paths in sorted(occurrences.items())
        if len(paths) >= 3
    ]
    if repeated:
        return HistoricalHeuristicObservation(
            rule_id="cross-cutting-fix-owned-centrally",
            question="Is a repeated cross-cutting fix implemented at one central owner?",
            status="unresolved",
            blocking=True,
            evidence=repeated,
            conclusion=(
                "The same behavior is added in at least three leaf files. Search the central "
                "runner/dispatcher owner before accepting local correctness results."
            ),
            failure_code="CENTRAL_OWNERSHIP_REVIEW_REQUIRED",
        )
    return HistoricalHeuristicObservation(
        rule_id="cross-cutting-fix-owned-centrally",
        question="Is a repeated cross-cutting fix implemented at one central owner?",
        status="pass",
        blocking=True,
        conclusion="No substantive line is newly duplicated across three or more changed files.",
    )


def detect_test_destination_taxonomy(
    before: Mapping[str, ast.Module], after: Mapping[str, ast.Module]
) -> HistoricalHeuristicObservation:
    """Require an explicit owner directory for a cluster of low-level tests."""

    operator_markers = {
        "activation",
        "attention",
        "block",
        "fp8",
        "gemm",
        "kernel",
        "layernorm",
        "moe",
        "norm",
        "quant",
    }
    moves: list[tuple[str, str]] = []
    for old_path, new_path in _matched_python_paths(before, after):
        if old_path == new_path or "test" not in PurePosixPath(new_path).name.lower():
            continue
        stem_tokens = set(PurePosixPath(new_path).stem.lower().replace("-", "_").split("_"))
        if stem_tokens & operator_markers:
            moves.append((old_path, new_path))

    by_parent: defaultdict[str, list[tuple[str, str]]] = defaultdict(list)
    for move in moves:
        by_parent[str(PurePosixPath(move[1]).parent)].append(move)
    broad_clusters = {
        parent: cluster
        for parent, cluster in by_parent.items()
        if len(cluster) >= 3 and len(PurePosixPath(parent).parts) <= 2
    }
    if broad_clusters:
        evidence = [
            f"{parent}: " + ", ".join(f"{old}->{new}" for old, new in cluster)
            for parent, cluster in sorted(broad_clusters.items())
        ]
        return HistoricalHeuristicObservation(
            rule_id="moved-tests-have-a-named-owner-directory",
            question=(
                "When three or more related low-level tests move together, does their "
                "destination name the owning domain rather than a broad subsystem root?"
            ),
            status="fail",
            blocking=True,
            evidence=evidence,
            conclusion=(
                "A cluster of low-level tests is moved directly into a broad test root. "
                "Choose or justify a named owner directory before accepting the relocation."
            ),
            failure_code="TEST_DIRECTORY_OWNERSHIP_UNRESOLVED",
        )
    return HistoricalHeuristicObservation(
        rule_id="moved-tests-have-a-named-owner-directory",
        question=(
            "When three or more related low-level tests move together, does their "
            "destination name the owning domain rather than a broad subsystem root?"
        ),
        status="pass",
        blocking=True,
        conclusion="No three-test low-level cluster is moved into a broad test root.",
    )


def _attribute_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = _attribute_name(node.value)
        return f"{owner}.{node.attr}" if owner else node.attr
    return ""


def _is_fp32(node: ast.AST) -> bool:
    return _attribute_name(node) in {"float32", "torch.float32"} or (
        isinstance(node, ast.Constant) and node.value in {"float32", "fp32"}
    )


def _fp32_parameter_storage_calls(tree: ast.Module) -> list[str]:
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        keywords = {item.arg: item.value for item in node.keywords if item.arg}
        params_dtype = keywords.get("params_dtype")
        if params_dtype is not None and _is_fp32(params_dtype):
            calls.append(f"{_attribute_name(node.func) or '<call>'}:{node.lineno}")
    return calls


def _new_fp32_parameter_storage(
    before: Mapping[str, ast.Module], after: Mapping[str, ast.Module]
) -> list[str]:
    widened: list[str] = []
    for old_path, new_path in _matched_python_paths(before, after):
        old_count = len(_fp32_parameter_storage_calls(before[old_path]))
        new_calls = _fp32_parameter_storage_calls(after[new_path])
        if len(new_calls) > old_count:
            widened.extend(f"{new_path}:{call}" for call in new_calls[old_count:])
    return widened


def detect_parameter_storage_widening(
    before: Mapping[str, ast.Module],
    after: Mapping[str, ast.Module],
    *,
    evidence_codes: frozenset[str],
) -> HistoricalHeuristicObservation:
    widened = _new_fp32_parameter_storage(before, after)
    question = (
        "If computation moves to FP32, is parameter storage kept narrow or is the added "
        "memory/checkpoint cost explicitly accounted for?"
    )
    if not widened:
        return HistoricalHeuristicObservation(
            rule_id="compute-dtype-is-separated-from-parameter-storage",
            question=question,
            status="not-applicable",
            blocking=True,
            conclusion="No new params_dtype=torch.float32 storage request was found.",
        )
    if "PARAMETER_STORAGE_COST_ACCOUNTED" in evidence_codes:
        return HistoricalHeuristicObservation(
            rule_id="compute-dtype-is-separated-from-parameter-storage",
            question=question,
            status="pass",
            blocking=True,
            evidence=widened,
            counterevidence=["PARAMETER_STORAGE_COST_ACCOUNTED"],
            conclusion="The direct FP32 storage widening has explicit memory/checkpoint evidence.",
        )
    return HistoricalHeuristicObservation(
        rule_id="compute-dtype-is-separated-from-parameter-storage",
        question=question,
        status="fail",
        blocking=True,
        evidence=widened,
        conclusion=(
            "The patch directly widens parameter storage to FP32 without an explicit memory "
            "and checkpoint contract; prefer a compute-only cast or provide that evidence."
        ),
        failure_code="PARAMETER_STORAGE_DTYPE_WIDENED_WITHOUT_COST_EVIDENCE",
    )


def _assignment_target_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    return [_attribute_name(target) for target in targets]


def _implicit_bias_parameters(tree: ast.Module) -> list[str]:
    implicit: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        targets = _assignment_target_names(node)
        if not any("bias" in target.lower() for target in targets):
            continue
        parameter_calls = [
            item
            for item in ast.walk(node.value)
            if isinstance(item, ast.Call)
            and _attribute_name(item.func).split(".")[-1] == "Parameter"
        ]
        if not parameter_calls:
            continue
        initializers = [
            item
            for item in ast.walk(node.value)
            if isinstance(item, ast.Call)
            and _attribute_name(item.func).split(".")[-1] in {"empty", "ones", "zeros"}
        ]
        for initializer in initializers:
            if not any(item.arg == "dtype" for item in initializer.keywords):
                implicit.append(f"{','.join(targets)}:{initializer.lineno}")
    return implicit


def detect_companion_parameter_dtype(
    before: Mapping[str, ast.Module], after: Mapping[str, ast.Module]
) -> HistoricalHeuristicObservation:
    widened = _new_fp32_parameter_storage(before, after)
    question = (
        "When a parameter path is widened to FP32, do companion bias/correction tensors "
        "declare the same dtype explicitly?"
    )
    if not widened:
        return HistoricalHeuristicObservation(
            rule_id="companion-parameter-dtypes-are-explicit",
            question=question,
            status="not-applicable",
            blocking=True,
            conclusion="No new FP32 parameter-storage path requires a companion dtype audit.",
        )
    implicit = [
        f"{path}:{item}"
        for path, tree in sorted(after.items())
        for item in _implicit_bias_parameters(tree)
    ]
    if implicit:
        return HistoricalHeuristicObservation(
            rule_id="companion-parameter-dtypes-are-explicit",
            question=question,
            status="fail",
            blocking=True,
            evidence=widened + implicit,
            conclusion=(
                "The main parameter storage is explicitly FP32 while a companion bias or "
                "correction parameter still inherits the ambient default dtype."
            ),
            failure_code="COMPANION_PARAMETER_DTYPE_NOT_EXPLICIT",
        )
    return HistoricalHeuristicObservation(
        rule_id="companion-parameter-dtypes-are-explicit",
        question=question,
        status="pass",
        blocking=True,
        evidence=widened,
        conclusion="Companion bias/correction initializers declare their dtype explicitly.",
    )


def _diff_additions_by_path(unified_diff: str) -> dict[str, list[str]]:
    additions: defaultdict[str, list[str]] = defaultdict(list)
    current_path: str | None = None
    for line in unified_diff.splitlines():
        if line.startswith("+++ b/"):
            current_path = line[6:]
        elif current_path and line.startswith("+") and not line.startswith("+++"):
            additions[current_path].append(line[1:])
    return dict(additions)


def _router_precision_change(unified_diff: str) -> list[str]:
    owner_markers = ("gate", "moe", "router", "topk")
    precision_markers = (
        ".float()",
        ".type(torch.float32)",
        "dtype=torch.float32",
        "params_dtype=torch.float32",
    )
    evidence: list[str] = []
    for path, additions in _diff_additions_by_path(unified_diff).items():
        compact = ["".join(line.split()).lower() for line in additions]
        owner_text = f"{path.lower()} {' '.join(compact)}"
        if not any(marker in owner_text for marker in owner_markers):
            continue
        for line_number, line in enumerate(compact, start=1):
            if any(marker in line for marker in precision_markers):
                evidence.append(f"{path}:added-line-{line_number}:{line}")
    return evidence


def detect_router_evidence_contracts(
    unified_diff: str, *, evidence_codes: frozenset[str]
) -> list[HistoricalHeuristicObservation]:
    change = _router_precision_change(unified_diff)
    requirements = [
        (
            "routing-steady-state-performance-is-evidenced",
            (
                "Does a router precision change include compile-free steady-state "
                "performance evidence?"
            ),
            "ROUTER_STEADY_STATE_BENCHMARK",
            "ROUTER_STEADY_STATE_PERFORMANCE_MISSING",
            "The routing precision path changed without steady-state performance evidence.",
        ),
        (
            "routing-end-to-end-quality-is-evidenced",
            "Does a router precision change include task-level quality evidence?",
            "ROUTER_END_TO_END_QUALITY",
            "ROUTER_END_TO_END_QUALITY_MISSING",
            "The routing precision path changed without task-level quality evidence.",
        ),
    ]
    observations: list[HistoricalHeuristicObservation] = []
    for rule_id, question, evidence_code, failure_code, failure_conclusion in requirements:
        if not change:
            observations.append(
                HistoricalHeuristicObservation(
                    rule_id=rule_id,
                    question=question,
                    status="not-applicable",
                    blocking=True,
                    conclusion="No router precision semantic change was found.",
                )
            )
        elif evidence_code in evidence_codes:
            observations.append(
                HistoricalHeuristicObservation(
                    rule_id=rule_id,
                    question=question,
                    status="pass",
                    blocking=True,
                    evidence=change,
                    counterevidence=[evidence_code],
                    conclusion=f"The required evidence contract is present: {evidence_code}.",
                )
            )
        else:
            observations.append(
                HistoricalHeuristicObservation(
                    rule_id=rule_id,
                    question=question,
                    status="fail",
                    blocking=True,
                    evidence=change,
                    conclusion=failure_conclusion,
                    failure_code=failure_code,
                )
            )
    return observations


def detect_minimal_concurrency_fix_scope(unified_diff: str) -> HistoricalHeuristicObservation:
    additions = _diff_additions_by_path(unified_diff)
    synchronization_markers = ("__syncthreads", "barrier", "synchronize", "thread_fence")
    code_lines = [
        f"{path}:{line.strip()}"
        for path, lines in additions.items()
        if PurePosixPath(path).suffix.lower() in {".c", ".cc", ".cpp", ".cu", ".cuh", ".h"}
        for line in lines
        if line.strip() and not line.lstrip().startswith(("//", "/*", "*"))
    ]
    synchronization = [
        line
        for line in code_lines
        if any(marker in line.lower() for marker in synchronization_markers)
    ]
    if synchronization and len(code_lines) <= 4:
        return HistoricalHeuristicObservation(
            rule_id="minimal-concurrency-repair-has-follow-up-owner",
            question=(
                "Is a minimal synchronization-only kernel repair explicitly scoped as an "
                "interim fix with a follow-up owner?"
            ),
            status="unresolved",
            blocking=False,
            evidence=synchronization,
            conclusion=(
                "The patch is a minimal synchronization repair. It can be accepted for "
                "correctness, but the broader kernel ownership/design follow-up must be named."
            ),
            failure_code="FOLLOW_UP_ARCHITECTURE_OWNERSHIP_REQUIRED",
        )
    return HistoricalHeuristicObservation(
        rule_id="minimal-concurrency-repair-has-follow-up-owner",
        question=(
            "Is a minimal synchronization-only kernel repair explicitly scoped as an "
            "interim fix with a follow-up owner?"
        ),
        status="not-applicable",
        blocking=False,
        conclusion="The patch is not a minimal synchronization-only native-kernel repair.",
    )


def analyze_integration_preflight(
    *,
    capability_contracts: Mapping[str, tuple[bool, bool, str]],
    targeted_architecture_generation: str | None,
    default_architecture_generation: str | None,
    successor_generation_covered: bool,
    competing_fix_refs: tuple[str, ...] = (),
    canonical_owner_ref: str | None = None,
    candidate_ref: str | None = None,
) -> list[HistoricalHeuristicObservation]:
    """Run r4 integration/ownership questions before expensive benchmarks.

    Each capability maps to ``(gate_present, fallback_present, evidence)``. The
    order is intentional: platform safety, architecture generation, then fix
    ownership. No score or weight is calculated.
    """

    missing = [
        f"{name}: gate={gate}, fallback={fallback}; {evidence}"
        for name, (gate, fallback, evidence) in sorted(capability_contracts.items())
        if not gate or not fallback
    ]
    if missing:
        capability = HistoricalHeuristicObservation(
            rule_id="runtime-capabilities-are-gated-with-fallbacks",
            question=(
                "Is every optional hardware/runtime capability checked before activation, "
                "with a tested fallback when unavailable?"
            ),
            status="fail",
            blocking=True,
            evidence=missing,
            conclusion=(
                "At least one optional capability can select the optimized path without both "
                "an availability gate and a fallback."
            ),
            failure_code="OPTIONAL_CAPABILITY_GATE_OR_FALLBACK_MISSING",
        )
    else:
        capability = HistoricalHeuristicObservation(
            rule_id="runtime-capabilities-are-gated-with-fallbacks",
            question=(
                "Is every optional hardware/runtime capability checked before activation, "
                "with a tested fallback when unavailable?"
            ),
            status="pass",
            blocking=True,
            evidence=[
                f"{name}: {evidence}"
                for name, (_, _, evidence) in sorted(capability_contracts.items())
            ],
            conclusion="Every declared optional capability has a gate and fallback.",
        )

    architecture_question = (
        "Does the change belong to the current default architecture generation, or cover its "
        "successor under the same owner?"
    )
    if (
        targeted_architecture_generation
        and default_architecture_generation
        and targeted_architecture_generation != default_architecture_generation
        and not successor_generation_covered
    ):
        architecture = HistoricalHeuristicObservation(
            rule_id="optimization-targets-current-architecture-generation",
            question=architecture_question,
            status="fail",
            blocking=True,
            evidence=[
                f"target={targeted_architecture_generation}",
                f"default={default_architecture_generation}",
                "successor_covered=false",
            ],
            conclusion=(
                "The optimization is confined to a non-default architecture generation and "
                "does not cover the active successor."
            ),
            failure_code="NON_DEFAULT_ARCHITECTURE_ONLY",
        )
    elif targeted_architecture_generation and default_architecture_generation:
        architecture = HistoricalHeuristicObservation(
            rule_id="optimization-targets-current-architecture-generation",
            question=architecture_question,
            status="pass",
            blocking=True,
            evidence=[
                f"target={targeted_architecture_generation}",
                f"default={default_architecture_generation}",
                f"successor_covered={str(successor_generation_covered).lower()}",
            ],
            conclusion="The patch targets the default generation or covers its successor.",
        )
    else:
        architecture = HistoricalHeuristicObservation(
            rule_id="optimization-targets-current-architecture-generation",
            question=architecture_question,
            status="not-applicable",
            blocking=True,
            conclusion="No architecture-generation split applies to this change.",
        )

    ownership_question = (
        "When another change claims the same owner/failure signature, is one canonical fix "
        "chosen before implementation is judged?"
    )
    if competing_fix_refs and canonical_owner_ref is None:
        ownership = HistoricalHeuristicObservation(
            rule_id="competing-fixes-have-one-canonical-owner",
            question=ownership_question,
            status="unresolved",
            blocking=True,
            evidence=list(competing_fix_refs),
            conclusion=(
                "Competing fixes exist, but the canonical owner is unresolved; do not reward "
                "duplicate local correctness work yet."
            ),
            failure_code="COMPETING_FIX_OWNERSHIP_UNRESOLVED",
        )
    elif (
        competing_fix_refs
        and canonical_owner_ref is not None
        and candidate_ref is not None
        and canonical_owner_ref != candidate_ref
    ):
        ownership = HistoricalHeuristicObservation(
            rule_id="competing-fixes-have-one-canonical-owner",
            question=ownership_question,
            status="fail",
            blocking=True,
            evidence=[*competing_fix_refs, f"canonical={canonical_owner_ref}"],
            conclusion="A different change is the selected canonical owner for this fix.",
            failure_code="COMPETING_FIX_HAS_DIFFERENT_CANONICAL_OWNER",
        )
    else:
        ownership = HistoricalHeuristicObservation(
            rule_id="competing-fixes-have-one-canonical-owner",
            question=ownership_question,
            status="pass",
            blocking=True,
            evidence=list(competing_fix_refs),
            conclusion=(
                "No competing fix is known, or the candidate is the declared canonical owner."
            ),
        )
    return [capability, architecture, ownership]


def analyze_python_changes(
    before_sources: Mapping[str, str],
    after_sources: Mapping[str, str],
    *,
    unified_diff: str = "",
    policy_id: HistoricalExplainablePolicyId = R4_POLICY_ID,
    evidence_codes: frozenset[str] = frozenset(),
) -> list[HistoricalHeuristicObservation]:
    before, before_errors = _parse_sources(before_sources)
    after, after_errors = _parse_sources(after_sources)
    errors = before_errors + after_errors
    if errors:
        return [
            HistoricalHeuristicObservation(
                rule_id="python-source-parseable",
                question="Can all changed Python sources be parsed for semantic checks?",
                status="unresolved",
                blocking=True,
                evidence=errors,
                conclusion=(
                    "At least one changed Python source cannot be parsed, so rules cannot run."
                ),
                failure_code="PYTHON_STATIC_ANALYSIS_PARSE_FAILED",
            )
        ]
    observations = [
        detect_ignored_parameters(before, after),
        detect_test_matrix_contraction(before, after),
        detect_broad_warning_suppression(before, after),
        detect_repeated_leaf_fix(unified_diff),
    ]
    if policy_id in {R3_POLICY_ID, R4_POLICY_ID, R5_POLICY_ID}:
        observations.extend(
            [
                detect_test_destination_taxonomy(before, after),
                detect_parameter_storage_widening(
                    before,
                    after,
                    evidence_codes=evidence_codes,
                ),
                detect_companion_parameter_dtype(before, after),
                *detect_router_evidence_contracts(
                    unified_diff,
                    evidence_codes=evidence_codes,
                ),
                detect_minimal_concurrency_fix_scope(unified_diff),
            ]
        )
    return observations


def compile_explainable_judgment(
    *,
    case_id: str,
    candidate_sha256: Digest,
    test_plan_sha256: Digest,
    evidence_sha256: Digest,
    observations: list[HistoricalHeuristicObservation],
    frozen_at: datetime | None = None,
    policy_id: HistoricalExplainablePolicyId = R4_POLICY_ID,
) -> HistoricalExplainableJudgmentMaterial:
    blocking_unresolved = any(
        item.blocking and item.status == "unresolved" for item in observations
    )
    failures = [item for item in observations if item.blocking and item.status == "fail"]
    hard_failure = any((item.failure_code or "").startswith("HARD_POLICY_") for item in failures)
    polarized = policy_id == R5_POLICY_ID
    decision = (
        "unresolved"
        if blocking_unresolved
        else "reject"
        if hard_failure or (polarized and failures)
        else "revise"
        if failures
        else "accept_with_scope"
    )
    rationale_codes = sorted(
        {
            item.failure_code
            for item in observations
            if item.blocking and item.failure_code is not None
        }
    )
    narrative = " ".join(
        f"[{item.rule_id}: {item.status}] {item.conclusion}" for item in observations
    )
    return HistoricalExplainableJudgmentMaterial(
        policy_id=policy_id,
        case_id=case_id,
        candidate_sha256=candidate_sha256,
        test_plan_sha256=test_plan_sha256,
        evidence_sha256=evidence_sha256,
        observations=observations,
        decision=decision,
        rationale_codes=rationale_codes,
        narrative=narrative,
        frozen_at=frozen_at or datetime.now(UTC),
    )


def freeze_explainable_judgment(
    material: HistoricalExplainableJudgmentMaterial,
) -> HistoricalExplainableJudgmentLock:
    return HistoricalExplainableJudgmentLock(
        material=material,
        lock_sha256=canonical_sha256(material),
    )


def audit_explainable_judgment_lock(lock: HistoricalExplainableJudgmentLock) -> bool:
    return lock.lock_sha256 == canonical_sha256(lock.material)

#!/usr/bin/env python3
"""Exact-source host probes for the non-GPU portions of R7."""

from __future__ import annotations

import argparse
import ast
import contextlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import time
import types
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from infraswe.history.blind import canonical_sha256
from infraswe.io import atomic_write_json


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object in {path}")
    return payload


def _verify_locks(selection: dict[str, Any], plan: dict[str, Any]) -> tuple[str, str]:
    selection_sha = selection["selection_lock_sha256"]
    if canonical_sha256(selection["selection_material"]) != selection_sha:
        raise ValueError("selection lock digest mismatch")
    material = dict(plan)
    plan_sha = material.pop("test_plan_sha256")
    if canonical_sha256(material) != plan_sha:
        raise ValueError("test plan digest mismatch")
    if plan["selection_lock_sha256"] != selection_sha:
        raise ValueError("test plan is not bound to selection")
    return selection_sha, plan_sha


class _StripAnnotations(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        node = self.generic_visit(node)
        node.decorator_list = []
        node.returns = None
        for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
            argument.annotation = None
        if node.args.vararg is not None:
            node.args.vararg.annotation = None
        if node.args.kwarg is not None:
            node.args.kwarg.annotation = None
        return node

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.Assign:
        value = node.value if node.value is not None else ast.Constant(None)
        return ast.copy_location(ast.Assign(targets=[node.target], value=value), node)


def _function_from_class(
    source_path: Path,
    class_name: str,
    function_name: str,
    globals_dict: dict[str, Any],
) -> Callable[..., Any]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    function: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            function = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, ast.FunctionDef) and child.name == function_name
                ),
                None,
            )
            break
    if function is None:
        raise ValueError(f"missing {class_name}.{function_name} in {source_path}")
    function = _StripAnnotations().visit(function)
    ast.fix_missing_locations(function)
    namespace = dict(globals_dict)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace[function_name]


def _source_identity(paths: list[Path]) -> dict[str, str]:
    return {str(path): canonical_sha256(path.read_text(encoding="utf-8")) for path in paths}


class _StateModule:
    def __init__(self, state: str, children: list[_StateModule] | None = None):
        self._training_state = state
        self._children = children or []

    def modules(self):
        yield self
        for child in self._children:
            yield from child.modules()


class _Node:
    def __init__(self, label: str, events: list[str]):
        self.label = label
        self.events = events

    def forward(self, value=None):
        self.events.append(f"{self.label}.forward")
        return value

    def backward(self, value=None):
        self.events.append(f"{self.label}.backward")
        return value

    def backward_dw(self):
        self.events.append(f"{self.label}.backward_dw")


class _Layer:
    def __init__(self, underlying: object, label: str):
        self.layer = underlying
        self.events: list[str] = []
        self.prepare_calls = 0
        self.config = SimpleNamespace(ep_overlap_early_attn_memory_release=False)
        self.attn = _Node(f"{label}.attn", self.events)
        self.mlp = _Node(f"{label}.mlp", self.events)
        self.moe_dispatch = _Node(f"{label}.dispatch", self.events)
        self.moe_combine = _Node(f"{label}.combine", self.events)
        self.mtp_post_process = _Node(f"{label}.mtp", self.events)
        self.mhc_recompute = None
        self._fsdp_prepare_forward_module = self._prepare

    def _prepare(self):
        self.prepare_calls += 1
        self.events.append("prepare")

    def get_fp8_context(self):
        return contextlib.nullcontext()


def _megatron_probe(head_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    fsdp_path = head_root / ("megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py")
    schedule_path = head_root / "megatron/core/models/common/model_chunk_schedule_plan.py"
    combined_path = head_root / "megatron/core/pipeline_parallel/combined_1f1b.py"
    test_path = head_root / "tests/unit_tests/a2a_overlap/test_fsdp_1f1b_overlap.py"

    training_state = SimpleNamespace(PRE_BACKWARD="pre_backward", IDLE="idle")
    prepare = _function_from_class(
        fsdp_path,
        "MegatronFSDP",
        "prepare_forward_module",
        {"TrainingState": training_state},
    )
    root = _StateModule(
        training_state.PRE_BACKWARD,
        [
            _StateModule(training_state.IDLE),
            _StateModule("forward"),
            _StateModule(training_state.PRE_BACKWARD),
        ],
    )
    before = [module._training_state for module in root.modules()]
    prepare(object(), root)
    after = [module._training_state for module in root.modules()]

    run = _function_from_class(
        schedule_path,
        "TransformerLayerSchedulePlan",
        "run",
        {},
    )

    scenario_results: dict[str, Any] = {}
    for scenario, same, use_forward, use_backward in (
        ("forward_only", False, True, False),
        ("backward_only", False, False, True),
        ("different_layer_overlap", False, True, True),
        ("same_underlying_layer_overlap", True, True, True),
    ):
        forward_underlying = object()
        backward_underlying = forward_underlying if same else object()
        f_layer = _Layer(forward_underlying, "f") if use_forward else None
        b_layer = _Layer(backward_underlying, "b") if use_backward else None
        run(f_layer, b_layer, f_input="input", b_grad="grad")
        scenario_results[scenario] = {
            "forward_prepare_calls": f_layer.prepare_calls if f_layer else 0,
            "forward_events": f_layer.events if f_layer else [],
            "backward_events": b_layer.events if b_layer else [],
        }

    matrix: list[dict[str, Any]] = []
    for layer_count in range(1, 8):
        underlying = [object() for _ in range(layer_count)]
        forward_layers = [_Layer(item, f"f{i}") for i, item in enumerate(underlying)]
        backward_layers = [_Layer(item, f"b{i}") for i, item in enumerate(underlying)]
        for index in range(layer_count):
            run(
                forward_layers[index],
                backward_layers[layer_count - 1 - index],
                f_input="input",
                b_grad="grad",
                is_last_layer_in_bwd=index == layer_count - 1,
            )
        observed = sum(layer.prepare_calls for layer in forward_layers)
        expected = layer_count - (1 if layer_count % 2 else 0)
        matrix.append(
            {
                "layer_count": layer_count,
                "observed_prepare_calls": observed,
                "expected_prepare_calls": expected,
                "matches_oracle": observed == expected,
            }
        )

    combined = combined_path.read_text(encoding="utf-8")
    tests = test_path.read_text(encoding="utf-8")
    return {
        "status": "pass",
        "failure_codes": [],
        "facts": {
            "state_transition": {
                "before": before,
                "after": after,
                "only_pre_backward_changed_to_idle": after
                == [training_state.IDLE, training_state.IDLE, "forward", training_state.IDLE],
            },
            "run_scenarios": scenario_results,
            "layer_boundary_matrix": matrix,
            "all_layer_boundaries_match_oracle": all(item["matches_oracle"] for item in matrix),
            "wrapper_callback_wired": "forward_fsdp_wrapper.prepare_forward_module" in combined,
            "direct_test_present": "def test_fsdp_1f1b_genuine_forward_state" in tests,
            "direct_test_uses_two_microbatches": "num_microbatches = 2" in tests,
            "available_gpu_count": 1,
            "multi_rank_runtime": "unresolved-single-h100-and-transformer-engine-unavailable",
            "performance_and_peak_memory": (
                "unresolved-single-h100-and-transformer-engine-unavailable"
            ),
            "steady_state_compile_seconds": 0.0,
        },
        "source_identity": _source_identity([fsdp_path, schedule_path, combined_path, test_path]),
        "duration_seconds": time.perf_counter() - started,
    }


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ModelTree:
    def __init__(self, has_moe: bool, marker: object):
        self.has_moe = has_moe
        self.marker = marker

    def traverse(self, _kind):
        return iter([self.marker] if self.has_moe else [])


def _titan_config(
    *,
    dp: int = 1,
    ep: int = 1,
    tp: int = 1,
    cp: int = 1,
    overlap: bool = False,
    cudagraph: bool = False,
    has_moe: bool = False,
    marker: object,
) -> Any:
    model_spec = (
        SimpleNamespace(model=_ModelTree(has_moe=has_moe, marker=marker)) if has_moe else None
    )
    return SimpleNamespace(
        parallelism=SimpleNamespace(
            data_parallel_shard_degree=dp,
            expert_parallel_degree=ep,
            tensor_parallel_degree=tp,
            context_parallel_degree=cp,
        ),
        compile=SimpleNamespace(
            enable_passes=cudagraph,
            disable_passes=[],
            enable_fsdp_ag_rs_overlap=overlap,
        ),
        model_spec=model_spec,
    )


def _run_titan_main(module: Any, config: Any, existing: str | None) -> dict[str, Any]:
    config_module = types.ModuleType("torchtitan.config")

    class ConfigManager:
        def parse_args(self):
            return config

    config_module.ConfigManager = ConfigManager
    previous = sys.modules.get("torchtitan.config")
    sys.modules["torchtitan.config"] = config_module
    old_value = os.environ.get("GPU_MAX_HW_QUEUES")
    if existing is None:
        os.environ.pop("GPU_MAX_HW_QUEUES", None)
    else:
        os.environ["GPU_MAX_HW_QUEUES"] = existing
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            module.main()
    finally:
        if old_value is None:
            os.environ.pop("GPU_MAX_HW_QUEUES", None)
        else:
            os.environ["GPU_MAX_HW_QUEUES"] = old_value
        if previous is None:
            sys.modules.pop("torchtitan.config", None)
        else:
            sys.modules["torchtitan.config"] = previous
    return {"stdout": stdout.getvalue(), "stderr": stderr.getvalue()}


def _torchtitan_probe(head_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    helper_path = head_root / "torchtitan/experiments/graph_trainer/hw_queues.py"
    test_path = head_root / "torchtitan/experiments/graph_trainer/tests/test_hw_queues.py"
    module = _load_module(helper_path, "infraswe_r7_torchtitan_hw_queues")

    marker = object()
    moe_module = types.ModuleType("torchtitan.models.common.moe")
    moe_module.MoE = SimpleNamespace(Config=type(marker))
    module_names = [
        "torchtitan",
        "torchtitan.models",
        "torchtitan.models.common",
        "torchtitan.models.common.moe",
    ]
    previous_modules = {name: sys.modules.get(name) for name in module_names}
    for name in module_names[:-1]:
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.modules[module_names[-1]] = moe_module
    try:
        matrix: list[dict[str, Any]] = []
        for dp in (1, 2, -1):
            for ep in (1, 2):
                for tp in (1, 2):
                    for cp in (1, 2):
                        for overlap in (False, True):
                            for cudagraph in (False, True):
                                for has_moe in (False, True):
                                    config = _titan_config(
                                        dp=dp,
                                        ep=ep,
                                        tp=tp,
                                        cp=cp,
                                        overlap=overlap,
                                        cudagraph=cudagraph,
                                        has_moe=has_moe,
                                        marker=marker,
                                    )
                                    queues, lanes = module.recommend_gpu_max_hw_queues(config)
                                    expected = 1 << (len(lanes) - 1).bit_length()
                                    matrix.append(
                                        {
                                            "inputs": [dp, ep, tp, cp, overlap, cudagraph, has_moe],
                                            "queues": queues,
                                            "lane_count": len(lanes),
                                            "matches_pow2_oracle": queues == expected,
                                        }
                                    )
        plain = _titan_config(marker=marker)
        wider = _titan_config(
            dp=2,
            ep=2,
            tp=2,
            cp=2,
            overlap=True,
            cudagraph=True,
            has_moe=True,
            marker=marker,
        )
        no_existing = _run_titan_main(module, plain, None)
        existing = _run_titan_main(module, plain, "64")
        second_launch = _run_titan_main(module, wider, None)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    helper = helper_path.read_text(encoding="utf-8")
    tests = test_path.read_text(encoding="utf-8")
    pytest_started = time.perf_counter()
    pytest_env = os.environ.copy()
    prior_pythonpath = pytest_env.get("PYTHONPATH")
    pytest_env["PYTHONPATH"] = (
        f"{head_root}{os.pathsep}{prior_pythonpath}" if prior_pythonpath else str(head_root)
    )
    pytest_run = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(test_path)],
        cwd=head_root,
        env=pytest_env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "status": "pass",
        "failure_codes": [],
        "facts": {
            "recommendation_matrix_cases": len(matrix),
            "recommendation_matrix_all_pass": all(item["matches_pow2_oracle"] for item in matrix),
            "main_without_existing_value": no_existing,
            "main_with_existing_value": existing,
            "existing_value_would_be_overwritten_by_eval": (
                "already set to 64" in existing["stderr"]
                and "export GPU_MAX_HW_QUEUES=" in existing["stdout"]
            ),
            "second_launch_recomputes_from_config": (
                no_existing["stdout"] != second_launch["stdout"]
            ),
            "helper_has_backend_parameter_or_rocm_detection": any(
                token in helper for token in ("torch.version.hip", "is_rocm", "backend=")
            ),
            "cuda_cpu_strict_noop": False,
            "missing_or_malformed_tool_branch": False,
            "helper_mutates_environment_itself": 'os.environ["GPU_MAX_HW_QUEUES"] =' in helper,
            "direct_test_count": tests.count("    def test_"),
            "direct_tests_cover_main": "hw_queues.main" in tests,
            "direct_tests_cover_existing_override": "GPU_MAX_HW_QUEUES" in tests,
            "direct_repo_test": {
                "return_code": pytest_run.returncode,
                "duration_seconds": time.perf_counter() - pytest_started,
                "stdout_sha256": canonical_sha256(pytest_run.stdout),
                "stderr_sha256": canonical_sha256(pytest_run.stderr),
                "stdout_tail": pytest_run.stdout[-2000:],
                "stderr_tail": pytest_run.stderr[-2000:],
            },
            "real_rocm_scheduling_impact": "unresolved-no-rocm-gpu",
            "steady_state_compile_seconds": 0.0,
        },
        "source_identity": _source_identity([helper_path, test_path]),
        "duration_seconds": time.perf_counter() - started,
    }


def _extract_model_assignments(path: Path) -> Callable[..., Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    run_node: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TaskRunner":
            run_node = next(
                (
                    child
                    for child in node.body
                    if isinstance(child, ast.FunctionDef) and child.name == "run"
                ),
                None,
            )
    if run_node is None:
        raise ValueError("missing TaskRunner.run")
    wanted: list[ast.stmt] = []
    targets = {"model_config", "tokenizer", "processor"}
    for statement in run_node.body:
        target_name: str | None = None
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            target_name = statement.target.id
        elif (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            target_name = statement.targets[0].id
        if target_name in targets:
            wanted.append(statement)
    if len(wanted) != 3:
        raise ValueError(f"expected three model assignments, found {len(wanted)}")
    function = ast.FunctionDef(
        name="run_model_init",
        args=ast.arguments(
            posonlyargs=[],
            args=[ast.arg(arg="config")],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
        ),
        body=[
            *wanted,
            ast.Return(
                value=ast.Tuple(
                    [
                        ast.Name("tokenizer", ctx=ast.Load()),
                        ast.Name("processor", ctx=ast.Load()),
                    ],
                    ctx=ast.Load(),
                )
            ),
        ],
        decorator_list=[],
    )
    function = _StripAnnotations().visit(function)
    ast.fix_missing_locations(function)
    namespace: dict[str, Any] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["run_model_init"]


class _Converter:
    def __init__(self, outputs: list[Any]):
        self.outputs = iter(outputs)
        self.calls: list[Any] = []

    def __call__(self, raw: Any):
        self.calls.append(raw)
        return next(self.outputs)


def _model_config_semantics(model_config_path: Path) -> dict[str, Any]:
    source = model_config_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "HFModelConfig"
    )
    field_names = {
        node.target.id
        for node in model_class.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    }
    post_init = next(
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "__post_init__"
    )
    calls: list[dict[str, Any]] = []
    for node in ast.walk(post_init):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr
            if name in {"copy_to_local", "hf_tokenizer", "hf_processor"}:
                calls.append(
                    {
                        "name": name,
                        "keywords": sorted(keyword.arg or "**" for keyword in node.keywords),
                        "source": ast.unparse(node),
                    }
                )
    return {
        "fields": sorted(field_names),
        "has_path": "path" in field_names,
        "has_tokenizer_path": "tokenizer_path" in field_names,
        "has_trust_remote_code": "trust_remote_code" in field_names,
        "has_use_shm": "use_shm" in field_names,
        "has_revision": "revision" in field_names,
        "has_processor_kwargs": "processor_kwargs" in field_names,
        "loader_calls": calls,
        "processor_use_fast_explicit": any(
            call["name"] == "hf_processor" and "use_fast" in call["keywords"] for call in calls
        ),
    }


def _verl_probe(base_root: Path, head_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    main_path = head_root / "verl/trainer/main_ppo_v0.py"
    v1_path = head_root / "verl/trainer/ppo/v1/trainer_base.py"
    base_main_path = base_root / "verl/trainer/main_ppo_v0.py"
    base_v1_path = base_root / "verl/trainer/ppo/v1/trainer_base.py"
    model_config_path = head_root / "verl/workers/config/model.py"

    main_init = _extract_model_assignments(main_path)
    v1_init = _function_from_class(
        v1_path,
        "PPOTrainer",
        "_init_tokenizer",
        {},
    )
    raw = object()
    config = SimpleNamespace(actor_rollout_ref=SimpleNamespace(model=raw))

    text_model = SimpleNamespace(tokenizer="text-tokenizer", processor=None)
    multi_model = SimpleNamespace(tokenizer="multi-tokenizer", processor="vision-processor")
    main_converter = _Converter([text_model, multi_model])
    main_init.__globals__.update(
        omega_conf_to_dataclass=main_converter,
        HFModelConfig=SimpleNamespace,
    )
    main_text = main_init(config)
    main_multi = main_init(config)

    v1_converter = _Converter([text_model, multi_model])
    v1_init.__globals__.update(
        omega_conf_to_dataclass=v1_converter,
        HFModelConfig=SimpleNamespace,
    )
    trainer = SimpleNamespace(config=config)
    v1_init(trainer)
    v1_text = (trainer.tokenizer, trainer.processor)
    v1_init(trainer)
    v1_multi = (trainer.tokenizer, trainer.processor)

    base_sources = base_main_path.read_text(encoding="utf-8") + base_v1_path.read_text(
        encoding="utf-8"
    )
    head_sources = main_path.read_text(encoding="utf-8") + v1_path.read_text(encoding="utf-8")
    config_semantics = _model_config_semantics(model_config_path)
    return {
        "status": "pass",
        "failure_codes": [],
        "facts": {
            "main_entry": {
                "text_result": list(main_text),
                "multimodal_result": list(main_multi),
                "conversion_call_count": len(main_converter.calls),
                "exact_model_config_object_forwarded": all(
                    item is raw for item in main_converter.calls
                ),
            },
            "v1_entry": {
                "text_result": list(v1_text),
                "multimodal_result": list(v1_multi),
                "conversion_call_count": len(v1_converter.calls),
                "exact_model_config_object_forwarded": all(
                    item is raw for item in v1_converter.calls
                ),
            },
            "both_entrypoints_match_text_and_multimodal": (
                main_text == v1_text and main_multi == v1_multi
            ),
            "repeated_init_replaces_stale_state": v1_text != v1_multi,
            "hf_model_config_semantics": config_semantics,
            "base_reads_data_trust_remote_code": 'config.data.get("trust_remote_code", False)'
            in base_sources,
            "head_reads_data_trust_remote_code": 'config.data.get("trust_remote_code", False)'
            in head_sources,
            "head_has_legacy_trust_fallback_or_warning": (
                "legacy" in head_sources.lower() and "trust_remote_code" in head_sources
            ),
            "direct_two_entrypoint_tests_in_changed_paths": False,
            "steady_state_compile_seconds": 0.0,
        },
        "source_identity": _source_identity(
            [main_path, v1_path, model_config_path, base_main_path, base_v1_path]
        ),
        "duration_seconds": time.perf_counter() - started,
    }


def _payload(
    case_id: str,
    project: str,
    probe: str,
    result: dict[str, Any],
    selection_sha: str,
    plan_sha: str,
    base_sha: str,
    head_sha: str,
) -> dict[str, Any]:
    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r7",
        "probe": probe,
        "case_id": case_id,
        "project": project,
        **result,
        "selection_lock_sha256": selection_sha,
        "test_plan_sha256": plan_sha,
        "base_sha": base_sha,
        "head_sha": head_sha,
        "created_at": datetime.now(UTC).isoformat(),
    }
    return {**material, "evidence_sha256": canonical_sha256(material)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worktree-root", type=Path, required=True)
    parser.add_argument("--selection-lock", type=Path, required=True)
    parser.add_argument("--test-plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    selection = _read(args.selection_lock)
    plan = _read(args.test_plan)
    selection_sha, plan_sha = _verify_locks(selection, plan)
    cases = {item["case_id"]: item for item in selection["selection_material"]["cases"]}
    results = {
        "megatron-pr-6174": (
            "megatron-core",
            "r7-megatron-fsdp-schedule-contract-v1",
            _megatron_probe(args.worktree_root / "megatron-head"),
        ),
        "torchtitan-pr-4032": (
            "torchtitan",
            "r7-torchtitan-hw-queues-contract-v1",
            _torchtitan_probe(args.worktree_root / "torchtitan-head"),
        ),
        "verl-pr-7220": (
            "verl",
            "r7-verl-hf-model-config-contract-v1",
            _verl_probe(args.worktree_root / "verl-base", args.worktree_root / "verl-head"),
        ),
    }
    for case_id, (project, probe, result) in results.items():
        case = cases[case_id]
        payload = _payload(
            case_id,
            project,
            probe,
            result,
            selection_sha,
            plan_sha,
            case["base_sha"],
            case["head_sha"],
        )
        atomic_write_json(args.output_root / f"{case_id}.json", payload)
        print(json.dumps({"case_id": case_id, "evidence_sha256": payload["evidence_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

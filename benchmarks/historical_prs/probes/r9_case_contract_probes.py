#!/usr/bin/env python3
"""Run exact-source, case-specific R9 base/head contract probes."""

from __future__ import annotations

import argparse
import ast
import base64
import copy
import functools
import hashlib
import json
import platform
import re
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, TypeVar, cast


def _canonical(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _gh(endpoint: str, *, paginate: bool = False) -> Any:
    command = ["gh", "api"]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    command.append(endpoint)
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(4):
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        if process.returncode == 0:
            break
        if attempt < 3:
            import time

            time.sleep(2**attempt)
    assert process is not None
    if process.returncode != 0:
        raise RuntimeError(f"GitHub API failed for {endpoint}: {process.stderr.strip()}")
    value = json.loads(process.stdout)
    if paginate:
        value = [item for page in value for item in page]
    return value


def _content(repository: str, path: str, revision: str) -> str | None:
    from urllib.parse import quote

    endpoint = f"repos/{repository}/contents/{quote(path)}?ref={revision}"
    try:
        payload = _gh(endpoint)
    except RuntimeError as error:
        if "HTTP 404" in str(error):
            return None
        raise
    if payload.get("encoding") != "base64":
        raise RuntimeError(f"unsupported content encoding for {repository}:{path}")
    return base64.b64decode(payload["content"]).decode("utf-8")


def _acquire(cases: list[dict[str, Any]]) -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    for case in cases:
        files = _gh(
            f"repos/{case['repository']}/pulls/{case['pull_number']}/files?per_page=100",
            paginate=True,
        )
        observed = sorted(item["filename"] for item in files)
        expected = sorted(case["paths"])
        if observed != expected:
            raise RuntimeError(
                f"path parity failed for {case['case_id']}: {observed} != {expected}"
            )
        sources: dict[str, dict[str, str | None]] = {}
        status_by_path = {item["filename"]: item["status"] for item in files}
        for path in case["paths"]:
            if not path.endswith((".py", ".pyi")):
                continue
            sources[path] = {
                "base": (
                    None
                    if status_by_path[path] == "added"
                    else _content(case["repository"], path, case["base_sha"])
                ),
                "head": (
                    None
                    if status_by_path[path] == "removed"
                    else _content(case["repository"], path, case["head_sha"])
                ),
            }
        bundle[case["case_id"]] = {
            "case": case,
            "files": [
                {
                    "filename": item["filename"],
                    "status": item["status"],
                    "additions": item["additions"],
                    "deletions": item["deletions"],
                    "patch": item.get("patch"),
                }
                for item in files
            ],
            "sources": sources,
        }
    return bundle


def _function(source: str, name: str) -> ast.FunctionDef | ast.AsyncFunctionDef:
    tree = ast.parse(source)
    matches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected exactly one {name}, found {len(matches)}")
    node = copy.deepcopy(matches[0])
    node.decorator_list = []
    return node


def _exec_function(source: str, name: str, namespace: dict[str, Any]) -> Any:
    node = _function(source, name)
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, f"<{name}>", "exec"), namespace)
    return namespace[name]


def _source_pair(item: dict[str, Any], suffix: str) -> tuple[str, str]:
    matches = [value for path, value in item["sources"].items() if path.endswith(suffix)]
    if len(matches) != 1 or matches[0]["base"] is None or matches[0]["head"] is None:
        raise AssertionError(f"missing unique base/head source for {suffix}")
    return str(matches[0]["base"]), str(matches[0]["head"])


def _test_source(item: dict[str, Any]) -> str:
    matches = [
        value["head"]
        for path, value in item["sources"].items()
        if "test" in path.lower() and value["head"] is not None
    ]
    if len(matches) != 1:
        raise AssertionError("expected exactly one changed Python test source")
    return str(matches[0])


def _probe_cutlass(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "cutlass/base_dsl/typing.py")
    assert "def _lowered_const_value" not in base
    lowered = _exec_function(head, "_lowered_const_value", {"Any": Any})

    def operand(value: Any, *, width: int = 64, signed: bool = False) -> Any:
        numeric_type = type("Signed" if signed else "Unsigned", (), {"signed": signed})
        result = numeric_type()
        result.value = value
        result.dtype = SimpleNamespace(width=width)
        return result

    target_values = [2**63, 2**63 + 12345, 2**64 - 1]
    target_outputs = [lowered(operand(value)) for value in target_values]
    expected_outputs = [value - 2**64 for value in target_values]
    assert target_outputs == expected_outputs
    neighbors = [0, 1, 2**63 - 1]
    assert [lowered(operand(value)) for value in neighbors] == neighbors
    assert lowered(operand(-1, signed=True)) == -1
    assert lowered(operand(2**63 - 1, signed=True)) == 2**63 - 1
    assert lowered(operand(2**127, width=128)) == 2**127
    assert lowered(operand(True)) is True
    test_source = _test_source(item)
    for token in ("2**63", "2**64 - 1", "Uint32", "Int64", "binary_ops"):
        assert token in test_source
    return {
        "base_target_helper_present": False,
        "head_target_helper_present": True,
        "target_values": target_values,
        "target_outputs": target_outputs,
        "neighbor_values_preserved": True,
        "signed_bool_and_128bit_controls_preserved": True,
        "direct_test_covers_conversion_and_binary_paths": True,
        "full_cutlass_dsl_compile_executed": False,
    }


def _probe_sglang_validation(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "sglang/srt/arg_groups/overrides.py")
    assert "def _check_tilelang_dsa_fp8_kv" not in base
    check = _exec_function(head, "_check_tilelang_dsa_fp8_kv", {"Optional": Optional})
    backends = [None, "tilelang", "flashmla_kv", "trtllm"]
    dtypes = ["fp8_e4m3", "bfloat16", "auto"]
    trials = 0
    rejected = 0
    for hip in (False, True):
        for dtype in dtypes:
            for prefill in backends:
                for decode in backends:
                    expected_reject = (
                        not hip and dtype == "fp8_e4m3" and "tilelang" in {prefill, decode}
                    )
                    try:
                        check(dtype, prefill, decode, hip=hip)
                        observed_reject = False
                    except ValueError as error:
                        observed_reject = True
                        assert "ROCm/HIP" in str(error) and "bfloat16" in str(error)
                    assert observed_reject == expected_reject
                    trials += 1
                    rejected += observed_reject
    resolution = _function(head, "_dsa_split_backend_resolution")
    calls = [
        node
        for node in ast.walk(resolution)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_check_tilelang_dsa_fp8_kv"
    ]
    assert len(calls) == 1
    test_source = _test_source(item)
    test_methods = [
        node
        for node in ast.walk(ast.parse(test_source))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    assert len(test_methods) == 5
    return {
        "base_validator_present": False,
        "head_validator_present": True,
        "truth_table_cases": trials,
        "truth_table_rejections": rejected,
        "exact_truth_table_pass": True,
        "normal_resolution_lifecycle_call_count": len(calls),
        "direct_test_methods": 5,
    }


def _probe_sglang_plan(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "attention/flashinfer_backend.py")
    name = "_narrow_deterministic_cuda_graph_decode_plan"
    assert f"def {name}" not in base
    tree = ast.parse(head)
    prefixes = ("_FLASHINFER_PREFILL_",)
    assignments = [
        copy.deepcopy(node)
        for node in tree.body
        if isinstance(node, (ast.Assign, ast.AnnAssign))
        and any(
            isinstance(target, ast.Name) and target.id.startswith(prefixes)
            for target in (node.targets if isinstance(node, ast.Assign) else [node.target])
        )
    ]
    node = _function(head, name)
    module = ast.fix_missing_locations(ast.Module(body=[*assignments, node], type_ignores=[]))
    namespace = {
        "Sequence": Sequence,
        "TypeVar": TypeVar,
        "cast": cast,
        "_PlanInfoT": TypeVar("_PlanInfoT", bound=Sequence[int]),
    }
    exec(compile(module, f"<{name}>", "exec"), namespace)
    narrow = namespace[name]

    def plan(*, padded: int, rows: int, tile: int, graph: int = 1, split: int = 0) -> list[int]:
        values = [0] * 15
        values[0] = padded
        values[1] = rows
        values[3] = tile
        values[13] = graph
        values[14] = split
        return values

    matrix = 0
    for batch in (1, 2, 3, 7, 16, 33, 64):
        for qo, kv in ((8, 8), (16, 2), (32, 1), (64, 2)):
            for tile in (1, 8, 16, 32):
                exact = batch * ((qo // kv + tile - 1) // tile)
                original = plan(padded=exact + 11, rows=batch, tile=tile)
                first = narrow(original, batch_size=batch, num_qo_heads=qo, num_kv_heads=kv)
                second = narrow(original, batch_size=batch, num_qo_heads=qo, num_kv_heads=kv)
                assert first[0] == exact and first[1:] == original[1:]
                assert first == second and original[0] == exact + 11
                matrix += 1
    equal = plan(padded=2, rows=2, tile=16)
    assert narrow(equal, batch_size=2, num_qo_heads=16, num_kv_heads=2) is equal
    invalid_calls = [
        (plan(padded=4, rows=2, tile=16, graph=0), 2, 16, 2),
        (plan(padded=4, rows=2, tile=16, split=1), 2, 16, 2),
        (plan(padded=4, rows=3, tile=16), 2, 16, 2),
        (plan(padded=1, rows=2, tile=16), 2, 16, 2),
        (plan(padded=4, rows=2, tile=0), 2, 16, 2),
        (plan(padded=4, rows=2, tile=16), 0, 16, 2),
        (plan(padded=4, rows=2, tile=16), 2, 10, 3),
    ]
    for value, batch, qo, kv in invalid_calls:
        try:
            narrow(value, batch_size=batch, num_qo_heads=qo, num_kv_heads=kv)
        except RuntimeError:
            pass
        else:
            raise AssertionError("invalid plan was not rejected")
    test_source = _test_source(item)
    test_methods = [
        node
        for node in ast.walk(ast.parse(test_source))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    ]
    integration = head[head.index("if (\n            disable_split_kv is True") :]
    assert "wrapper.is_cuda_graph_enabled" in integration
    assert "wrapper.use_tensor_cores" in integration
    assert 'getattr(wrapper, "_backend", None) == "fa2"' in integration
    return {
        "base_narrow_helper_present": False,
        "head_narrow_helper_present": True,
        "boundary_matrix_cases": matrix,
        "boundary_matrix_pass": True,
        "invalid_controls_rejected": len(invalid_calls),
        "same_bound_returns_original": True,
        "input_not_mutated": True,
        "integration_guard_has_graph_tensorcore_fa2_splitkv_conditions": True,
        "direct_test_methods": len(test_methods),
        "actual_cuda_graph_replay_executed": False,
    }


def _probe_vllm(item: dict[str, Any]) -> dict[str, Any]:
    base, head = _source_pair(item, "vllm/utils/flashinfer.py")
    name = "has_flashinfer_cutlass_fused_moe_fp4"
    assert f"def {name}" not in base
    function = _function(head, name)
    function.decorator_list = []
    namespace: dict[str, Any] = {
        "functools": functools,
        "re": re,
        "has_flashinfer_cutlass_fused_moe": lambda: True,
        "has_flashinfer_cubin": lambda: False,
    }
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    exec(compile(module, f"<{name}>", "exec"), namespace)
    raw = namespace[name]

    calls: list[dict[str, Any]] = []

    def run_scenario(
        *,
        base_gate: bool = True,
        cubin: bool = False,
        jit_cache: bool = False,
        nvcc: str | None = None,
        version: str = "Cuda compilation tools, release 12.8, V12.8.0",
        error: Exception | None = None,
    ) -> bool:
        namespace["has_flashinfer_cutlass_fused_moe"] = lambda: base_gate
        namespace["has_flashinfer_cubin"] = lambda: cubin
        namespace["importlib"] = SimpleNamespace(
            util=SimpleNamespace(find_spec=lambda module: object() if jit_cache else None)
        )
        namespace["shutil"] = SimpleNamespace(which=lambda binary: nvcc)

        def fake_run(*args: Any, **kwargs: Any) -> Any:
            calls.append({"args": args, "kwargs": kwargs})
            if error is not None:
                raise error
            return SimpleNamespace(stdout=version)

        namespace["subprocess"] = SimpleNamespace(
            run=fake_run,
            SubprocessError=subprocess.SubprocessError,
        )
        return bool(raw())

    scenarios = {
        "base_capability_missing": run_scenario(base_gate=False),
        "generic_cubin_present": run_scenario(cubin=True),
        "generic_jit_cache_present": run_scenario(jit_cache=True),
        "no_artifact_or_nvcc": run_scenario(),
        "nvcc_12_7": run_scenario(nvcc="/nvcc", version="release 12.7, V12.7.9"),
        "nvcc_12_8": run_scenario(nvcc="/nvcc", version="release 12.8, V12.8.0"),
        "nvcc_13_0": run_scenario(nvcc="/nvcc", version="release 13.0, V13.0.0"),
        "malformed_nvcc": run_scenario(nvcc="/nvcc", version="unknown"),
        "nvcc_oserror": run_scenario(nvcc="/nvcc", error=OSError("missing")),
        "nvcc_timeout": run_scenario(nvcc="/nvcc", error=subprocess.TimeoutExpired("nvcc", 5)),
    }
    expected = {
        "base_capability_missing": False,
        "generic_cubin_present": True,
        "generic_jit_cache_present": True,
        "no_artifact_or_nvcc": False,
        "nvcc_12_7": False,
        "nvcc_12_8": True,
        "nvcc_13_0": True,
        "malformed_nvcc": False,
        "nvcc_oserror": False,
        "nvcc_timeout": False,
    }
    assert scenarios == expected
    assert all(
        call["kwargs"].get("timeout") == 5 and call["kwargs"].get("check") is True for call in calls
    )
    gate_base, gate_head = _source_pair(item, "flashinfer_cutlass_moe.py")
    assert name not in gate_base
    assert gate_head.count(f"{name}()") == 2
    test_source = _test_source(item)
    test_names = {
        node.name
        for node in ast.walk(ast.parse(test_source))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test_")
    }
    return {
        "base_fp4_specific_gate_present": False,
        "head_fp4_specific_gate_present": True,
        "scenario_results": scenarios,
        "scenario_matrix_pass": True,
        "nvcc_calls_use_check_and_timeout": True,
        "quant_scheme_gate_call_count": 2,
        "direct_test_methods": sorted(test_names),
        "direct_test_covers_base_capability_missing": False,
        "direct_test_covers_jit_cache_branch": False,
        "direct_test_covers_malformed_or_failed_nvcc": False,
        "generic_cubin_presence_proves_specific_fp4_kernel": False,
        "generic_jit_cache_presence_proves_specific_fp4_kernel": False,
    }


class _FakeSymmDeviceMemory:
    def __init__(self) -> None:
        self.mapped = False

    def _create_and_map_handles(self, backend: Any) -> None:
        del backend
        self.mapped = True


def _probe_flashinfer(item: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as error:
        return {"torch_available": False, "unresolved_reason": str(error)}
    base, head = _source_pair(item, "flashinfer/comm/trtllm_mnnvl_ar.py")

    def build(source: str) -> type:
        init = _function(source, "_initialize_protocol")
        restore = _function(source, "checkpoint_restore")
        namespace = {
            "torch": torch,
            "CommBackend": object,
            "SymmDeviceMemory": _FakeSymmDeviceMemory,
        }
        module = ast.fix_missing_locations(ast.Module(body=[init, restore], type_ignores=[]))
        exec(compile(module, "<mnnvl-methods>", "exec"), namespace)
        return type(
            "Workspace",
            (),
            {
                "_initialize_protocol": namespace["_initialize_protocol"],
                "checkpoint_restore": namespace["checkpoint_restore"],
            },
        )

    base_class = build(base)
    head_class = build(head)

    def workspace(cls: type, *, inference_tensor: bool) -> tuple[Any, Any]:
        instance = cls()
        memory = _FakeSymmDeviceMemory()
        handle = SimpleNamespace(
            mcast_device_memory=memory,
            lamport_initialize=lambda rank, dtype: None,
        )
        instance.handle = handle
        instance.rank = 0
        instance.buffer_size_bytes = 1024
        if inference_tensor:
            with torch.inference_mode():
                instance.buffer_flags = torch.ones(9, dtype=torch.uint32)
        else:
            instance.buffer_flags = torch.ones(9, dtype=torch.uint32)
        return instance, memory

    original_sync = torch.cuda.synchronize
    torch.cuda.synchronize = lambda: None
    try:
        base_workspace, base_memory = workspace(base_class, inference_tensor=True)
        try:
            base_workspace.checkpoint_restore(SimpleNamespace(barrier=lambda: None))
        except RuntimeError as error:
            base_failed = "inference tensor" in str(error).lower()
        else:
            base_failed = False
        assert base_failed and base_memory.mapped

        barrier_calls = 0

        def barrier() -> None:
            nonlocal barrier_calls
            barrier_calls += 1

        head_workspace, head_memory = workspace(head_class, inference_tensor=True)
        head_workspace.checkpoint_restore(SimpleNamespace(barrier=barrier))
        expected = [0, 2, 1024, 0, 0, 0, 0, 0, 0]
        assert head_workspace.buffer_flags.tolist() == expected
        assert head_memory.mapped and barrier_calls == 1
        assert not torch.is_inference_mode_enabled()
        head_workspace.checkpoint_restore(SimpleNamespace(barrier=barrier))
        assert barrier_calls == 1

        for cls in (base_class, head_class):
            normal, _ = workspace(cls, inference_tensor=False)
            normal._initialize_protocol()
            assert normal.buffer_flags.tolist() == expected
    finally:
        torch.cuda.synchronize = original_sync
    test_source = _test_source(item)
    assert "torch.is_inference" in test_source
    assert "workspace._initialize_protocol()" in test_source
    return {
        "torch_available": True,
        "torch_version": torch.__version__,
        "base_checkpoint_restore_reproduces_inference_tensor_failure": True,
        "head_checkpoint_restore_passes": True,
        "head_flags_match_protocol": True,
        "ordinary_tensor_base_and_head_pass": True,
        "repeated_restore_is_noop": True,
        "global_inference_mode_restored": True,
        "direct_new_test_calls_checkpoint_restore": False,
        "direct_new_test_calls_initialize_protocol": True,
        "real_two_gpu_mnnvl_transport_executed": False,
    }


PROBES = {
    "cutlass-pr-3332": _probe_cutlass,
    "flashinfer-pr-3950": _probe_flashinfer,
    "sglang-pr-31346": _probe_sglang_validation,
    "sglang-pr-31349": _probe_sglang_plan,
    "vllm-pr-48695": _probe_vllm,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--bundle-output", type=Path)
    parser.add_argument("--case", choices=sorted(PROBES), action="append")
    args = parser.parse_args()

    selection = _read(args.selection)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != _canonical(selection_material):
        raise SystemExit("R9 selection digest mismatch")
    plan = _read(args.plan)
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != _canonical(plan_material):
        raise SystemExit("R9 plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R9 plan/selection binding mismatch")
    case_by_id = {item["case_id"]: item for item in selection_material["cases"]}
    selected_ids = args.case or sorted(PROBES)
    if not set(selected_ids) <= set(case_by_id):
        raise SystemExit("requested probe case is not selected")

    if args.source_bundle:
        bundle = _read(args.source_bundle)
    else:
        bundle = _acquire([case_by_id[case_id] for case_id in selected_ids])
    if args.bundle_output:
        args.bundle_output.parent.mkdir(parents=True, exist_ok=True)
        args.bundle_output.write_text(json.dumps(bundle), encoding="utf-8")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = 0
    for case_id in selected_ids:
        item = bundle[case_id]
        source_digests = {
            f"{revision}:{path}": _canonical(source)
            for path, revisions in item["sources"].items()
            for revision, source in revisions.items()
            if source is not None
        }
        try:
            facts = PROBES[case_id](item)
            unresolved = bool(facts.get("unresolved_reason"))
            status = "unresolved" if unresolved else "pass"
            failure_codes = ["R9_REQUIRED_RUNTIME_UNAVAILABLE"] if unresolved else []
        except Exception as error:
            facts = {"exception_type": type(error).__name__, "exception": str(error)}
            status = "fail"
            failure_codes = ["R9_CASE_CONTRACT_PROBE_FAILED"]
            failures += 1
        material = {
            "schema_version": "0.1",
            "protocol_id": selection_material["protocol_id"],
            "case_id": case_id,
            "selection_lock_sha256": selection["selection_lock_sha256"],
            "test_plan_sha256": plan["test_plan_sha256"],
            "base_sha": case_by_id[case_id]["base_sha"],
            "head_sha": case_by_id[case_id]["head_sha"],
            "source_digests": source_digests,
            "path_parity": True,
            "status": status,
            "failure_codes": failure_codes,
            "facts": facts,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
            },
            "observed_at": datetime.now(UTC).isoformat(),
        }
        payload = {**material, "evidence_sha256": _canonical(material)}
        path = args.output_dir / f"{case_id}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{case_id}: {status} {payload['evidence_sha256']}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

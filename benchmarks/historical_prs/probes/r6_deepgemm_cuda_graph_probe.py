#!/usr/bin/env python3
"""H100 CUDA-graph probe for DeepGEMM PR 55 map_ctype changes."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.util
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any

import torch


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _map_result(module: ModuleType, value: Any) -> dict[str, Any]:
    try:
        mapped = module.map_ctype(value)
    except Exception as exc:
        return {
            "status": "rejected",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    raw_value = mapped.value if isinstance(mapped, ctypes._SimpleCData) else None
    return {
        "status": "mapped",
        "ctype": type(mapped).__name__,
        "value": raw_value,
    }


def _cuda_graph_case(module: ModuleType, tensor: torch.Tensor) -> dict[str, Any]:
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor.add_(1)
    torch.cuda.synchronize()
    capture_values: dict[str, Any] = {}
    try:
        with torch.cuda.graph(graph):
            capture_values["tensor"] = module.map_ctype(tensor).value
            capture_values["stream"] = module.map_ctype(torch.cuda.current_stream()).value
            tensor.mul_(2)
        capture_output = tensor.detach().clone()
        graph.replay()
        torch.cuda.synchronize()
        replay_output = tensor.detach().clone()
    except Exception as exc:
        return {
            "status": "fail",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    return {
        "status": "pass",
        "capture_values": capture_values,
        "tensor_pointer_matches": capture_values["tensor"] == tensor.data_ptr(),
        "capture_output": [float(item) for item in capture_output.cpu().tolist()],
        "replay_output": [float(item) for item in replay_output.cpu().tolist()],
        "replay_changed_output": not torch.equal(capture_output, replay_output),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-template", type=Path, required=True)
    parser.add_argument("--head-template", type=Path, required=True)
    parser.add_argument("--base-runtime", type=Path, required=True)
    parser.add_argument("--head-runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    sources = {
        "base_template": args.base_template.read_text(encoding="utf-8"),
        "head_template": args.head_template.read_text(encoding="utf-8"),
        "base_runtime": args.base_runtime.read_text(encoding="utf-8"),
        "head_runtime": args.head_runtime.read_text(encoding="utf-8"),
    }
    source_contract = {
        "runtime_source_unchanged": sources["base_runtime"] == sources["head_runtime"],
        "base_uses_tensor_isinstance": "isinstance(value, torch.Tensor)"
        in sources["base_template"],
        "head_uses_data_ptr_capability": "hasattr(value, 'data_ptr')" in sources["head_template"],
        "head_has_float16_pointer_branch": "value.dtype == torch.float16"
        in sources["head_template"],
        "head_generator_supports_float16": "torch.float16: ('void*'" in sources["head_template"],
        "runtime_retains_tensor_dtype_assertion": (
            "assert arg.dtype == dtype" in sources["head_runtime"]
        ),
    }
    if not all(
        (
            source_contract["runtime_source_unchanged"],
            source_contract["base_uses_tensor_isinstance"],
            source_contract["head_uses_data_ptr_capability"],
            source_contract["head_has_float16_pointer_branch"],
            source_contract["runtime_retains_tensor_dtype_assertion"],
        )
    ):
        raise ValueError(f"unexpected exact-source contract: {source_contract}")

    base = _load(args.base_template, "deepgemm_r6_base_template")
    head = _load(args.head_template, "deepgemm_r6_head_template")
    tensors = {
        "float32": torch.empty(8, device="cuda", dtype=torch.float32),
        "bfloat16": torch.empty(8, device="cuda", dtype=torch.bfloat16),
        "float16": torch.empty(8, device="cuda", dtype=torch.float16),
        "float8_e4m3fn": torch.empty(8, device="cuda", dtype=torch.float8_e4m3fn),
        "int32": torch.empty(8, device="cuda", dtype=torch.int32),
        "int8": torch.empty(8, device="cuda", dtype=torch.int8),
    }
    mapping: dict[str, dict[str, Any]] = {}
    for name, tensor in tensors.items():
        mapping[name] = {
            "base": _map_result(base, tensor),
            "head": _map_result(head, tensor),
            "data_ptr": tensor.data_ptr(),
        }
    scalars = {
        "bool": True,
        "int": 7,
        "float": 0.25,
        "stream": torch.cuda.current_stream(),
    }
    scalar_mapping = {
        name: {"base": _map_result(base, value), "head": _map_result(head, value)}
        for name, value in scalars.items()
    }
    graph_input_base = torch.tensor([1.0, 2.0], device="cuda")
    graph_input_head = graph_input_base.detach().clone()
    graph_results = {
        "base": _cuda_graph_case(base, graph_input_base),
        "head": _cuda_graph_case(head, graph_input_head),
    }

    expected_common = ("float32", "bfloat16", "float8_e4m3fn", "int32")
    common_mapping_ok = all(
        mapping[name][variant]["status"] == "mapped"
        and mapping[name][variant]["value"] == mapping[name]["data_ptr"]
        for name in expected_common
        for variant in ("base", "head")
    )
    graph_ok = all(
        result["status"] == "pass"
        and result["tensor_pointer_matches"]
        and result["replay_changed_output"]
        for result in graph_results.values()
    )
    scalar_mapping_ok = all(
        result[variant]["status"] == "mapped"
        for result in scalar_mapping.values()
        for variant in ("base", "head")
    )
    failure_codes: list[str] = []
    if not common_mapping_ok:
        failure_codes.append("DEEPGEMM_COMMON_CTYPE_MAPPING_CHANGED")
    if not graph_ok:
        failure_codes.append("DEEPGEMM_CUDA_GRAPH_CAPTURE_OR_REPLAY_FAILED")
    if not scalar_mapping_ok:
        failure_codes.append("DEEPGEMM_SCALAR_OR_STREAM_MAPPING_CHANGED")

    material = {
        "schema_version": "0.5",
        "protocol_id": "historical-pr-blind-cross-project-v0.5-r6",
        "probe": "deepgemm-map-ctype-cuda-graph-h100-v1",
        "case_id": "deepgemm-pr-55",
        "status": "pass" if not failure_codes else "fail",
        "failure_codes": failure_codes,
        "facts": {
            "device": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_version": torch.__version__,
            "source_contract": source_contract,
            "tensor_mapping": mapping,
            "scalar_mapping": scalar_mapping,
            "cuda_graph": graph_results,
            "common_mapping_ok": common_mapping_ok,
            "scalar_mapping_ok": scalar_mapping_ok,
            "cuda_graph_capture_and_replay_ok": graph_ok,
            "head_accepts_float16_while_generator_lacks_float16": (
                mapping["float16"]["head"]["status"] == "mapped"
                and not source_contract["head_generator_supports_float16"]
            ),
            "head_accepts_unregistered_int8_pointer": (
                mapping["int8"]["head"]["status"] == "mapped"
            ),
            "compilation_path": "not-required",
            "steady_state_compile_seconds": 0.0,
        },
        "source_identity": {name: _digest(source) for name, source in sources.items()},
        "environment": {"cuda_version": torch.version.cuda},
        "duration_seconds": time.perf_counter() - started,
        "created_at": datetime.now(UTC).isoformat(),
    }
    payload = {**material, "evidence_sha256": _digest(material)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 1 if failure_codes else 0


if __name__ == "__main__":
    raise SystemExit(main())

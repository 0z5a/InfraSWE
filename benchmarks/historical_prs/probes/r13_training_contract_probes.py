#!/usr/bin/env python3
# ruff: noqa: E501
"""Acquire and probe exact base/head sources for the R13 training cohort."""

from __future__ import annotations

import argparse
import ast
import base64
import contextlib
import gc
import hashlib
import json
import platform
import re
import subprocess
import tempfile
import time
import weakref
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _canonical(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected object in {path}")
    return value


def _run_json(command: list[str]) -> tuple[Any, str]:
    process: subprocess.CompletedProcess[str] | None = None
    for attempt in range(4):
        process = subprocess.run(command, check=False, capture_output=True, text=True)
        if process.returncode == 0:
            break
        if attempt < 3:
            time.sleep(2**attempt)
    assert process is not None
    if process.returncode != 0:
        raise RuntimeError(f"command failed: {' '.join(command)}: {process.stderr.strip()}")
    return json.loads(process.stdout), process.stdout


def _gh(endpoint: str, *, paginate: bool = False) -> tuple[Any, str]:
    command = ["gh", "api"]
    if paginate:
        command.extend(["--paginate", "--slurp"])
    command.append(endpoint)
    value, raw = _run_json(command)
    if paginate:
        value = [item for page in value for item in page]
    return value, raw


def _candidate_projection(repository: str, pull_number: int) -> tuple[dict[str, str], str]:
    owner, name = repository.split("/", 1)
    query = """
      query($owner: String!, $name: String!, $number: Int!) {
        repository(owner: $owner, name: $name) {
          pullRequest(number: $number) { title body headRefOid }
        }
      }
    """
    payload, raw = _run_json(
        [
            "gh",
            "api",
            "graphql",
            "-f",
            f"query={query}",
            "-F",
            f"owner={owner}",
            "-F",
            f"name={name}",
            "-F",
            f"number={pull_number}",
        ]
    )
    projected = payload["data"]["repository"]["pullRequest"]
    if set(projected) != {"title", "body", "headRefOid"}:
        raise RuntimeError("candidate evidence projection returned an unexpected field")
    return {
        "title": str(projected["title"]),
        "body": str(projected["body"] or ""),
        "head_sha": str(projected["headRefOid"]),
    }, raw


def _content(repository: str, path: str, revision: str) -> str | None:
    from urllib.parse import quote

    endpoint = f"repos/{repository}/contents/{quote(path)}?ref={revision}"
    try:
        payload, _ = _gh(endpoint)
    except RuntimeError as error:
        if "HTTP 404" in str(error):
            return None
        raise
    if payload.get("encoding") != "base64":
        raise RuntimeError(f"unsupported content encoding for {repository}:{path}")
    return base64.b64decode(payload["content"]).decode("utf-8")


def _is_text_probe_path(path: str) -> bool:
    return path.endswith(
        (
            ".py",
            ".pyi",
            ".h",
            ".hpp",
            ".cuh",
            ".cpp",
            ".cu",
            ".rst",
            ".txt",
            ".md",
            ".sh",
        )
    )


def _acquire(cases: list[dict[str, Any]]) -> dict[str, Any]:
    bundle: dict[str, Any] = {}
    for case in cases:
        projection, raw_projection = _candidate_projection(
            case["repository"], case["pull_number"]
        )
        if projection["title"] != case["title"]:
            raise RuntimeError(f"candidate title changed for {case['case_id']}")
        if projection["head_sha"] != case["head_sha"]:
            raise RuntimeError(f"candidate head changed for {case['case_id']}")
        files, raw_files = _gh(
            f"repos/{case['repository']}/pulls/{case['pull_number']}/files?per_page=100",
            paginate=True,
        )
        observed = sorted(item["filename"] for item in files)
        expected = sorted(case["paths"])
        if observed != expected:
            raise RuntimeError(
                f"path parity failed for {case['case_id']}: {observed} != {expected}"
            )
        status_by_path = {item["filename"]: item["status"] for item in files}
        sources: dict[str, dict[str, str | None]] = {}
        for path in case["paths"]:
            if not _is_text_probe_path(path):
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
            "candidate_evidence_projection": {
                **projection,
                "allowed_fields": ["title", "body", "headRefOid"],
                "forbidden_outcome_review_ci_fields_requested": False,
                "graphql_response_sha256": _canonical(json.loads(raw_projection)),
            },
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
            "files_response_sha256": _canonical(json.loads(raw_files)),
            "sources": sources,
        }
    return bundle


def _source_pair(item: dict[str, Any], suffix: str) -> tuple[str, str]:
    matches = [value for path, value in item["sources"].items() if path.endswith(suffix)]
    if len(matches) != 1 or matches[0]["base"] is None or matches[0]["head"] is None:
        raise AssertionError(f"missing unique source pair for {suffix}")
    return str(matches[0]["base"]), str(matches[0]["head"])


def _head_source(item: dict[str, Any], suffix: str) -> str:
    matches = [
        value["head"]
        for path, value in item["sources"].items()
        if path.endswith(suffix) and value["head"] is not None
    ]
    if len(matches) != 1:
        raise AssertionError(f"missing unique head source for {suffix}")
    return str(matches[0])


def _patches(item: dict[str, Any]) -> str:
    return "\n".join(str(file.get("patch") or "") for file in item["files"])


def _candidate_evidence(item: dict[str, Any]) -> dict[str, Any]:
    body = item["candidate_evidence_projection"]["body"]
    return {
        "body_present": bool(body.strip()),
        "body_sha256": _canonical(body),
        "mentions_test": bool(re.search(r"\b(test|pytest|unit test|accuracy)\b", body, re.I)),
        "mentions_benchmark": bool(
            re.search(r"\b(benchmark|throughput|latency|speed|memory|oom|nsys)\b", body, re.I)
        ),
        "mentions_multi_gpu": bool(re.search(r"\b([248]-?gpu|multi-?gpu|world.?size)\b", body, re.I)),
        "line_count": len(body.splitlines()),
    }


def _is_test_path(path: str) -> bool:
    parts = path.lower().split("/")
    name = parts[-1]
    return (
        any(part in {"test", "tests", "testing"} for part in parts[:-1])
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def _common_facts(item: dict[str, Any]) -> dict[str, Any]:
    parse_errors: list[dict[str, str]] = []
    conflict_markers: list[dict[str, Any]] = []
    for path, revisions in item["sources"].items():
        for revision, source in revisions.items():
            if source is None:
                continue
            if path.endswith((".py", ".pyi")):
                try:
                    ast.parse(source, filename=f"{revision}:{path}")
                except SyntaxError as error:
                    parse_errors.append(
                        {
                            "revision": revision,
                            "path": path,
                            "error": f"{error.msg} at {error.lineno}:{error.offset}",
                        }
                    )
            for line_number, line in enumerate(source.splitlines(), start=1):
                if re.match(r"^(<{7}|={7}|>{7})(?:\s|$)", line):
                    conflict_markers.append(
                        {
                            "revision": revision,
                            "path": path,
                            "line": line_number,
                            "marker": line[:16],
                        }
                    )
    paths = [file["filename"] for file in item["files"]]
    test_paths = [path for path in paths if _is_test_path(path)]
    return {
        "candidate_evidence": _candidate_evidence(item),
        "source_file_count": len(item["sources"]),
        "changed_path_count": len(paths),
        "changed_test_paths": test_paths,
        "changed_test_count": len(test_paths),
        "python_parse_errors": parse_errors,
        "merge_conflict_markers": conflict_markers,
        "head_python_syntax_ok": not any(error["revision"] == "head" for error in parse_errors),
        "head_conflict_marker_free": not any(
            marker["revision"] == "head" for marker in conflict_markers
        ),
        "patch_sha256": _canonical(_patches(item)),
    }


def _source(item: dict[str, Any], path: str, revision: str = "head") -> str:
    source = item["sources"][path][revision]
    if source is None:
        raise AssertionError(f"missing {revision} source for {path}")
    return str(source)


def _extract_top_level(source: str, names: set[str]) -> str:
    tree = ast.parse(source)
    segments = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in names:
            segment = ast.get_source_segment(source, node)
            if segment is None:
                raise AssertionError(f"could not extract {node.name}")
            segments.append(segment)
    missing = names - {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if missing:
        raise AssertionError(f"missing top-level definitions: {sorted(missing)}")
    return "\n\n".join(segments)


def _torch() -> Any:
    import torch

    return torch


def _device() -> Any:
    torch = _torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _max_abs(left: Any, right: Any) -> float:
    torch = _torch()
    if left.numel() == 0:
        return 0.0
    return float(torch.max(torch.abs(left.detach().float() - right.detach().float())).item())


def _probe_flashattention_2654(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)
    backward = _source(item, "flash_attn/cute/flash_bwd.py")
    interface = _source(item, "flash_attn/cute/interface.py")
    base_backward = _source(item, "flash_attn/cute/flash_bwd.py", "base")
    body = item["candidate_evidence_projection"]["body"]
    facts.update(
        {
            "head_score_mod_bwd_references": backward.count("score_mod_bwd"),
            "base_score_mod_bwd_references": base_backward.count("score_mod_bwd"),
            "head_passes_runtime_indices_to_score_mod": all(
                token in backward for token in ("q_idx", "kv_idx", "head_idx", "batch_idx")
            ),
            "head_interface_sm80_or_sm120_score_mod_rejection_present": bool(
                re.search(r"sm_(?:80|120).{0,160}(?:score_mod|score mod).{0,80}(?:raise|assert)", interface, re.S | re.I)
            ),
            "candidate_validation_architectures": sorted(
                set(re.findall(r"\b(?:sm\d{2,3}[a-z]?|RTX\s*\d+\s*Ti|A100|H100|B200)\b", body, re.I))
            ),
            "candidate_says_local_uncommitted_prerequisite_fixes": "locally modified" in body.lower()
            and "not part of this feature pr" in body.lower(),
            "candidate_mentions_ignored_errors": "were ignored" in body.lower(),
        }
    )
    return facts


def _probe_liger_1274(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    device = _device()

    def sapo(coef: Any, temperature: float) -> Any:
        return torch.sigmoid(temperature * (coef - 1)) * 4 / temperature

    def base_fn(coef: Any, advantages: Any) -> Any:
        expanded = advantages.unsqueeze(1).expand_as(coef)
        positive = expanded > 0
        output = torch.empty_like(coef)
        output[positive] = sapo(coef[positive], 1.0)
        output[~positive] = sapo(coef[~positive], 1.05)
        return -output * expanded

    def head_fn(coef: Any, advantages: Any) -> Any:
        expanded = advantages.unsqueeze(1).expand_as(coef)
        return -torch.where(expanded > 0, sapo(coef, 1.0), sapo(coef, 1.05)) * expanded

    numeric = []
    for dtype in (torch.float32, torch.bfloat16):
        coef_base = torch.tensor(
            [[0.25, 0.8, 1.0, 1.3], [0.4, 0.9, 1.2, 2.0], [0.7, 1.0, 1.1, 3.0]],
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        coef_head = coef_base.detach().clone().requires_grad_(True)
        advantages = torch.tensor([2.0, -1.5, 0.0], device=device, dtype=dtype)
        base_output = base_fn(coef_base, advantages)
        head_output = head_fn(coef_head, advantages)
        base_output.float().sum().backward()
        head_output.float().sum().backward()
        numeric.append(
            {
                "dtype": str(dtype),
                "output_max_abs": _max_abs(base_output, head_output),
                "gradient_max_abs": _max_abs(coef_base.grad, coef_head.grad),
            }
        )

    compile_results: dict[str, Any] = {}
    if hasattr(torch, "compile"):
        for name, function in (("base", base_fn), ("head", head_fn)):
            try:
                if hasattr(torch, "_dynamo"):
                    torch._dynamo.reset()
                compiled = torch.compile(function, fullgraph=True)
                coef = torch.linspace(0.2, 2.0, 24, device=device).view(3, 8).requires_grad_(True)
                advantages = torch.tensor([1.0, -1.0, 0.0], device=device)
                output = compiled(coef, advantages)
                output.sum().backward()
                if device.type == "cuda":
                    torch.cuda.synchronize()
                compile_results[name] = {
                    "fullgraph_succeeded": True,
                    "finite_output": bool(torch.isfinite(output).all().item()),
                    "finite_gradient": bool(torch.isfinite(coef.grad).all().item()),
                }
            except Exception as error:
                compile_results[name] = {
                    "fullgraph_succeeded": False,
                    "error": f"{type(error).__name__}: {str(error)[:800]}",
                }
    facts.update(
        {
            "eager_base_head_numeric_matrix": numeric,
            "fullgraph_compile": compile_results,
            "head_uses_boolean_index_assignment": "per_token_loss[positive_advantages_mask]" in _source(
                item, "src/liger_kernel/chunked_loss/grpo_loss.py"
            ),
            "head_uses_torch_where": "per_token_loss = torch.where(" in _source(
                item, "src/liger_kernel/chunked_loss/grpo_loss.py"
            ),
        }
    )
    return facts


def _probe_liger_1268(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    device = _device()
    max_base_head_error = 0.0
    max_oracle_error = 0.0
    matrix = []
    for softcap in (None, 1.0, 5.0):
        for smoothing in (0.0, 0.2):
            for weighted in (False, True):
                for reduction in ("sum", "mean"):
                    original = torch.tensor(
                        [[-3.0, 0.5, 2.0, 8.0], [1.0, -7.0, 0.25, 3.0], [2.5, 1.5, -1.0, 0.0]],
                        dtype=torch.float64,
                        device=device,
                        requires_grad=True,
                    )
                    labels = torch.tensor([3, -100, 0], device=device)
                    weights = (
                        torch.tensor([0.4, 1.3, 0.8, 2.0], dtype=torch.float64, device=device)
                        if weighted
                        else torch.ones(4, dtype=torch.float64, device=device)
                    )
                    transformed = original if softcap is None else softcap * torch.tanh(original / softcap)
                    chain = torch.ones_like(original) if softcap is None else 1 - torch.tanh(original / softcap) ** 2
                    probs = transformed.softmax(dim=-1)
                    valid = labels != -100
                    safe_labels = labels.clone()
                    safe_labels[~valid] = 0
                    epsilon = smoothing / original.shape[-1]
                    target_weight = weights[safe_labels]
                    denominator = (
                        target_weight[valid].sum()
                        if reduction == "mean" and weighted
                        else valid.sum()
                        if reduction == "mean"
                        else torch.tensor(1.0, dtype=torch.float64, device=device)
                    )
                    if weighted:
                        common = (1 - smoothing) * target_weight[:, None] * probs
                        common = common + epsilon * (-weights[None, :] + probs * weights.sum())
                    else:
                        common = probs - epsilon
                    common = common / denominator
                    old = common.clone()
                    rows = torch.arange(original.shape[0], device=device)
                    old[rows, safe_labels] -= (1 - smoothing) * target_weight / denominator
                    old = old * chain
                    old[~valid] = 0
                    head = common * chain
                    target_chain = chain[rows, safe_labels]
                    head[rows, safe_labels] += (
                        -(1 - smoothing) * target_weight / denominator * target_chain
                    )
                    head[~valid] = 0

                    log_probs = transformed.log_softmax(dim=-1)
                    per_row = -(1 - smoothing) * target_weight * log_probs[rows, safe_labels]
                    if weighted:
                        per_row = per_row - epsilon * (log_probs * weights[None, :]).sum(dim=-1)
                    else:
                        per_row = per_row - epsilon * log_probs.sum(dim=-1)
                    oracle_loss = per_row[valid].sum() / denominator
                    oracle = torch.autograd.grad(oracle_loss, original)[0]
                    base_head_error = _max_abs(old, head)
                    oracle_error = _max_abs(head, oracle)
                    max_base_head_error = max(max_base_head_error, base_head_error)
                    max_oracle_error = max(max_oracle_error, oracle_error)
                    matrix.append(
                        {
                            "softcap": softcap,
                            "label_smoothing": smoothing,
                            "weighted": weighted,
                            "reduction": reduction,
                            "base_head_max_abs": base_head_error,
                            "oracle_max_abs": oracle_error,
                            "ignored_row_zero": bool(torch.count_nonzero(head[1]).item() == 0),
                        }
                    )
    source = _source(item, "src/liger_kernel/ops/cross_entropy.py")
    facts.update(
        {
            "algebra_matrix": matrix,
            "max_base_head_gradient_error": max_base_head_error,
            "max_independent_oracle_gradient_error": max_oracle_error,
            "ignore_index_returns_before_target_correction": source.index("if y == ignore_index:")
            < source.index("# dx_y correction"),
            "softcap_target_chain_uses_softcapped_value_over_softcap": "t_y = ori_X_y / softcap" in source,
            "target_correction_barrier_present": "tl.debug_barrier()\n        dxy" in source,
        }
    )
    return facts


def _probe_liger_1230(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)
    source = _source(item, "src/liger_kernel/transformers/trainer/orpo_trainer.py")
    calls: list[str] = []

    def redirect(*_args: Any) -> None:
        calls.append("redirect")
        raise AssertionError("FSDP redirection requires an FSDP model")

    def partial(*_args: Any) -> tuple[str, list[str]]:
        calls.append("direct")
        return "finite-loss", ["aux"]

    base_plain_failed = False
    try:
        redirect(object(), partial, object(), object(), object(), object())
    except AssertionError:
        base_plain_failed = True
    head_plain_result = partial(object(), object(), object(), object())
    facts.update(
        {
            "base_plain_model_redirection_fails": base_plain_failed,
            "head_plain_model_direct_result": list(head_plain_result),
            "isinstance_fsdp_branch_present": "isinstance(model, FullyShardedDataParallel)" in source,
            "fsdp_branch_still_calls_redirection": "_FSDPForwardRedirection()(" in source,
            "trl_new_location_with_old_fallback": "from trl.experimental.orpo import ORPOTrainer" in source
            and "from trl.trainer import ORPOTrainer" in source,
            "simulated_call_trace": calls,
        }
    )
    return facts


def _probe_megatron_5808(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    events: list[str] = []

    class Nested(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(2.0))
            self.child = torch.nn.Linear(2, 2, bias=False)

        def forward(self, value: Any) -> Any:
            return self.child(value) * self.weight

    model = Nested().to(_device())
    model.register_forward_pre_hook(lambda *_args: events.append("root-pre"))
    model.register_forward_hook(lambda *_args: events.append("root-post"))
    model.child.register_forward_pre_hook(lambda *_args: events.append("child-pre"))
    model.child.register_forward_hook(lambda *_args: events.append("child-post"))
    value = torch.ones(2, 2, device=_device(), requires_grad=True)
    direct = model.forward(value)
    direct_events = list(events)
    events.clear()
    called = model(value)
    called_events = list(events)
    grad_direct = torch.autograd.grad(
        direct.sum(), (*tuple(model.parameters()), value), retain_graph=True
    )
    grad_called = torch.autograd.grad(
        called.sum(), (*tuple(model.parameters()), value)
    )
    source = _source(
        item, "megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py"
    )
    facts.update(
        {
            "direct_forward_events": direct_events,
            "module_call_events": called_events,
            "root_hooks_restored_exactly_once": called_events.count("root-pre") == 1
            and called_events.count("root-post") == 1,
            "child_hooks_exactly_once_in_both": direct_events == ["child-pre", "child-post"]
            and called_events.count("child-pre") == 1
            and called_events.count("child-post") == 1,
            "output_max_abs": _max_abs(direct, called),
            "gradient_max_abs": max(
                _max_abs(left, right) for left, right in zip(grad_direct, grad_called, strict=True)
            ),
            "head_uses_module_call_protocol": "output = self.module(*inputs, **kwargs)" in source,
        }
    )
    return facts


def _probe_megatron_5798(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)

    def accumulated_count(valid_counts: list[int], groups: list[list[int]], fixed: bool) -> float:
        total = 0.0
        for group in groups:
            mean_valid = sum(valid_counts[index] for index in group) / len(group)
            total += mean_valid * (len(group) if fixed else 1)
        return total

    layouts = []
    for valid_counts in ([32, 32, 32, 32], [32, 24, 16, 8]):
        split = [[0], [1], [2], [3]]
        packed = [[0, 1, 2, 3]]
        layouts.append(
            {
                "valid_counts": list(valid_counts),
                "base_mbs1": accumulated_count(list(valid_counts), split, False),
                "base_mbs4": accumulated_count(list(valid_counts), packed, False),
                "head_mbs1": accumulated_count(list(valid_counts), split, True),
                "head_mbs4": accumulated_count(list(valid_counts), packed, True),
            }
        )
    source = _source(item, "megatron/core/transformer/moe/router.py")
    facts.update(
        {
            "partition_count_matrix": layouts,
            "head_all_layouts_partition_invariant": all(
                row["head_mbs1"] == row["head_mbs4"] for row in layouts
            ),
            "base_distinguishes_mbs4": all(row["base_mbs1"] != row["base_mbs4"] for row in layouts),
            "source_multiplies_mean_count_by_batch_size": "valid_token_count=local_num_tokens * bsz" in source,
        }
    )
    return facts


def _probe_megatron_5743(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    rank_microbatch_grads = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]]]
    )
    eager = rank_microbatch_grads.mean(dim=0).sum(dim=0)
    deferred = rank_microbatch_grads.sum(dim=1).mean(dim=0)
    dbuffer = _source(
        item, "megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/dbuffer.py"
    )
    parameter_group = _source(
        item, "megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/parameter_group.py"
    )
    module = _source(
        item, "megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py"
    )
    facts.update(
        {
            "analytic_eager_deferred_max_abs": _max_abs(eager, deferred),
            "last_microbatch_threaded_from_context": "self.context.is_last_microbatch" in module,
            "outer_finalize_is_last_gated": "if is_last_microbatch:" in parameter_group
            and "self.main_grad.redistribute(self.main_weight.placements)" in parameter_group,
            "replicate_to_partial_rejects_non_avg": "Replicate -> Partial redistribute supports AVG only" in dbuffer,
            "partial_dtensor_roundtrip_supported": "dist_tensor.Partial(reduce_op)" in dbuffer,
            "accumulation_placements_persisted": "self._accumulation_placements" in parameter_group,
        }
    )
    return facts


def _probe_megatron_5742(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    source = _source(item, "megatron/core/optimizer/distrib_optimizer.py")

    def lion_step(param: Any, moment: Any, grad: Any, lr: float = 0.01) -> tuple[Any, Any]:
        beta1, beta2 = 0.9, 0.99
        update = beta1 * moment + (1 - beta1) * grad
        next_param = param - lr * torch.sign(update)
        next_moment = beta2 * moment + (1 - beta2) * grad
        return next_param, next_moment

    initial = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float64)
    gradients = [
        torch.tensor([0.2, -0.1, 0.7], dtype=torch.float64),
        torch.tensor([-0.4, 0.3, 0.1], dtype=torch.float64),
        torch.tensor([0.5, -0.8, -0.2], dtype=torch.float64),
        torch.tensor([0.1, 0.2, -0.5], dtype=torch.float64),
    ]
    continuous_param = initial.clone()
    continuous_moment = torch.zeros_like(initial)
    for gradient in gradients:
        continuous_param, continuous_moment = lion_step(
            continuous_param, continuous_moment, gradient
        )
    resumed_param = initial.clone()
    resumed_moment = torch.zeros_like(initial)
    for gradient in gradients[:2]:
        resumed_param, resumed_moment = lion_step(resumed_param, resumed_moment, gradient)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "lion.pt"
        torch.save(
            {"param": resumed_param, "exp_avg": resumed_moment, "step": torch.tensor(2)},
            checkpoint,
        )
        restored = torch.load(checkpoint, weights_only=True)
    for gradient in gradients[2:]:
        resumed_param, resumed_moment = lion_step(
            restored["param"], restored["exp_avg"], gradient
        )
        restored = {"param": resumed_param, "exp_avg": resumed_moment}
    facts.update(
        {
            "continuous_resume_param_max_abs": _max_abs(continuous_param, resumed_param),
            "continuous_resume_moment_max_abs": _max_abs(continuous_moment, resumed_moment),
            "lion_state_key_property_present": '"lion": ("exp_avg",)' in source,
            "muon_scalar_optimizer_resolution_present": "self.config.muon_scalar_optimizer" in source,
            "dynamic_state_key_loop_count": source.count("self.optimizer_state_keys"),
            "constructor_accepts_init_state_fn": "or init_state_fn is not None" in source,
        }
    )
    return facts


def _probe_torchtitan_3841(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)
    source = _source(
        item, "torchtitan/experiments/graph_trainer/graph_pp/split_di_dw.py"
    )
    tests = _source(
        item, "torchtitan/experiments/graph_trainer/tests/test_graph_pp_passes.py"
    )
    tree = ast.parse(source)
    functions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    candidate_tests = sorted(
        set(re.findall(r"def (test_[A-Za-z0-9_]*split[A-Za-z0-9_]*)", tests))
    )
    facts.update(
        {
            "split_source_lines": len(source.splitlines()),
            "split_functions": sorted(functions),
            "candidate_split_tests": candidate_tests,
            "candidate_has_real_dsv3_moe_control": "test_real_dsv3_moe_block_split_reconstructs_backward"
            in tests,
            "candidate_has_no_input_grad_control": "test_real_dsv3_moe_block_without_input_grad_skips_split"
            in tests,
            "candidate_has_explicit_residual_control": bool(
                re.search(r"test_[^(]*(?:residual|skip)", tests, re.I)
            ),
            "candidate_has_explicit_shared_parameter_control": bool(
                re.search(r"test_[^(]*shared", tests, re.I)
            ),
            "graphs_linted_and_recompiled": source.count(".graph.lint()") >= 2
            and source.count(".recompile()") >= 2,
            "get_attr_kept_as_graph_constant": "if node.op == \"get_attr\"" in source,
            "symbolic_shape_liveins_separated": "saved_sym_nodes" in source,
            "no_input_grad_returns_none": "if num_input_grads == 0:\n        return None"
            in source,
        }
    )
    return facts


def _probe_torchtitan_3897(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)
    trainer = _source(item, "torchtitan/experiments/rl/actors/trainer.py")
    loss = _source(item, "torchtitan/components/loss.py")
    attention = _source(item, "torchtitan/models/common/attention.py")
    integration = _source(
        item, "torchtitan/experiments/rl/tests/integration_tests.py"
    )
    facts.update(
        {
            "head_parse_error_count": sum(
                error["revision"] == "head" for error in facts["python_parse_errors"]
            ),
            "head_conflict_marker_count": sum(
                marker["revision"] == "head" for marker in facts["merge_conflict_markers"]
            ),
            "integration_file_has_unresolved_conflict": "<<<<<<<" in integration
            and "=======" in integration
            and ">>>>>>>" in integration,
            "grad_scaler_wired": "GradScaler" in trainer and "scaler.step" in trainer,
            "chunked_loss_backward_scale_wired": "backward_scale" in loss,
            "attention_has_fp16_safe_change": "float16" in _patches(item).lower()
            or "fp16" in _patches(item).lower(),
            "attention_source_sha256": _canonical(attention),
            "compound_changed_component_count": 4,
        }
    )
    return facts


def _probe_torchtitan_3867(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    device = _device()
    fused = torch.arange(48, dtype=torch.float32, device=device).reshape(12, 4).clone()
    original = fused.clone()
    split_state = {
        "wq.weight": fused[:4].clone(),
        "wk.weight": fused[4:8].clone(),
        "wv.weight": fused[8:].clone(),
    }
    split_aliases_live = any(
        tensor.untyped_storage().data_ptr() == fused.untyped_storage().data_ptr()
        for tensor in split_state.values()
    )
    expected = {}
    for index, (name, tensor) in enumerate(split_state.items(), start=1):
        expected[name] = torch.full_like(tensor, float(index * 10))
        tensor.copy_(expected[name])
    base_live_changed = not torch.equal(fused, original)
    fused.copy_(
        torch.cat(
            [expected["wq.weight"], expected["wk.weight"], expected["wv.weight"]],
            dim=0,
        )
    )
    head_roundtrip_error = _max_abs(
        fused,
        torch.cat(list(expected.values()), dim=0),
    )

    plain = torch.nn.Linear(4, 3, bias=True, device=device)
    partial = {"weight": torch.ones_like(plain.weight)}
    incompatible = plain.load_state_dict(partial, strict=False)
    generator = _source(item, "torchtitan/experiments/rl/actors/generator.py")
    tests = _source(item, "tests/unit_tests/test_fused_qkv.py")
    facts.update(
        {
            "split_state_aliases_live_fused_storage": split_aliases_live,
            "inplace_store_write_changed_live_fused_param": base_live_changed,
            "post_load_fused_roundtrip_max_abs": head_roundtrip_error,
            "generator_reloads_received_state": "model.load_state_dict(model_sd, strict=False)"
            in generator,
            "strict_false_silently_reports_missing_keys": list(incompatible.missing_keys),
            "strict_false_reports_unexpected_keys": list(incompatible.unexpected_keys),
            "candidate_test_covers_fused_qkv": "TestFusedQKVInplaceWrite" in tests,
            "candidate_test_covers_fused_swiglu": "SwiGLU" in tests,
            "candidate_test_exercises_malformed_or_missing_keys": bool(
                re.search(r"test_[^(]*(?:missing|malformed|unexpected)", tests, re.I)
            ),
        }
    )
    return facts


def _probe_verl_7014(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    device = _device()

    class Toy:
        def __init__(self) -> None:
            self.weight = torch.tensor([2.0, -1.0], device=device)

    @contextlib.contextmanager
    def merged(module: Toy):
        module.weight.add_(torch.tensor([3.0, 4.0], device=device))
        try:
            yield
        finally:
            module.weight.sub_(torch.tensor([3.0, 4.0], device=device))

    old_module = Toy()
    with merged(old_module):
        old_alias = {"weight": old_module.weight}
    old_delayed_value = old_alias["weight"].clone()

    new_module = Toy()

    def stream():
        with merged(new_module):
            yield "weight", new_module.weight.detach().clone()

    streamed = dict(stream())
    expected_merged = torch.tensor([5.0, 3.0], device=device)
    expected_base = torch.tensor([2.0, -1.0], device=device)
    source = _source(item, "verl/workers/engine/fsdp/transformer_impl.py")
    return_position = source.index("return self._merged_lora_per_tensor_param(), None")
    qat_positions = [
        match.start() for match in re.finditer(r"(?:qat|quantization)", source, re.I)
    ]
    facts.update(
        {
            "old_delayed_export_max_abs_from_base": _max_abs(
                old_delayed_value, expected_base
            ),
            "old_delayed_export_max_abs_from_merged": _max_abs(
                old_delayed_value, expected_merged
            ),
            "head_streamed_export_max_abs_from_merged": _max_abs(
                streamed["weight"], expected_merged
            ),
            "head_post_context_restore_max_abs": _max_abs(
                new_module.weight, expected_base
            ),
            "plain_tensor_is_cloned_inside_generator": "else param.detach().clone()"
            in source,
            "dtensor_is_materialized_inside_generator": ".full_tensor()" in source,
            "try_finally_offload_cleanup_present": "finally:" in source[
                source.index("def _merged_lora_per_tensor_param"):
            ],
            "early_return_precedes_later_qat_or_quantization_references": any(
                position > return_position for position in qat_positions
            ),
        }
    )
    return facts


def _probe_verl_7013(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)

    def advance(value: float, observations: list[tuple[float, int]]) -> float:
        for current_kl, n_steps in observations:
            error = max(-0.2, min(0.2, current_kl / 1.0 - 1.0))
            value *= 1 + error * n_steps / 10_000
        return value

    observations = [(1.7, 128), (0.2, 64), (1.4, 256), (0.5, 32)]
    uninterrupted = advance(0.1, observations)
    prefix = advance(0.1, observations[:2])
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "kl_ctrl.pt"
        torch.save(
            {
                "version": torch.tensor(1, dtype=torch.int64),
                "value": torch.tensor(prefix, dtype=torch.float64),
            },
            path,
        )
        state = torch.load(path, map_location="cpu", weights_only=True)
    resumed = advance(float(state["value"].item()), observations[2:])
    core = _source(item, "verl/trainer/ppo/core_algos.py")
    trainer_paths = [
        "verl/trainer/ppo/ray_trainer.py",
        "verl/trainer/ppo/v1/trainer_base.py",
        "verl/experimental/fully_async_policy/fully_async_trainer.py",
    ]
    wiring = {
        path: {
            "save": ".save(" in _source(item, path),
            "load": ".load(" in _source(item, path),
        }
        for path in trainer_paths
    }
    facts.update(
        {
            "resume_trajectory_abs_error": abs(uninterrupted - resumed),
            "checkpoint_payload_tensor_only": all(
                isinstance(value, torch.Tensor) for value in state.values()
            ),
            "trainer_wiring": wiring,
            "all_three_trainers_save_and_load": all(
                value["save"] and value["load"] for value in wiring.values()
            ),
            "weights_only_load_present": "weights_only=True" in core,
            "atomic_replace_present": "os.replace(temporary, target)" in core,
            "missing_state_compatibility_present": "return False" in core,
            "malformed_and_nonfinite_fail_closed": "must be a scalar tensor" in core
            and "must be finite and non-negative" in core,
        }
    )
    return facts


def _probe_verl_6984(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    device = _device()

    def run(pop_output: bool) -> tuple[int, int]:
        outputs = []
        references = []
        for _ in range(8):
            tensor = torch.empty((256, 1024), device=device)
            references.append(weakref.ref(tensor))
            metadata = {"model_output": {"nested": [tensor]}, "loss": 1.0}
            if pop_output:
                metadata.pop("model_output", None)
            outputs.append(metadata)
        del tensor, metadata
        gc.collect()
        live = sum(reference() is not None for reference in references)
        retained_numel = sum(
            value["model_output"]["nested"][0].numel()
            for value in outputs
            if "model_output" in value
        )
        return live, retained_numel

    base_live, base_numel = run(False)
    head_live, head_numel = run(True)
    source = _source(item, "verl/workers/engine/fsdp/transformer_impl.py")
    facts.update(
        {
            "base_retained_model_output_tensor_count": base_live,
            "head_retained_model_output_tensor_count": head_live,
            "base_retained_numel": base_numel,
            "head_retained_numel": head_numel,
            "pop_occurs_only_after_backward": source.index(
                'meta_info.pop("model_output", None)'
            )
            > source.index("loss.backward()"),
            "forward_only_guard_preserves_output": "if not forward_only:" in source[
                max(0, source.index('meta_info.pop("model_output", None)') - 1600): source.index(
                    'meta_info.pop("model_output", None)'
                )
            ],
        }
    )
    return facts


def _probe_megatron_5819(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    tensor = torch.tensor([1.0], device=_device())

    def copy_shell(value: Any) -> Any:
        if isinstance(value, tuple):
            return tuple(copy_shell(element) for element in value)
        if isinstance(value, list):
            return [copy_shell(element) for element in value]
        if isinstance(value, dict):
            return {key: copy_shell(element) for key, element in value.items()}
        return value

    base_cache = {"batch": [{"tokens": tensor, "labels": tensor + 1}]}
    base_result = base_cache
    base_result["batch"][0].pop("tokens")
    head_cache = {"batch": [{"tokens": tensor, "labels": tensor + 1}]}
    head_result = copy_shell(head_cache)
    head_result["batch"][0].pop("tokens")
    source = _source(item, "megatron/core/full_cuda_graph.py")
    facts.update(
        {
            "base_destructive_consumer_corrupts_static_cache": "tokens"
            not in base_cache["batch"][0],
            "head_destructive_consumer_preserves_static_cache": "tokens"
            in head_cache["batch"][0],
            "head_nested_dict_is_distinct": head_result["batch"][0]
            is not head_cache["batch"][0],
            "head_tensor_identity_is_preserved": head_cache["batch"][0]["labels"]
            is head_result["batch"][0]["labels"],
            "recursive_tuple_list_dict_copy_present": all(
                token in source
                for token in ("isinstance(src, tuple)", "isinstance(src, list)", "isinstance(src, dict)")
            ),
            "candidate_changed_test_count": facts["changed_test_count"],
        }
    )
    return facts


def _probe_megatron_5761(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)
    events: list[str] = []

    class Recorder:
        def __init__(self, name: str) -> None:
            self.name = name

        def __enter__(self) -> Recorder:
            events.append(f"enter:{self.name}")
            return self

        def __exit__(self, *_args: Any) -> None:
            events.append(f"exit:{self.name}")

    autocast = Recorder("autocast")
    fp8 = Recorder("fp8")
    with (autocast and fp8):
        events.append("body")
    base_events = list(events)
    events.clear()
    with autocast, fp8:
        events.append("body")
    head_events = list(events)
    source = _source(item, "megatron/core/pipeline_parallel/combined_1f1b.py")
    facts.update(
        {
            "base_context_events": base_events,
            "head_context_events": head_events,
            "base_enters_autocast": "enter:autocast" in base_events,
            "head_enters_both_contexts": head_events
            == ["enter:autocast", "enter:fp8", "body", "exit:fp8", "exit:autocast"],
            "head_source_uses_multi_context_with": "with context_manager, outer_fp8_context:"
            in source,
        }
    )
    return facts


def _probe_megatron_5724(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)
    packed = _source(item, "megatron/core/packed_seq_params.py")
    config = _source(item, "megatron/core/transformer/transformer_config.py")

    def resolve(alignment: str | int, maximum: int, capacity: int | None, graph: bool):
        if graph:
            return None, int(maximum), capacity
        if alignment == "max":
            return None, int(maximum), capacity
        return int(alignment), None, capacity

    matrix = [
        {
            "alignment": alignment,
            "capacity": capacity,
            "graph": graph,
            "resolved": list(resolve(alignment, 2048, capacity, graph)),
        }
        for alignment in ("max", 128)
        for capacity in (None, 32)
        for graph in (False, True)
    ]
    facts.update(
        {
            "padding_resolution_matrix": matrix,
            "eager_explicit_capacity_matches_graph": all(
                resolve(alignment, 2048, 32, False)[2]
                == resolve(alignment, 2048, 32, True)[2]
                for alignment in ("max", 128)
            ),
            "eager_default_preserves_dynamic_cu_seqlens": all(
                resolve(alignment, 2048, None, False)[2] is None
                for alignment in ("max", 128)
            ),
            "config_default_is_none": bool(
                re.search(r"thd_max_packed_sequences:\s*Optional\[int\].*?default=None", config, re.S)
            ),
            "graph_without_capacity_fails_closed": "THD CUDA Graph requires --thd-max-packed-sequences to be set."
            in config,
            "head_function_matches_expected_branches": "return None, int(max_seqlen_per_dp_cp_rank), thd_max_packed_sequences"
            in packed,
        }
    )
    return facts


def _probe_megatron_5714(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)
    mlp = _source(item, "megatron/core/transformer/mlp.py")
    matrix = []
    half_axis_size = 24
    for dp_size in (1, 2, 3, 4, 6, 8):
        local_axis_size = (2 * half_axis_size) / dp_size
        integral_local = local_axis_size.is_integer()
        local_value = int(local_axis_size) if integral_local else None
        completely_inside_half = bool(
            integral_local
            and local_value
            and half_axis_size % local_value == 0
            and dp_size == 2 * (half_axis_size // local_value)
        )
        offsets = []
        if completely_inside_half:
            shards_per_half = half_axis_size // int(local_value)
            offsets = [
                (rank // shards_per_half, rank % shards_per_half)
                for rank in range(dp_size)
            ]
        matrix.append(
            {
                "dp_size": dp_size,
                "local_axis_size": local_value,
                "mapping_supported": completely_inside_half,
                "half_and_local_offsets": offsets,
            }
        )
    facts.update(
        {
            "dp_mapping_matrix": matrix,
            "all_even_divisors_supported": all(
                row["mapping_supported"]
                for row in matrix
                if row["dp_size"] in (2, 4, 6, 8)
            ),
            "odd_dp3_rejected_by_mapping": not next(
                row["mapping_supported"] for row in matrix if row["dp_size"] == 3
            ),
            "source_asserts_each_shard_inside_one_half": "completely inside either the W or V half"
            in mlp,
            "source_asserts_two_halves_cover_dp": "assert dp_size == 2 * shards_per_half"
            in mlp,
            "candidate_test_topology_mentions_tp2_dp4": "TP2 DP{N}" in _patches(item)
            and "dp_cp_size" in _patches(item),
        }
    )
    return facts


def _probe_megatron_5710(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    device = _device()
    events: list[str] = []
    module = torch.nn.Linear(4, 4, bias=False).to(device)
    module.weight.requires_grad_(False)
    module.register_full_backward_pre_hook(lambda *_args: events.append("pre"))
    module.register_full_backward_hook(lambda *_args: events.append("post"))
    for _ in range(2):
        value = torch.ones(2, 4, device=device, requires_grad=True)
        module(value).sum().backward()
    source = _source(
        item,
        "megatron/core/distributed/fsdp/src/megatron_fsdp/experimental/module.py",
    )
    facts.update(
        {
            "all_frozen_repeated_backward_hook_events": events,
            "all_frozen_pre_post_balanced": events == ["pre", "post", "pre", "post"],
            "full_backward_hook_only_for_zero_trainable": "if self._num_trainable_parameters == 0:"
            in source
            and "register_full_backward_hook" in source,
            "trainable_completion_uses_grad_hooks": "_make_grad_hook" in source,
            "candidate_has_frozen_outside_backward_graph_control": "frozen_child_without_grad_inputs_skips_backward"
            in _patches(item),
        }
    )
    return facts


def _probe_slime_2207(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)

    def base_mask(response_length: int, _metadata_lengths: tuple[int, ...]) -> list[int]:
        return [0] * response_length

    def head_mask(response_length: int, metadata_lengths: tuple[int, ...]) -> list[int]:
        mask = [0] * response_length
        if any(length != response_length for length in metadata_lengths):
            raise ValueError("response metadata length mismatch")
        return mask

    equivalence = []
    for response_length in (0, 1, 2, 17):
        metadata = (response_length, response_length)
        equivalence.append(
            {
                "response_length": response_length,
                "equal": base_mask(response_length, metadata)
                == head_mask(response_length, metadata),
            }
        )
    base_invalid_silent = base_mask(3, (2, 3)) == [0, 0, 0]
    try:
        head_mask(3, (2, 3))
        head_invalid_failed_closed = False
    except ValueError:
        head_invalid_failed_closed = True
    rollout = _source(item, "slime/rollout/sglang_rollout.py")
    types_source = _source(item, "slime/utils/types.py")
    facts.update(
        {
            "valid_input_base_head_equivalence": equivalence,
            "all_valid_inputs_semantically_identical": all(
                row["equal"] for row in equivalence
            ),
            "base_invalid_metadata_is_silent": base_invalid_silent,
            "head_invalid_metadata_fails_closed": head_invalid_failed_closed,
            "production_change_is_assignment_to_helper_only": "sample.mask_response_tokens(0)"
            in rollout,
            "helper_assigns_same_response_length_mask": "self.loss_mask = [int(value)] * self.response_length"
            in types_source,
            "helper_adds_only_post_assignment_validation": "self._validate_response_metadata_lengths()"
            in types_source,
        }
    )
    return facts


def _probe_slime_2205(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    device = _device()

    def vectorized(rewards: Any, discount: float, chunk_size: int = 128) -> Any:
        batch, length = rewards.shape
        reversed_rewards = torch.flip(rewards, dims=[1])
        pad = (chunk_size - length % chunk_size) % chunk_size
        if pad:
            reversed_rewards = torch.nn.functional.pad(reversed_rewards, (0, pad))
        padded_length = reversed_rewards.shape[1]
        if padded_length == 0:
            return rewards.clone()
        chunks = padded_length // chunk_size
        reward_chunks = reversed_rewards.view(batch, chunks, chunk_size)
        indices = torch.arange(chunk_size, device=rewards.device)
        difference = indices[None, :] - indices[:, None]
        matrix = torch.zeros(chunk_size, chunk_size, device=rewards.device, dtype=rewards.dtype)
        mask = difference >= 0
        if discount == 0.0:
            matrix[mask & (difference == 0)] = 1
            powers = torch.zeros(chunk_size, device=rewards.device, dtype=rewards.dtype)
        else:
            matrix[mask] = discount ** difference[mask].to(rewards.dtype)
            powers = discount ** torch.arange(
                1, chunk_size + 1, device=rewards.device, dtype=rewards.dtype
            )
        local = (reward_chunks.reshape(batch * chunks, chunk_size) @ matrix).view(
            batch, chunks, chunk_size
        )
        output = reversed_rewards.new_zeros(batch, padded_length)
        previous = torch.zeros(batch, device=rewards.device, dtype=rewards.dtype)
        for chunk in range(chunks):
            chunk_length = chunk_size if chunk < chunks - 1 or not pad else chunk_size - pad
            start = chunk * chunk_size
            values = local[:, chunk, :chunk_length] + previous[:, None] * powers[:chunk_length]
            output[:, start : start + chunk_length] = values
            previous = values[:, -1]
        return torch.flip(output[:, :length], dims=[1])

    def recurrence(rewards: Any, discount: float) -> Any:
        output = torch.zeros_like(rewards)
        running = torch.zeros(rewards.shape[0], device=rewards.device, dtype=rewards.dtype)
        for index in range(rewards.shape[1] - 1, -1, -1):
            running = rewards[:, index] + discount * running
            output[:, index] = running
        return output

    matrix = []
    generator = torch.Generator(device="cpu").manual_seed(20260902)
    for dtype in (torch.float32, torch.bfloat16):
        for length in (0, 1, 7, 127, 128, 129, 513):
            for discount in (0.0, 0.5, 0.99, 1.0):
                cpu = torch.randn(3, max(length, 1) * 2, generator=generator)[:, ::2]
                rewards = cpu[:, :length].to(device=device, dtype=dtype)
                observed = vectorized(rewards, discount)
                expected = recurrence(rewards, discount)
                matrix.append(
                    {
                        "dtype": str(dtype),
                        "length": length,
                        "discount": discount,
                        "input_noncontiguous": not rewards.is_contiguous() if length else False,
                        "max_abs": _max_abs(observed, expected),
                        "finite": bool(torch.isfinite(observed).all().item()),
                        "dtype_preserved": observed.dtype == rewards.dtype,
                    }
                )
    timing_rewards = torch.randn(8, 2048, device=device)
    for _ in range(3):
        vectorized(timing_rewards, 0.99)
        recurrence(timing_rewards, 0.99)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(10):
        vectorized(timing_rewards, 0.99)
    if device.type == "cuda":
        torch.cuda.synchronize()
    vectorized_ms = (time.perf_counter() - started) * 100
    started = time.perf_counter()
    for _ in range(10):
        recurrence(timing_rewards, 0.99)
    if device.type == "cuda":
        torch.cuda.synchronize()
    recurrence_ms = (time.perf_counter() - started) * 100
    facts.update(
        {
            "correctness_matrix": matrix,
            "max_fp32_error": max(
                row["max_abs"] for row in matrix if row["dtype"] == "torch.float32"
            ),
            "max_bfloat16_error": max(
                row["max_abs"] for row in matrix if row["dtype"] == "torch.bfloat16"
            ),
            "paired_timing_ms_per_call": {
                "vectorized": vectorized_ms,
                "scalar_recurrence": recurrence_ms,
                "speedup": recurrence_ms / vectorized_ms if vectorized_ms else None,
            },
            "candidate_function_present": "def chunked_discounted_returns(" in _source(
                item, "slime/utils/ppo_utils.py"
            ),
        }
    )
    return facts


def _probe_slime_2204(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)

    def normalize(rewards: list[float], groups: list[int], use_std: bool) -> list[float]:
        positions: dict[int, list[int]] = defaultdict(list)
        for position, group in enumerate(groups):
            positions[group].append(position)
        values = torch.tensor(rewards, dtype=torch.float32)
        result = torch.empty_like(values)
        for indices in positions.values():
            group_values = values[indices] - values[indices].mean()
            if use_std and len(indices) > 1:
                group_values = group_values / (group_values.std() + 1e-6)
            result[indices] = group_values
        return result.tolist()

    rewards = [-1.0, 4.0, 0.0, 7.0, 1.0, 4.0, 9.0]
    groups = [10, 20, 10, 30, 10, 20, 40]
    original = normalize(rewards, groups, True)
    permutation = [6, 2, 5, 0, 3, 1, 4]
    permuted = normalize(
        [rewards[index] for index in permutation],
        [groups[index] for index in permutation],
        True,
    )
    restored = [0.0] * len(permutation)
    for new_position, old_position in enumerate(permutation):
        restored[old_position] = permuted[new_position]
    source = _source(item, "slime/rollout/reward_utils.py")
    facts.update(
        {
            "permutation_invariance_max_abs": max(
                abs(left - right) for left, right in zip(original, restored, strict=True)
            ),
            "singleton_groups_zero": all(
                abs(original[index]) == 0 for index in (3, 6)
            ),
            "each_group_zero_mean": {
                str(group): abs(
                    sum(original[index] for index, value in enumerate(groups) if value == group)
                    / sum(value == group for value in groups)
                )
                for group in sorted(set(groups))
            },
            "missing_group_identity_fails_closed": "group_index is required for reward normalization"
            in source,
            "legacy_uniform_fallback_explicit": "fallback_group_size" in source,
            "uses_explicit_positions_by_group": "positions_by_group[group_index].append(position)"
            in source,
        }
    )
    return facts


def _probe_slime_2198(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    device = _device()
    matrix = []
    for dtype in (torch.float32, torch.bfloat16):
        ppo_kl = torch.tensor(
            [-1000.0, -25.0, -1.0, -0.1, 0.0, 0.1, 1.0, 25.0, 1000.0],
            dtype=dtype,
            device=device,
            requires_grad=True,
        )
        advantages = torch.tensor(
            [1.0, -1.0, 1.0, -2.0, 0.5, 1.0, -1.0, 1.0, -1.0],
            dtype=dtype,
            device=device,
        )
        ratio = (-ppo_kl).float().clamp(-20, 20).exp()
        loss = torch.maximum(
            -ratio * advantages,
            -ratio.clamp(0.8, 1.2) * advantages,
        )
        loss.sum().backward()
        healthy = torch.tensor([-0.1, 0.0, 0.1], dtype=dtype, device=device)
        head_healthy = (-healthy).float().clamp(-20, 20).exp()
        base_healthy = (-healthy).exp()
        matrix.append(
            {
                "dtype": str(dtype),
                "all_extreme_outputs_finite": bool(torch.isfinite(loss).all().item()),
                "all_extreme_gradients_finite": bool(torch.isfinite(ppo_kl.grad).all().item()),
                "healthy_ratio_max_abs": _max_abs(head_healthy, base_healthy),
                "output_dtype": str(loss.dtype),
                "gradient_dtype": str(ppo_kl.grad.dtype),
                "far_tail_zero_gradient_count": int(
                    (ppo_kl.grad[[0, 1, 7, 8]] == 0).sum().item()
                ),
            }
        )
    nan = torch.tensor([float("nan")], device=device)
    source = _source(item, "slime/utils/ppo_utils.py")
    facts.update(
        {
            "dtype_matrix": matrix,
            "nan_remains_visible": bool(
                torch.isnan(nan.float().clamp(-20, 20).exp()).all().item()
            ),
            "clamp_bound": 20.0,
            "helper_casts_to_float32": "return log_ratio.float().clamp" in source,
            "all_policy_ratio_call_sites_changed": "ratio = _clamped_exp(-ppo_kl)"
            in source,
            "low_variance_kl_only_is_clamped": 'if kl_loss_type == "low_var_kl"' in source,
        }
    )
    return facts


def _probe_slime_2152(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    device = _device()
    source = _source(item, "slime/utils/ppo_utils.py")
    namespace: dict[str, Any] = {
        "torch": torch,
        "dist": torch.distributed,
        "_get_vocab_parallel_rank_size": lambda _group: (0, 1),
        "_maybe_all_reduce": lambda *_args, **_kwargs: None,
    }
    exec(
        "from __future__ import annotations\n"
        + _extract_top_level(source, {"_VocabParallelLogProbEntropy"}),
        namespace,
    )
    operation = namespace["_VocabParallelLogProbEntropy"]
    generator = torch.Generator(device="cpu").manual_seed(2152)
    base_logits = torch.randn(9, 17, generator=generator).to(device)
    targets = torch.tensor([0, 1, 2, 3, 4, 8, 12, 15, 16], device=device)
    scenarios = []
    for with_entropy_grad in (False, True):
        logits = base_logits.clone().requires_grad_(True)
        log_prob, entropy = operation.apply(
            logits, targets, None, None, True, with_entropy_grad
        )
        oracle_logits = base_logits.clone().requires_grad_(True)
        oracle_log_prob = oracle_logits.log_softmax(-1).gather(
            1, targets[:, None]
        )
        oracle_entropy = -(oracle_logits.softmax(-1) * oracle_logits.log_softmax(-1)).sum(-1)
        loss = -log_prob.sum()
        oracle_loss = -oracle_log_prob.sum()
        if with_entropy_grad:
            loss = loss + 0.03 * entropy.sum()
            oracle_loss = oracle_loss + 0.03 * oracle_entropy.sum()
        loss.backward(retain_graph=True)
        oracle_loss.backward()
        first_gradient = logits.grad.clone()
        first_error = _max_abs(first_gradient, oracle_logits.grad)
        logits.grad.zero_()
        repeat_error = None
        repeat_exception = None
        try:
            loss.backward()
            repeat_error = _max_abs(logits.grad, first_gradient)
        except Exception as error:
            repeat_exception = f"{type(error).__name__}: {str(error)[:500]}"
        scenarios.append(
            {
                "with_entropy_grad": with_entropy_grad,
                "log_prob_max_abs": _max_abs(log_prob, oracle_log_prob),
                "entropy_max_abs": _max_abs(entropy, oracle_entropy),
                "first_backward_gradient_max_abs": first_error,
                "entropy_requires_grad": entropy.requires_grad,
                "repeat_backward_gradient_max_abs": repeat_error,
                "repeat_backward_exception": repeat_exception,
            }
        )
    facts.update(
        {
            "tp1_value_gradient_and_repeat_matrix": scenarios,
            "backward_mutates_saved_logprob_softmax_inplace": "log_prob_softmax.neg_()"
            in source,
            "metric_only_entropy_marked_non_differentiable": "ctx.mark_non_differentiable(entropy)"
            in source,
            "metric_only_entropy_avoids_full_vocab_saved_tensors": "saved_entropy_softmax = entropy_softmax if with_entropy_grad"
            in source,
            "inplace_softmax_scratch_present": "normalized_logits.exp_()" in source
            and "exp_logits.div_(sum_exp_logits)" in source,
        }
    )
    return facts


def _probe_verl_7012(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)
    source = _source(item, "verl/trainer/distillation/megatron/losses.py")
    alignment_matrix = []
    for cp_size in (1, 2, 4):
        for local_student_length in (1, 3, 8, 17):
            forced_total = local_student_length * cp_size
            alignment_matrix.append(
                {
                    "cp_size": cp_size,
                    "student_local_length": local_student_length,
                    "forced_teacher_total_length": forced_total,
                    "teacher_local_length_after_split": forced_total // cp_size,
                    "aligned": forced_total // cp_size == local_student_length,
                }
            )
    facts.update(
        {
            "shape_alignment_matrix": alignment_matrix,
            "all_integer_local_lengths_align": all(
                row["aligned"] for row in alignment_matrix
            ),
            "forced_length_derived_from_student_and_cp": "student_logits.shape[1] * cp_size"
            in source,
            "forced_length_applied_to_both_teacher_values_and_ids": source.count(
                "forced_max_seqlen=forced_max_seqlen"
            )
            == 2,
            "thd_path_unchanged_by_forced_length": "preprocess_thd_engine(teacher_topk_log_probs, pre_process=True)"
            in source,
            "post_split_shape_assertion_present": "Shape mismatch after CP split" in source,
            "candidate_changed_test_count": facts["changed_test_count"],
        }
    )
    return facts


def _probe_verl_7005(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)
    source = _source(item, "verl/workers/engine/fsdp/transformer_impl.py")
    shard_sizes = [9, 7, 5, 3]
    staged_peak = sum(shard_sizes) + max(shard_sizes)
    streamed_peak = max(shard_sizes) * 2
    facts.update(
        {
            "analytic_elements_live_peak": {
                "whole_shard_staging_then_materialize": staged_peak,
                "per_tensor_stream_only": streamed_peak,
                "reduction": staged_peak - streamed_peak,
            },
            "skip_scope_is_fsdp2_non_peft_only": "fsdp_version(self.module) == 2 and not _is_peft"
            in source,
            "fsdp1_and_lora_keep_staging": "not _skip_staging" in source,
            "per_dtensor_materialization_still_present": ".to(device, non_blocking=True).full_tensor()"
            in source,
            "offload_is_symmetric_with_skipped_load": "self._is_offload_param and not _skip_staging"
            in source,
            "candidate_changed_test_count": facts["changed_test_count"],
        }
    )
    return facts


def _probe_verl_6996(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)
    source = _source(item, "verl/workers/engine/fsdp/transformer_impl.py")
    value_branch_start = source.index('assert self.model_config.model_type == "value_model"')
    value_branch_end = source.index("use_liger =", value_branch_start)
    value_branch = source[value_branch_start:value_branch_end]
    facts.update(
        {
            "language_non_src_uses_empty_init": "with init_empty_weights():" in source
            and "from_config(" in source,
            "only_rank0_builds_full_state": "module.state_dict() if torch.distributed.get_rank() == 0 else {}"
            in source,
            "value_branch_broadcast_tensor_device": "cpu"
            if 'device="cpu"' in value_branch
            else "not-explicitly-cpu",
            "value_branch_uses_default_process_group": "torch.distributed.broadcast(use_trl, src=0)"
            in value_branch,
            "value_branch_has_backend_aware_cpu_group": bool(
                re.search(r"new_group|backend\s*=\s*[\"']gloo", value_branch)
            ),
            "candidate_changed_test_count": facts["changed_test_count"],
            "async_or_thread_primitive_present": bool(
                re.search(r"async\s+def|async_op\s*=\s*True|Thread|Future", _patches(item))
            ),
        }
    )
    return facts


def _probe_verl_6963(item: dict[str, Any]) -> dict[str, Any]:
    facts = _common_facts(item)
    detach = _source(item, "verl/experimental/fully_async_policy/detach_utils.py")
    rollouter = _source(
        item, "verl/experimental/fully_async_policy/fully_async_rollouter.py"
    )
    trainer = _source(item, "verl/experimental/separation/ray_trainer.py")
    losses = _source(item, "verl/workers/utils/losses.py")
    changed = [detach, rollouter, trainer, losses]
    facts.update(
        {
            "changed_path_guard_matrix": {
                "assembly_missing_guard": "trainer batch is missing rollout_log_probs" in detach,
                "assembly_shape_guard": "rollout_log_probs shape must match responses shape"
                in detach,
                "post_balance_missing_guard": "missing after batch balancing" in detach,
                "rollouter_generation_missing_guard": "returned no rollout_log_probs" in rollouter,
                "separation_debug_missing_guard": "required but missing before rollout-vs-actor debug metrics"
                in trainer,
                "loss_preserves_optional_rollout_log_probs": 'fields.append("rollout_log_probs")'
                in losses,
            },
            "explicit_missing_guard_count": sum(
                source.count("missing") + source.count("returned no") for source in changed
            ),
            "present_empty_not_rejected_by_presence_checks": all(
                '"rollout_log_probs" not in' in source
                for source in (detach, rollouter, trainer)
            ),
            "request_config_checks_both_producer_and_actor_flags": all(
                "calculate_log_probs" in source and "use_rollout_log_probs" in source
                for source in (detach, rollouter, trainer)
            ),
            "candidate_changed_test_count": facts["changed_test_count"],
        }
    )
    return facts


def _probe_verl_6960(item: dict[str, Any]) -> dict[str, Any]:
    torch = _torch()
    facts = _common_facts(item)
    device = _device()
    base = torch.arange(48, dtype=torch.float32, device=device).reshape(6, 8)
    dlogprobs = base[:, ::2]
    dentropy = base.t()[::2, :]
    fixed_logprobs = dlogprobs.contiguous()
    fixed_entropy = dentropy.contiguous()
    source = _source(item, "verl/utils/kernel/linear_cross_entropy.py")
    facts.update(
        {
            "upstream_grad_layouts": {
                "dlogprobs_contiguous_before": dlogprobs.is_contiguous(),
                "dlogprobs_stride_before": list(dlogprobs.stride()),
                "dentropy_contiguous_before": dentropy.is_contiguous(),
                "dentropy_stride_before": list(dentropy.stride()),
                "dlogprobs_contiguous_after": fixed_logprobs.is_contiguous(),
                "dentropy_contiguous_after": fixed_entropy.is_contiguous(),
            },
            "contiguous_copy_value_error": max(
                _max_abs(dlogprobs, fixed_logprobs),
                _max_abs(dentropy, fixed_entropy),
            ),
            "both_kernel_grad_inputs_normalized": "dlogprobs = dlogprobs.contiguous()"
            in source
            and "dentropy = dentropy.contiguous()" in source,
            "normalization_precedes_fused_kernel_call": source.index(
                "dlogprobs = dlogprobs.contiguous()"
            )
            < source.index("kernels.efficient_entropy_backward("),
            "candidate_changed_test_count": facts["changed_test_count"],
        }
    )
    return facts


def _environment() -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
    }
    try:
        import torch

        environment.update(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "torch_cuda_available": torch.cuda.is_available(),
                "gpu_count": torch.cuda.device_count(),
                "gpu_names": [
                    torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
                ],
            }
        )
    except Exception as error:
        environment["torch_error"] = f"{type(error).__name__}: {error}"
    return environment


def _not_implemented(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_evidence": _candidate_evidence(item),
        "source_file_count": len(item["sources"]),
        "patch_sha256": _canonical(_patches(item)),
        "probe_implementation_pending": True,
    }


PROBES = {
    "flashattention-pr-2654": _probe_flashattention_2654,
    "liger-pr-1274": _probe_liger_1274,
    "liger-pr-1268": _probe_liger_1268,
    "liger-pr-1230": _probe_liger_1230,
    "megatron-pr-5808": _probe_megatron_5808,
    "megatron-pr-5798": _probe_megatron_5798,
    "megatron-pr-5743": _probe_megatron_5743,
    "megatron-pr-5742": _probe_megatron_5742,
    "torchtitan-pr-3841": _probe_torchtitan_3841,
    "torchtitan-pr-3897": _probe_torchtitan_3897,
    "torchtitan-pr-3867": _probe_torchtitan_3867,
    "verl-pr-7014": _probe_verl_7014,
    "verl-pr-7013": _probe_verl_7013,
    "verl-pr-6984": _probe_verl_6984,
    "megatron-pr-5819": _probe_megatron_5819,
    "megatron-pr-5761": _probe_megatron_5761,
    "megatron-pr-5724": _probe_megatron_5724,
    "megatron-pr-5714": _probe_megatron_5714,
    "megatron-pr-5710": _probe_megatron_5710,
    "slime-pr-2207": _probe_slime_2207,
    "slime-pr-2205": _probe_slime_2205,
    "slime-pr-2204": _probe_slime_2204,
    "slime-pr-2198": _probe_slime_2198,
    "slime-pr-2152": _probe_slime_2152,
    "verl-pr-7012": _probe_verl_7012,
    "verl-pr-7005": _probe_verl_7005,
    "verl-pr-6996": _probe_verl_6996,
    "verl-pr-6963": _probe_verl_6963,
    "verl-pr-6960": _probe_verl_6960,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path)
    parser.add_argument("--bundle-output", type=Path)
    parser.add_argument("--acquire-only", action="store_true")
    parser.add_argument("--case", choices=sorted(PROBES), action="append")
    args = parser.parse_args()

    selection = _read(args.selection)
    selection_material = selection["selection_material"]
    if selection["selection_lock_sha256"] != _canonical(selection_material):
        raise SystemExit("R13 selection digest mismatch")
    plan = _read(args.plan)
    plan_material = {key: value for key, value in plan.items() if key != "test_plan_sha256"}
    if plan["test_plan_sha256"] != _canonical(plan_material):
        raise SystemExit("R13 plan digest mismatch")
    if plan["selection_lock_sha256"] != selection["selection_lock_sha256"]:
        raise SystemExit("R13 plan/selection binding mismatch")
    source_boundary_frozen = plan.get("frozen_before_source_diff_content_inspection") is True or (
        plan.get("extension_frozen_before_source_diff_content_inspection") is True
        and plan.get("base_contracts_preserved_byte_for_byte") is True
    )
    if not source_boundary_frozen:
        raise SystemExit("R13 plan did not preserve the source-inspection boundary")
    body_boundary_frozen = plan.get("frozen_before_candidate_body_acquisition") is True or (
        plan.get("extension_frozen_before_candidate_body_acquisition") is True
        and plan.get("base_contracts_preserved_byte_for_byte") is True
    )
    if not body_boundary_frozen:
        raise SystemExit("R13 plan did not preserve the candidate-body boundary")
    case_by_id = {item["case_id"]: item for item in selection_material["cases"]}
    selected_ids = args.case or list(PROBES)
    if not set(selected_ids) <= set(case_by_id):
        raise SystemExit("requested probe case is not selected")

    if args.source_bundle:
        bundle = _read(args.source_bundle)
    else:
        bundle = _acquire([case_by_id[case_id] for case_id in selected_ids])
    if args.bundle_output:
        args.bundle_output.parent.mkdir(parents=True, exist_ok=True)
        args.bundle_output.write_text(
            json.dumps(bundle, ensure_ascii=False, sort_keys=True), encoding="utf-8"
        )
    if args.acquire_only:
        print(f"acquired_cases={len(bundle)}")
        print(f"source_bundle_sha256={_canonical(bundle)}")
        return 0

    if set(bundle) != set(case_by_id):
        raise SystemExit("R13 source bundle case set differs from selection")
    environment = _environment()
    environment_sha256 = _canonical(environment)
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
        started = datetime.now(UTC)
        try:
            facts = PROBES[case_id](item)
            if facts.get("probe_implementation_pending"):
                raise RuntimeError("R13 probe implementation pending")
            status = "pass"
            failure_codes: list[str] = []
        except Exception as error:
            facts = {"exception_type": type(error).__name__, "exception": str(error)}
            status = "fail"
            failure_codes = ["R13_TRAINING_CONTRACT_PROBE_FAILED"]
            failures += 1
        material = {
            "schema_version": "0.1",
            "protocol_id": selection_material["protocol_id"],
            "case_id": case_id,
            "selection_lock_sha256": selection["selection_lock_sha256"],
            "test_plan_sha256": plan["test_plan_sha256"],
            "source_bundle_sha256": _canonical(bundle),
            "base_sha": case_by_id[case_id]["base_sha"],
            "head_sha": case_by_id[case_id]["head_sha"],
            "source_digests": source_digests,
            "candidate_evidence": _candidate_evidence(item),
            "candidate_evidence_projection_sha256": _canonical(
                item["candidate_evidence_projection"]
            ),
            "environment": environment,
            "environment_sha256": environment_sha256,
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "probe_status": status,
            "failure_codes": failure_codes,
            "facts": facts,
        }
        payload = {**material, "evidence_sha256": _canonical(material)}
        output = args.output_dir / f"{case_id}.json"
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"{case_id}: {status} {payload['evidence_sha256']}")
    print(f"source_bundle_sha256={_canonical(bundle)}")
    print(f"environment_sha256={environment_sha256}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

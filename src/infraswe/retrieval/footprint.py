from __future__ import annotations

import ast
import re
from collections.abc import Mapping
from pathlib import PurePosixPath

from infraswe.models.retrieval import (
    CandidateFootprint,
    CommunicationFootprint,
    FootprintExtractionRequest,
    MemoryTieringFootprint,
)

_COMMUNICATION_TERMS = re.compile(
    r"(?<![a-z0-9])(?:all[_-]?reduce|all[_-]?gather|reduce[_-]?scatter|broadcast|"
    r"collective|communicator|process[_-]?group|nccl|rccl|uccl|ucc|ucx|nvshmem|"
    r"rdma|transport)(?![a-z0-9])",
    re.IGNORECASE,
)
_MEMORY_TERMS = re.compile(
    r"(?<![a-z0-9])(?:kv[_-]?cache|offload|prefetch|evict(?:ion)?|residency|"
    r"host[_-]?pinned|pageable|optimizer[_-]?state|activation|checkpoint[_-]?staging|"
    r"numa)(?![a-z0-9])",
    re.IGNORECASE,
)
_LIFECYCLE_TERM = re.compile(
    r"(?:init|create|start|close|destroy|shutdown|teardown|free|abort|cancel|"
    r"prefetch|evict|release|flush)",
    re.IGNORECASE,
)
_DISPATCH_TERM = re.compile(
    r"(?:dispatch|register|registry|select|provider|backend|algorithm|protocol|transport)",
    re.IGNORECASE,
)
_FAILURE_TAGS = {
    "deadlock": re.compile(r"\bdeadlock\b", re.IGNORECASE),
    "timeout": re.compile(r"\btime(?:d)?[ _-]?out\b|\bwatchdog\b", re.IGNORECASE),
    "out-of-memory": re.compile(r"\boom\b|out[ _-]of[ _-]memory", re.IGNORECASE),
    "resource-leak": re.compile(r"\b(?:memory|resource|handle)[ _-]?leak\b", re.IGNORECASE),
    "stale-data": re.compile(r"\bstale[ _-]?(?:data|read|version|cache)?\b", re.IGNORECASE),
    "use-after-free": re.compile(r"use[ _-]after[ _-]free|\buaf\b", re.IGNORECASE),
    "partial-copy": re.compile(r"\bpartial[ _-]copy\b", re.IGNORECASE),
    "silent-fallback": re.compile(r"\bsilent[ _-]fallback\b", re.IGNORECASE),
}


def _unique(values) -> list[str]:
    return sorted({value for value in values if value})


def _call_name(node: ast.Call) -> str | None:
    target = node.func
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        parts = [target.attr]
        value = target.value
        while isinstance(value, ast.Attribute):
            parts.append(value.attr)
            value = value.value
        if isinstance(value, ast.Name):
            parts.append(value.id)
        return ".".join(reversed(parts))
    return None


def _python_anchors(text: str) -> tuple[list[str], list[str], list[str]]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return [], [], []
    definitions = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    calls = [
        name
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and (name := _call_name(node)) is not None
    ]
    callers = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if any(isinstance(child, ast.Call) for child in ast.walk(node)):
            callers.append(node.name)
    return _unique(definitions), _unique(calls), _unique(callers)


def _source_symbols(path: str, text: str) -> tuple[list[str], list[str], list[str]]:
    if PurePosixPath(path).suffix == ".py":
        return _python_anchors(text)
    definitions = re.findall(
        r"\b(?:class|struct|enum|fn|def|void|bool|int|size_t|auto)\s+"
        r"([A-Za-z_][A-Za-z0-9_:]*)\s*(?:\(|\{|:)",
        text,
    )
    calls = re.findall(r"\b([A-Za-z_][A-Za-z0-9_:]*)\s*\(", text)
    return _unique(definitions), _unique(calls), []


def _config_keys(text: str) -> list[str]:
    quoted = re.findall(
        r"(?:getenv|environ(?:\.get)?|get_config)\s*\(\s*['\"]([A-Za-z0-9_.-]+)",
        text,
    )
    flags = re.findall(r"['\"]--([a-z][a-z0-9-]{2,})['\"]", text)
    return _unique([*quoted, *(flag.replace("-", "_").upper() for flag in flags)])


def _communication_footprint(text: str, symbols: list[str]) -> CommunicationFootprint:
    lowered = text.lower().replace("-", "_")
    collective_patterns = {
        "all_reduce": r"all_?reduce",
        "all_gather": r"all_?gather",
        "reduce_scatter": r"reduce_?scatter",
        "broadcast": r"broadcast",
        "all_to_all": r"all_?to_?all",
    }
    collectives = [
        name for name, pattern in collective_patterns.items() if re.search(pattern, lowered)
    ]
    if re.search(r"\bnvshmem\b|\bone[_ ]sided\b", lowered):
        family = "one-sided-runtime"
    elif re.search(r"\bprocess_?group\b", lowered):
        family = "framework-process-group"
    elif re.search(r"\bucx\b|\btransport\b", lowered) and not collectives:
        family = "transport-runtime"
    else:
        family = "collective-library"
    protocols = [
        name
        for name, pattern in {
            "simple": r"\bprotocol[_ :=-]*simple\b",
            "ll": r"\bprotocol[_ :=-]*ll\b",
            "ll128": r"\bll128\b",
        }.items()
        if re.search(pattern, lowered)
    ]
    transports = [
        name
        for name in ("p2p", "shm", "rdma", "nvlink", "infiniband", "tcp")
        if re.search(rf"\b{re.escape(name)}\b", lowered)
    ]
    topology = [
        name
        for name in ("ring", "tree", "mesh", "numa", "multi_node", "single_node")
        if re.search(rf"\b{re.escape(name)}\b", lowered)
    ]
    operations = [symbol for symbol in symbols if _COMMUNICATION_TERMS.search(symbol)]
    lifecycle = [symbol for symbol in symbols if _LIFECYCLE_TERM.search(symbol)]
    concurrency = [
        symbol
        for symbol in symbols
        if re.search(r"(?:async|stream|event|queue|worker|concurrent)", symbol, re.I)
    ]
    failure_surfaces = [tag for tag, pattern in _FAILURE_TAGS.items() if pattern.search(text)]
    providers = [
        provider
        for provider in ("nccl", "rccl", "uccl", "ucc", "ucx", "nvshmem")
        if re.search(rf"(?<![a-z0-9]){provider}(?![a-z0-9])", lowered)
    ]
    return CommunicationFootprint(
        family=family,
        collectives=_unique(collectives),
        operations=_unique([*operations, *providers]),
        protocols=_unique(protocols),
        transports=_unique(transports),
        topology_features=_unique(topology),
        communicator_lifecycle=_unique(lifecycle),
        concurrency_surfaces=_unique(concurrency),
        failure_surfaces=_unique(failure_surfaces),
    )


def _memory_object(text: str) -> tuple[str, str, bool]:
    patterns = {
        "kv-cache": r"kv[_ -]?cache|decode[_ -]?state",
        "checkpoint-staging": r"checkpoint|durable[_ -]?stag",
        "training-state": r"optimizer|gradient|trainable[_ -]?param",
        "activation": r"activation|forward.+backward|backward.+forward",
        "weight": r"expert[_ -]?weight|model[_ -]?(?:weight|parameter)",
    }
    matches = [kind for kind, pattern in patterns.items() if re.search(pattern, text, re.I | re.S)]
    if len(matches) != 1:
        return "unresolved", "unresolved", True
    kind = matches[0]
    mutability = {
        "kv-cache": "request-scoped-mutable",
        "weight": "model-revision-read-only",
        "training-state": "step-scoped-mutable",
        "activation": "forward-produced-read-only-until-backward",
        "checkpoint-staging": "versioned-durable-state",
    }[kind]
    return kind, mutability, False


def _memory_footprint(text: str, symbols: list[str]) -> tuple[MemoryTieringFootprint, bool]:
    lowered = text.lower().replace("-", "_")
    object_kind, mutability, unresolved = _memory_object(text)
    destination = "unresolved"
    for tier, pattern in (
        ("host-pinned", r"host_?pinned|pinned_?memory"),
        ("host-pageable", r"host_?pageable|pageable_?memory"),
        ("cxl", r"\bcxl\b"),
        ("nvme", r"\bnvme\b"),
    ):
        if re.search(pattern, lowered):
            destination = tier
            break
    states = re.findall(
        r"\b(?:UNALLOCATED|DEVICE_RESIDENT|EVICTING|HOST_RESIDENT|PREFETCHING|"
        r"DEVICE_READY|INVALIDATED|FAILED|FREED)\b",
        text,
    )
    return (
        MemoryTieringFootprint(
            offload_object_kind=object_kind,
            mutability=mutability,
            source_tier="device",
            destination_tier=destination,
            residency_states=_unique(states),
            transition_symbols=_unique(
                symbol
                for symbol in symbols
                if re.search(r"(?:offload|prefetch|evict|transition|residen)", symbol, re.I)
            ),
            allocator_symbols=_unique(
                symbol
                for symbol in symbols
                if re.search(r"(?:alloc|pool|free|release)", symbol, re.I)
            ),
            prefetch_symbols=_unique(
                symbol for symbol in symbols if re.search(r"prefetch", symbol, re.I)
            ),
            eviction_symbols=_unique(
                symbol for symbol in symbols if re.search(r"evict|offload", symbol, re.I)
            ),
            budget_symbols=_unique(
                symbol
                for symbol in symbols
                if re.search(r"(?:budget|limit|capacity|high_water)", symbol, re.I)
            ),
            copy_stream_symbols=_unique(
                symbol
                for symbol in symbols
                if re.search(r"(?:copy|transfer|dma|stream)", symbol, re.I)
            ),
            event_order_symbols=_unique(
                symbol
                for symbol in symbols
                if re.search(r"(?:event|wait|ready|fence|barrier)", symbol, re.I)
            ),
            version_key_symbols=_unique(
                symbol
                for symbol in symbols
                if re.search(r"(?:version|generation|epoch|step)", symbol, re.I)
            ),
            numa_policy_symbols=_unique(
                symbol for symbol in symbols if re.search(r"numa|affinity", symbol, re.I)
            ),
        ),
        unresolved or destination == "unresolved",
    )


def extract_candidate_footprint(
    request: FootprintExtractionRequest,
    sources: Mapping[str, str],
) -> CandidateFootprint:
    if set(sources) != set(request.files):
        raise ValueError("source map must exactly match the frozen footprint file list")
    combined = "\n".join(f"/* {path} */\n{sources[path]}" for path in sorted(sources))
    definitions: list[str] = []
    calls: list[str] = []
    callers: list[str] = []
    for path in sorted(sources):
        file_definitions, file_calls, file_callers = _source_symbols(path, sources[path])
        definitions.extend(file_definitions)
        calls.extend(file_calls)
        callers.extend(file_callers)
    symbols = _unique([*definitions, *calls])
    communication_signal = bool(_COMMUNICATION_TERMS.search(combined))
    memory_signal = bool(_MEMORY_TERMS.search(combined))
    if request.domain == "auto":
        if communication_signal == memory_signal:
            raise ValueError(
                "automatic footprint domain is ambiguous; split the patch or select a domain"
            )
        domain = "distributed-communication" if communication_signal else "memory-tiering"
    else:
        domain = request.domain
    unresolved_surfaces: list[str] = []
    communication = None
    memory_tiering = None
    if domain == "distributed-communication":
        communication = _communication_footprint(combined, symbols)
    else:
        memory_tiering, unresolved = _memory_footprint(combined, symbols)
        if unresolved:
            unresolved_surfaces.append("unresolved-dynamic-dispatch")
    return CandidateFootprint(
        draft_id=request.draft_id,
        draft_revision=request.draft_revision,
        candidate_sha256=request.candidate_sha256,
        files=sorted(request.files),
        symbols=symbols,
        callers=_unique(callers),
        dispatcher_points=_unique(symbol for symbol in symbols if _DISPATCH_TERM.search(symbol)),
        build_targets=_unique(
            path
            for path in request.files
            if PurePosixPath(path).name
            in {"BUILD", "BUILD.bazel", "CMakeLists.txt", "Makefile", "pyproject.toml", "setup.py"}
        ),
        tests=_unique(
            path
            for path in request.files
            if "test" in PurePosixPath(path).name.lower() or "tests" in PurePosixPath(path).parts
        ),
        config_keys=_config_keys(combined),
        failure_signatures=_unique(
            tag for tag, pattern in _FAILURE_TAGS.items() if pattern.search(combined)
        ),
        resource_lifecycles=_unique(symbol for symbol in symbols if _LIFECYCLE_TERM.search(symbol)),
        workload_cases=_unique(
            symbol for symbol in symbols if re.search(r"(?:bench|workload|scenario)", symbol, re.I)
        ),
        unresolved_surfaces=unresolved_surfaces,
        communication=communication,
        memory_tiering=memory_tiering,
    )

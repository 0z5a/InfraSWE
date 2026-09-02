from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


class TrainingCapabilityError(RuntimeError):
    """A requested semantic capability is explicit and unsupported.

    Adapters use this exception instead of selecting an approximate framework default or
    silently falling back to another graph/kernel path.
    """


@runtime_checkable
class TrainingAdapter(Protocol):
    adapter_id: str
    framework_version: str

    def capabilities(self) -> Mapping[str, Any]: ...

    def normalize_config(self, task: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def build_model(self, fixture: Mapping[str, Any]) -> Any: ...

    def build_data(self, fixture: Mapping[str, Any]) -> Any: ...

    def build_optimizer(self, fixture: Mapping[str, Any]) -> Any: ...

    def run_reference_step(self, batch: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def run_candidate_step(self, batch: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def run_rollout_cycle(self, prompts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]: ...

    def save_checkpoint(self, path: Path) -> Mapping[str, Any]: ...

    def resume_checkpoint(self, path: Path) -> Mapping[str, Any]: ...

    def synchronize_weights(self) -> Mapping[str, Any]: ...

    def collect_callgraph(self) -> Mapping[str, Any]: ...

    def collect_compile_state(self) -> Mapping[str, Any]: ...

    def memory_stats(self) -> Mapping[str, Any]: ...

    def shutdown(self) -> Mapping[str, Any]: ...


REQUIRED_ADAPTER_METHODS = (
    "capabilities",
    "normalize_config",
    "build_model",
    "build_data",
    "build_optimizer",
    "run_reference_step",
    "run_candidate_step",
    "run_rollout_cycle",
    "save_checkpoint",
    "resume_checkpoint",
    "synchronize_weights",
    "collect_callgraph",
    "collect_compile_state",
    "memory_stats",
    "shutdown",
)


def validate_adapter_conformance(adapter: object) -> list[str]:
    """Return stable failure codes for a structural adapter conformance check."""

    failures: list[str] = []
    adapter_id = getattr(adapter, "adapter_id", None)
    framework_version = getattr(adapter, "framework_version", None)
    if not isinstance(adapter_id, str) or not adapter_id:
        failures.append("TRAIN_ADAPTER_ID_MISSING")
    if not isinstance(framework_version, str) or not framework_version:
        failures.append("TRAIN_FRAMEWORK_VERSION_MISSING")
    failures.extend(
        f"TRAIN_ADAPTER_METHOD_MISSING:{name}"
        for name in REQUIRED_ADAPTER_METHODS
        if not callable(getattr(adapter, name, None))
    )
    return failures

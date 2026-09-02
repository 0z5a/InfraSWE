from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from infraswe.models.training import NormalizedTrainingConfig
from infraswe.training.adapter import TrainingCapabilityError


def _torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - optional external runtime
        raise TrainingCapabilityError(
            "native-pytorch requires an exact, locally installed torch runtime"
        ) from error
    return torch


class NativePyTorchAdapter:
    """Hermetic tiny-model eager reference adapter.

    The adapter deliberately exposes only implemented semantics. Online rollout and distributed
    weight synchronization fail explicitly instead of being approximated by eager generation.
    """

    adapter_id = "native-pytorch"

    def __init__(self) -> None:
        try:
            self.framework_version = str(_torch().__version__)
        except TrainingCapabilityError:
            self.framework_version = "unavailable"
        self._config: NormalizedTrainingConfig | None = None
        self._model: Any = None
        self._optimizer: Any = None
        self._last_batch: Mapping[str, Any] | None = None
        self._device = "cpu"
        self._dtype: Any = None

    def capabilities(self) -> Mapping[str, Any]:
        if self.framework_version == "unavailable":
            return {
                "capability_level": "adapter-implemented",
                "runtime_available": False,
                "algorithms": {},
                "reason": "torch is not installed",
            }
        torch = _torch()
        return {
            "capability_level": "adapter-implemented",
            "runtime_available": True,
            "algorithms": {
                "sft": "implemented",
                "grpo-contract": "implemented",
                "dapo-loss-contract": "implemented",
                "muon": "implemented" if hasattr(torch.optim, "Muon") else "unsupported",
            },
            "cuda_available": torch.cuda.is_available(),
            "cell_certified": False,
        }

    def normalize_config(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        self._config = NormalizedTrainingConfig.model_validate(dict(task))
        return self._config.model_dump(mode="json")

    def build_model(self, fixture: Mapping[str, Any]) -> Any:
        torch = _torch()
        vocab_size = int(fixture.get("vocab_size", 32))
        hidden_size = int(fixture.get("hidden_size", 16))
        seed = int(fixture.get("seed", 101))
        self._device = str(fixture.get("device", "cpu"))
        dtype_name = str(fixture.get("dtype", "fp32"))
        dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
        if dtype_name not in dtype_map:
            raise TrainingCapabilityError(f"unsupported native dtype: {dtype_name}")
        if self._device.startswith("cuda") and not torch.cuda.is_available():
            raise TrainingCapabilityError("CUDA fixture requested but CUDA is unavailable")
        self._dtype = dtype_map[dtype_name]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        class TinyCausalLM(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = torch.nn.Embedding(vocab_size, hidden_size)
                self.projection = torch.nn.Linear(hidden_size, vocab_size, bias=False)

            def forward(self, input_ids):
                return self.projection(self.embedding(input_ids))

        self._model = TinyCausalLM().to(device=self._device, dtype=self._dtype)
        return self._model

    def build_data(self, fixture: Mapping[str, Any]) -> Any:
        torch = _torch()
        input_ids = fixture.get("input_ids", [[1, 2, 3, 4], [5, 6, 7, 8]])
        labels = fixture.get("labels", [[2, 3, 4, -100], [6, 7, 8, -100]])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=self._device),
            "labels": torch.tensor(labels, dtype=torch.long, device=self._device),
        }

    def build_optimizer(self, fixture: Mapping[str, Any]) -> Any:
        if self._model is None:
            raise TrainingCapabilityError("build_model must run before build_optimizer")
        torch = _torch()
        optimizer_id = str(fixture.get("optimizer", "adamw"))
        learning_rate = float(fixture.get("learning_rate", 1e-3))
        if optimizer_id == "adamw":
            self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=learning_rate)
        elif optimizer_id == "muon":
            if not hasattr(torch.optim, "Muon"):
                raise TrainingCapabilityError("this torch version does not expose torch.optim.Muon")
            self._optimizer = torch.optim.Muon(self._model.parameters(), lr=learning_rate)
        else:
            raise TrainingCapabilityError(f"unsupported native optimizer: {optimizer_id}")
        return self._optimizer

    def _run_step(self, batch: Mapping[str, Any], *, path_id: str) -> Mapping[str, Any]:
        if self._model is None or self._optimizer is None:
            raise TrainingCapabilityError("model and optimizer must be built before a step")
        torch = _torch()
        self._last_batch = batch
        self._optimizer.zero_grad(set_to_none=True)
        logits = self._model(batch["input_ids"])
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            batch["labels"].reshape(-1),
            ignore_index=-100,
            reduction="mean",
        )
        loss.backward()
        gradients = [
            parameter.grad.detach().float().cpu().reshape(-1).tolist()
            for parameter in self._model.parameters()
            if parameter.grad is not None
        ]
        self._optimizer.step()
        return {
            "path_id": path_id,
            "loss": float(loss.detach().cpu()),
            "logits": logits.detach().float().cpu().reshape(-1).tolist(),
            "gradients": gradients,
            "parameters": [
                parameter.detach().float().cpu().reshape(-1).tolist()
                for parameter in self._model.parameters()
            ],
            "fallback_calls": 0,
        }

    def run_reference_step(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._run_step(batch, path_id="pytorch-eager-reference")

    def run_candidate_step(self, batch: Mapping[str, Any]) -> Mapping[str, Any]:
        return self._run_step(batch, path_id="pytorch-eager-candidate")

    def run_rollout_cycle(self, prompts: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
        raise TrainingCapabilityError(
            "native-pytorch v0.1 implements fixed-rollout contract verification only"
        )

    def save_checkpoint(self, path: Path) -> Mapping[str, Any]:
        if self._model is None or self._optimizer is None:
            raise TrainingCapabilityError("nothing has been initialized for checkpointing")
        torch = _torch()
        payload = {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "normalized_config": self._config.model_dump() if self._config else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)
        return {"path": str(path), "components": sorted(payload)}

    def resume_checkpoint(self, path: Path) -> Mapping[str, Any]:
        if self._model is None or self._optimizer is None:
            raise TrainingCapabilityError("build model and optimizer before resuming")
        torch = _torch()
        payload = torch.load(path, map_location=self._device, weights_only=False)
        self._model.load_state_dict(payload["model"])
        self._optimizer.load_state_dict(payload["optimizer"])
        torch.set_rng_state(payload["cpu_rng"])
        if torch.cuda.is_available() and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return {"path": str(path), "restored": sorted(payload)}

    def synchronize_weights(self) -> Mapping[str, Any]:
        return {"status": "not_applicable", "reason": "single-process eager reference"}

    def collect_callgraph(self) -> Mapping[str, Any]:
        return {
            "path_id": "pytorch-eager-reference",
            "fallback_calls": 0,
            "model_type": type(self._model).__name__ if self._model is not None else None,
        }

    def collect_compile_state(self) -> Mapping[str, Any]:
        return {
            "status": "not_applicable",
            "graph_mode": "eager",
            "reason": "reference adapter does not claim torch.compile evidence",
        }

    def memory_stats(self) -> Mapping[str, Any]:
        torch = _torch()
        if not torch.cuda.is_available():
            return {"status": "not_applicable", "reason": "CUDA is unavailable"}
        return {
            "status": "captured",
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(),
        }

    def shutdown(self) -> Mapping[str, Any]:
        self._last_batch = None
        self._optimizer = None
        self._model = None
        self._device = "cpu"
        self._dtype = None
        try:
            torch = _torch()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except TrainingCapabilityError:
            pass
        return {"status": "complete"}

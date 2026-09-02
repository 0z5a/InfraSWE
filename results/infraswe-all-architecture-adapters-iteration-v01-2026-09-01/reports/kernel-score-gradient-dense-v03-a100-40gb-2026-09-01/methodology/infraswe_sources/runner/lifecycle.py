from __future__ import annotations

from pathlib import Path

from infraswe.io import append_jsonl, atomic_write_json
from infraswe.models.trial import TrialRecord, TrialState


class Lifecycle:
    def __init__(self, record: TrialRecord, run_dir: Path) -> None:
        self.record = record
        self.run_dir = run_dir
        self.audit_path = run_dir / "evidence" / "logs" / "audit.jsonl"
        self.checkpoint_path = run_dir / "protocol.json"
        self.persist()

    def transition(self, state: TrialState, detail: str = "") -> None:
        self.record.transition(state, detail)
        append_jsonl(
            self.audit_path,
            {
                "event": "state_transition",
                "state": state.value,
                "detail": detail,
                "trial_id": self.record.trial_id,
            },
        )
        self.persist()

    def persist(self) -> None:
        atomic_write_json(
            self.checkpoint_path,
            self.record.model_dump(mode="json"),
        )

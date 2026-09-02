from dataclasses import dataclass


@dataclass(frozen=True)
class AnchorObservation:
    agent: str
    task: str
    stable_resolved: bool

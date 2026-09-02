from __future__ import annotations

from infraswe.models.score import InfraComponents
from infraswe.scoring.infra import infra_score


def test_infra_extension_weights_match_protocol() -> None:
    score = infra_score(
        InfraComponents(
            slo_goodput=1,
            fault_recovery=1,
            safety_rollback=1,
            resource_efficiency=1,
            topology_robustness=1,
            observability=1,
        )
    )
    assert score == 100


def test_infra_extension_is_not_peak_throughput_only() -> None:
    score = infra_score(
        InfraComponents(
            slo_goodput=1,
            fault_recovery=0,
            safety_rollback=0,
            resource_efficiency=1,
            topology_robustness=1,
            observability=0,
        )
    )
    assert score == 50

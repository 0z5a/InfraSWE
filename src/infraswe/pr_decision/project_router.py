from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from infraswe.models.draft import Digest
from infraswe.pr_decision.contracts import DecisionPlaneModel

FORBIDDEN_PROJECT_FEATURES = {
    "author",
    "author_association",
    "closed_at",
    "merge_commit_sha",
    "merged",
    "merged_at",
    "oracle_label",
    "pr_number",
}


class ProjectDecisionProfile(DecisionPlaneModel):
    profile_id: str
    project: str
    profile_digest: Digest
    training_support: int = Field(ge=0)
    module_contracts: dict[str, list[str]] = Field(default_factory=dict)
    routing_features: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def profile_has_no_identity_or_outcome_features(self) -> ProjectDecisionProfile:
        leaked = FORBIDDEN_PROJECT_FEATURES.intersection(self.routing_features)
        if leaked:
            raise ValueError(f"project profile contains forbidden features: {sorted(leaked)}")
        return self


class ProjectRoute(DecisionPlaneModel):
    project: str
    route_kind: Literal["project-conditioned", "shared-fallback"]
    selected_profile_digest: Digest
    reason: str


def route_project_profile(
    *,
    project: str,
    shared_profile_digest: Digest,
    profiles: dict[str, ProjectDecisionProfile],
    minimum_project_support: int,
) -> ProjectRoute:
    profile = profiles.get(project)
    if profile is None or profile.training_support < minimum_project_support:
        return ProjectRoute(
            project=project,
            route_kind="shared-fallback",
            selected_profile_digest=shared_profile_digest,
            reason="project profile missing or below frozen support minimum",
        )
    return ProjectRoute(
        project=project,
        route_kind="project-conditioned",
        selected_profile_digest=profile.profile_digest,
        reason="project profile meets frozen support minimum",
    )

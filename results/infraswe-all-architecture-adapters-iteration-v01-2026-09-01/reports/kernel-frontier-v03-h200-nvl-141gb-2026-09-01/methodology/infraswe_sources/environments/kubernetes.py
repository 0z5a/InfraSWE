"""Kubernetes adapter contract; concrete cluster provisioning is provider-specific."""

from dataclasses import dataclass


@dataclass(frozen=True)
class KubernetesEnvironment:
    namespace: str
    context: str

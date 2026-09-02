from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from infraswe.kernel.models import Authority, Scope


class RoleNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    authority: Authority
    scope: Scope
    needs: list[str] = Field(default_factory=list)
    image_digest: str
    timeout_sec: int = Field(gt=0)
    on_error: str
    on_fail: str | None = None


class RoleGraph(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = Field(default="0.3", pattern=r"^0\.3$")
    graph_id: str
    nodes: list[RoleNode]
    edges_digest: str

    @model_validator(mode="after")
    def graph_is_closed_and_acyclic(self) -> RoleGraph:
        by_id = {node.id: node for node in self.nodes}
        if len(by_id) != len(self.nodes):
            raise ValueError("role graph node ids must be unique")
        for node in self.nodes:
            missing = set(node.needs) - by_id.keys()
            if missing:
                raise ValueError(f"role {node.id} has missing dependencies: {sorted(missing)}")

        state: dict[str, int] = {}

        def visit(node_id: str) -> None:
            if state.get(node_id) == 1:
                raise ValueError("role graph must be acyclic")
            if state.get(node_id) == 2:
                return
            state[node_id] = 1
            for dependency in by_id[node_id].needs:
                visit(dependency)
            state[node_id] = 2

        for node_id in by_id:
            visit(node_id)

        for node in self.nodes:
            if node.authority != Authority.SCORE:
                continue
            stack = list(node.needs)
            seen: set[str] = set()
            while stack:
                dependency = stack.pop()
                if dependency in seen:
                    continue
                seen.add(dependency)
                if by_id[dependency].authority == Authority.ADVISORY:
                    raise ValueError("advisory role cannot influence a scorer")
                stack.extend(by_id[dependency].needs)
        return self

    def topological_order(self) -> list[str]:
        by_id = {node.id: node for node in self.nodes}
        output: list[str] = []
        seen: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in seen:
                return
            for dependency in by_id[node_id].needs:
                visit(dependency)
            seen.add(node_id)
            output.append(node_id)

        for node in self.nodes:
            visit(node.id)
        return output

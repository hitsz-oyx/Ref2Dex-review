"""Artifact contract validation independent of any tensor library."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from .artifact import Artifact
from .component import PortSpec


@dataclass(frozen=True)
class Contract:
    type: str
    schema_version: Optional[str] = None
    constraints: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_port(cls, port: PortSpec) -> "Contract":
        constraints = dict(port.constraints)
        if port.unit is not None:
            constraints.setdefault("unit", port.unit)
        if port.shape is not None:
            constraints.setdefault("shape", port.shape)
        if port.dtype is not None:
            constraints.setdefault("dtype", port.dtype)
        return cls(type=port.type, constraints=constraints)

    def validate(self, artifact: Artifact) -> tuple[str, ...]:
        issues = []
        if artifact.type != self.type:
            issues.append(f"type {artifact.type!r} != {self.type!r}")
        if self.schema_version is not None and artifact.schema_version != self.schema_version:
            issues.append(
                f"schema_version {artifact.schema_version!r} != {self.schema_version!r}"
            )
        for key, expected in self.constraints.items():
            actual = artifact.metadata.get(key)
            if actual is not None and actual != expected:
                issues.append(f"constraint {key!r} {actual!r} != {expected!r}")
        return tuple(issues)

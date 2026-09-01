"""Artifact contract validation independent of any tensor library."""
from __future__ import annotations

from dataclasses import dataclass, field
import re
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

    def validate(self, artifact: Artifact, *, require_metadata: bool = True) -> tuple[str, ...]:
        """Return contract mismatches; declared metadata is required by default.

        Legacy adapters can explicitly pass ``require_metadata=False`` while
        they are still upgrading old Artifact producers.
        """
        issues = []
        if artifact.type != self.type:
            issues.append(f"type {artifact.type!r} != {self.type!r}")
        if self.schema_version is not None and artifact.schema_version != self.schema_version:
            issues.append(
                f"schema_version {artifact.schema_version!r} != {self.schema_version!r}"
            )
        for key, expected in self.constraints.items():
            actual = artifact.metadata.get(key)
            if actual is None and require_metadata:
                issues.append(f"constraint {key!r} is missing (expected {expected!r})")
            elif actual is not None and actual != expected:
                issues.append(f"constraint {key!r} {actual!r} != {expected!r}")
        issues.extend(_validate_materialized_value(artifact.value, self.constraints))
        return tuple(issues)


def _validate_materialized_value(value: Any, constraints: Mapping[str, Any]) -> tuple[str, ...]:
    """Check declared shape/dtype against a materialized tensor-like value.

    The base layer deliberately avoids importing torch/numpy.  Any object with
    ``shape`` and ``dtype`` attributes (including torch and numpy arrays) is
    supported; path-only Artifacts remain metadata-only contracts.
    """
    if value is None:
        return ()
    issues: list[str] = []
    expected_shape = constraints.get("shape")
    actual_shape = getattr(value, "shape", None)
    if expected_shape is not None and actual_shape is not None:
        parsed = _parse_shape(expected_shape)
        if parsed is not None:
            actual = tuple(int(dimension) for dimension in actual_shape)
            if len(actual) != len(parsed):
                issues.append(f"shape {actual!r} has rank {len(actual)}, expected {expected_shape!r}")
            else:
                for index, (token, dimension) in enumerate(zip(parsed, actual)):
                    if token.isdigit() and int(token) != dimension:
                        issues.append(
                            f"shape dimension {index}={dimension} does not match {token} in {expected_shape!r}"
                        )
    expected_dtype = constraints.get("dtype")
    actual_dtype = getattr(value, "dtype", None)
    if expected_dtype is not None and actual_dtype is not None:
        expected = _canonical_dtype(expected_dtype)
        actual = _canonical_dtype(actual_dtype)
        if expected is not None and actual is not None and expected != actual:
            issues.append(f"dtype {actual!r} != {expected!r}")
    return tuple(issues)


def _parse_shape(value: Any) -> tuple[str, ...] | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if len(text) >= 2 and text[0] in "[({" and text[-1] in ")]}":
        text = text[1:-1].strip()
    if not text or text in {"*", "..."}:
        return None
    dimensions = tuple(part.strip() for part in text.split(","))
    if any(not part or part in {"...", "*"} for part in dimensions):
        return None
    if any(not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*|\d+|-\d+", part) for part in dimensions):
        return None
    return dimensions


def _canonical_dtype(value: Any) -> str | None:
    text = str(value).strip().lower()
    if text.startswith("torch."):
        text = text[6:]
    if text.startswith("numpy."):
        text = text[6:]
    aliases = {
        "half": "float16",
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
        "float": "float32",
        "double": "float64",
        "fp64": "float64",
        "long": "int64",
        "int": "int32",
        "uint": "uint32",
    }
    return aliases.get(text, text or None)

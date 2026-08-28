"""通用科研组件协议。

这一层故意不依赖 PyTorch 或任何具体任务。组件只通过 manifest 声明身份、
能力和端口合同；具体 runtime 可以把现有的 BaseRunner、数据适配器或模型包
装成组件。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple
from abc import ABC, abstractmethod

import yaml

from .artifact import Artifact
from .context import ExecutionContext


class ManifestError(ValueError):
    """组件 manifest 不符合通用协议。"""


def resolve_entrypoint(reference: str) -> Any:
    """解析 ``module:attribute`` 或 ``module.attribute``，不实例化对象。"""
    if not isinstance(reference, str) or not reference.strip():
        raise ManifestError("entrypoint must be a non-empty string.")
    if ":" in reference:
        module_name, attribute = reference.rsplit(":", 1)
    else:
        module_name, _, attribute = reference.rpartition(".")
    if not module_name or not attribute:
        raise ManifestError(f"Invalid entrypoint reference: {reference!r}")
    try:
        module = importlib.import_module(module_name)
        return getattr(module, attribute)
    except (ImportError, AttributeError) as exc:
        raise ManifestError(f"Cannot resolve entrypoint {reference!r}: {exc}") from exc


class Component(ABC):
    """所有可执行组件的最小运行时接口。

    组件作者可以直接实现该接口，也可以通过 adapter 包装现有 Task Runner。
    """

    @abstractmethod
    def spec(self) -> "ComponentSpec":
        raise NotImplementedError

    @abstractmethod
    def execute(self, inputs: Mapping[str, Artifact], context: ExecutionContext) -> Mapping[str, Artifact]:
        raise NotImplementedError


@dataclass(frozen=True)
class PortSpec:
    name: str
    type: str
    shape: Optional[str] = None
    dtype: Optional[str] = None
    unit: Optional[str] = None
    optional: bool = False
    constraints: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, name: str, value: Any) -> "PortSpec":
        if isinstance(value, str):
            value = {"type": value}
        if not isinstance(value, Mapping):
            raise ManifestError(f"Port {name!r} must be a mapping or type string.")
        port_type = value.get("type")
        if not isinstance(port_type, str) or not port_type.strip():
            raise ManifestError(f"Port {name!r} must declare a non-empty string type.")
        constraints = value.get("constraints", {})
        if not isinstance(constraints, Mapping):
            raise ManifestError(f"Port {name!r}.constraints must be a mapping.")
        return cls(
            name=name,
            type=port_type,
            shape=_optional_string(value.get("shape"), f"Port {name!r}.shape"),
            dtype=_optional_string(value.get("dtype"), f"Port {name!r}.dtype"),
            unit=_optional_string(value.get("unit"), f"Port {name!r}.unit"),
            optional=bool(value.get("optional", False)),
            constraints=dict(constraints),
        )


@dataclass(frozen=True)
class ComponentSpec:
    api_version: str
    id: str
    version: str
    entrypoint: str
    capabilities: Tuple[str, ...]
    status: str
    inputs: Mapping[str, PortSpec]
    outputs: Mapping[str, PortSpec]
    requires: Mapping[str, Any] = field(default_factory=dict)
    tags: Tuple[str, ...] = ()
    manifest_path: Optional[Path] = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], manifest_path: Optional[Path] = None) -> "ComponentSpec":
        required = ("api_version", "id", "version", "entrypoint")
        missing = [key for key in required if not isinstance(raw.get(key), str) or not raw[key].strip()]
        if missing:
            raise ManifestError(f"Missing required manifest fields: {', '.join(missing)}")
        capabilities = _string_tuple(raw.get("capabilities", ()), "capabilities")
        tags = _string_tuple(raw.get("tags", ()), "tags")
        status = str(raw.get("status", "experimental"))
        if status not in {"experimental", "reference", "active", "deprecated"}:
            raise ManifestError(
                f"Unsupported component status {status!r}; expected experimental/reference/active/deprecated."
            )
        inputs = _ports(raw.get("inputs", {}), "inputs")
        outputs = _ports(raw.get("outputs", {}), "outputs")
        requires = raw.get("requires", {})
        if not isinstance(requires, Mapping):
            raise ManifestError("requires must be a mapping.")
        return cls(
            api_version=raw["api_version"],
            id=raw["id"],
            version=raw["version"],
            entrypoint=raw["entrypoint"],
            capabilities=capabilities,
            status=status,
            inputs=inputs,
            outputs=outputs,
            requires=dict(requires),
            tags=tags,
            manifest_path=manifest_path,
        )


def load_manifest(path: str | Path) -> ComponentSpec:
    """读取并校验一个 ``component.yaml``。"""
    manifest_path = Path(path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "component.yaml"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"Invalid YAML in {manifest_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ManifestError(f"Manifest must contain a mapping: {manifest_path}")
    return ComponentSpec.from_mapping(raw, manifest_path.resolve())


def compatible_ports(output: PortSpec, input_port: PortSpec) -> Tuple[str, ...]:
    """返回 output → input 不兼容的原因；空元组表示兼容。"""
    issues = []
    if output.type != input_port.type:
        issues.append(f"type {output.type!r} != {input_port.type!r}")
    for field_name in ("shape", "dtype", "unit"):
        source = getattr(output, field_name)
        target = getattr(input_port, field_name)
        if source is not None and target is not None and source != target:
            issues.append(f"{field_name} {source!r} != {target!r}")
    for key, target_value in input_port.constraints.items():
        if key in output.constraints and output.constraints[key] != target_value:
            issues.append(
                f"constraint {key!r} {output.constraints[key]!r} != {target_value!r}"
            )
    return tuple(issues)


def _ports(value: Any, field_name: str) -> Dict[str, PortSpec]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ManifestError(f"{field_name} must be a mapping of port names.")
    return {str(name): PortSpec.from_mapping(str(name), port) for name, port in value.items()}


def _string_tuple(value: Any, field_name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    if not isinstance(value, Iterable) or isinstance(value, (bytes, Mapping)):
        raise ManifestError(f"{field_name} must be a list of strings.")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result):
        raise ManifestError(f"{field_name} cannot contain empty values.")
    return result


def _optional_string(value: Any, field_name: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ManifestError(f"{field_name} must be a string or null.")
    return value

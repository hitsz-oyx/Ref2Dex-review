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

    def validate_inputs(self, inputs: Mapping[str, Artifact], *, strict_metadata: bool = False) -> None:
        """Validate required input ports before executing a component."""
        self._validate_ports(inputs, self.spec().inputs, direction="input", strict_metadata=strict_metadata)

    def validate_outputs(self, outputs: Mapping[str, Artifact], *, strict_metadata: bool = False) -> None:
        """Validate required output ports produced by a component."""
        self._validate_ports(outputs, self.spec().outputs, direction="output", strict_metadata=strict_metadata)

    @staticmethod
    def _validate_ports(
        artifacts: Mapping[str, Artifact],
        ports: Mapping[str, "PortSpec"],
        *,
        direction: str,
        strict_metadata: bool,
    ) -> None:
        if not isinstance(artifacts, Mapping):
            raise TypeError(f"Component {direction}s must be a mapping of Artifact objects.")
        missing = [name for name, port in ports.items() if not port.optional and name not in artifacts]
        if missing:
            raise ValueError(f"Missing required {direction}s: {', '.join(sorted(missing))}")
        from .contract import Contract

        for name, artifact in artifacts.items():
            if not isinstance(artifact, Artifact):
                raise TypeError(f"Component {direction} {name!r} must be an Artifact.")
            port = ports.get(name)
            if port is None:
                # Extra fields are allowed for forward-compatible adapters.
                continue
            issues = Contract.from_port(port).validate(artifact, require_metadata=strict_metadata)
            if issues:
                raise ValueError(f"Invalid {direction} {name!r}: {'; '.join(issues)}")

    @abstractmethod
    def execute(self, inputs: Mapping[str, Artifact], context: ExecutionContext | None = None) -> Mapping[str, Artifact]:
        raise NotImplementedError


class TrainableComponent(Component):
    """Optional lifecycle protocol for trainable PyTorch-like Components.

    ``Component`` deliberately stays framework-neutral.  A trainable adapter
    opts into this protocol when it owns parameters and must be built,
    checkpointed, and restored through stable names.  The base class does not
    import torch; duck typing keeps it usable by other tensor runtimes.
    """

    @classmethod
    def build(cls, config: Any, **kwargs: Any) -> "TrainableComponent":
        """Construct the Component from a Task configuration.

        Concrete implementations should keep all Task-specific parameter
        extraction here, instead of making a Pipeline depend on constructor
        details.  ``config`` is intentionally opaque at this shared layer.
        """
        del config, kwargs
        raise NotImplementedError(
            f"Trainable Component {cls.__name__} must implement build(config, **kwargs)."
        )

    def parameter_groups(self, *, prefix: str = "") -> list[dict[str, Any]]:
        """Return optimizer groups with an explicit checkpoint/name prefix."""
        parameters_fn = getattr(self, "parameters", None)
        if not callable(parameters_fn):
            raise TypeError(f"Trainable Component {type(self).__name__} has no parameters() method.")
        parameters = [parameter for parameter in parameters_fn() if getattr(parameter, "requires_grad", True)]
        if not parameters:
            raise ValueError(f"Trainable Component {type(self).__name__} has no trainable parameters.")
        namespace = prefix.strip() or self.checkpoint_namespace()
        return [{"name": namespace, "parameter_prefix": namespace, "params": parameters}]

    def checkpoint_namespace(self) -> str:
        spec = self.spec()
        return f"{spec.id}@{spec.version}"

    def checkpoint_state_dict(self) -> dict[str, Any]:
        """Wrap ``state_dict`` with identity metadata for safe restoration."""
        state_dict_fn = getattr(self, "state_dict", None)
        if not callable(state_dict_fn):
            raise TypeError(f"Trainable Component {type(self).__name__} has no state_dict() method.")
        spec = self.spec()
        return {
            "namespace": self.checkpoint_namespace(),
            "component_id": spec.id,
            "component_version": spec.version,
            "state_dict": state_dict_fn(),
        }

    def load_checkpoint_state_dict(self, payload: Mapping[str, Any], *, strict: bool = True) -> Any:
        """Restore a namespaced payload and reject a different Component."""
        if not isinstance(payload, Mapping):
            raise TypeError("Component checkpoint payload must be a mapping.")
        spec = self.spec()
        expected_namespace = self.checkpoint_namespace()
        if payload.get("namespace") != expected_namespace:
            raise ValueError(
                f"Checkpoint namespace {payload.get('namespace')!r} != {expected_namespace!r}."
            )
        if payload.get("component_id") != spec.id or payload.get("component_version") != spec.version:
            raise ValueError(
                "Checkpoint Component identity does not match "
                f"{spec.id}@{spec.version}."
            )
        state_dict = payload.get("state_dict")
        if not isinstance(state_dict, Mapping):
            raise ValueError("Component checkpoint payload must contain a state_dict mapping.")
        load_state_dict_fn = getattr(self, "load_state_dict", None)
        if not callable(load_state_dict_fn):
            raise TypeError(f"Trainable Component {type(self).__name__} has no load_state_dict() method.")
        return load_state_dict_fn(state_dict, strict=strict)


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
        constraints = dict(constraints)
        coordinate = value.get("coordinate")
        if coordinate is not None:
            if not isinstance(coordinate, str):
                raise ManifestError(f"Port {name!r}.coordinate must be a string or null.")
            constraints.setdefault("coordinate_frame", coordinate)
        return cls(
            name=name,
            type=port_type,
            shape=_optional_string(value.get("shape"), f"Port {name!r}.shape"),
            dtype=_optional_string(value.get("dtype"), f"Port {name!r}.dtype"),
            unit=_optional_string(value.get("unit"), f"Port {name!r}.unit"),
            optional=bool(value.get("optional", False)),
            constraints=constraints,
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
    entrypoint_kind: str = "component"
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
        if status not in {"experimental", "reference", "active", "deprecated", "archived"}:
            raise ManifestError(
                f"Unsupported component status {status!r}; expected experimental/reference/active/deprecated/archived."
            )
        entrypoint_kind = str(raw.get("entrypoint_kind", "component")).strip().lower()
        if entrypoint_kind not in {"component", "task"}:
            raise ManifestError("entrypoint_kind must be 'component' or 'task'.")
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
            entrypoint_kind=entrypoint_kind,
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
        elif key not in output.constraints:
            issues.append(f"constraint {key!r} is not declared by output")
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

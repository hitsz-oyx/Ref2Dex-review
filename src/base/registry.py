"""组件发现、索引和 pipeline 合同检查。"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import yaml

from .component import ComponentSpec, compatible_ports, load_manifest
from .pipeline import PipelineSpec


class RegistryError(ValueError):
    """组件 registry 无法建立或 pipeline 无法解析。"""


class ComponentRegistry:
    """从显式 roots 发现 ``component.yaml``，不导入 entrypoint。"""

    def __init__(self, specs: Iterable[ComponentSpec] = ()) -> None:
        self._specs: Dict[str, ComponentSpec] = {}
        for spec in specs:
            self.add(spec)

    def add(self, spec: ComponentSpec) -> None:
        previous = self._specs.get(spec.id)
        if previous is not None:
            raise RegistryError(
                f"Duplicate component id {spec.id!r}: {previous.manifest_path} and {spec.manifest_path}"
            )
        self._specs[spec.id] = spec

    def get(self, component_id: str) -> ComponentSpec:
        try:
            return self._specs[component_id]
        except KeyError as exc:
            raise RegistryError(f"Unknown component id: {component_id}") from exc

    def list(self, *, status: str | None = None, capability: str | None = None,
             tag: str | None = None) -> List[ComponentSpec]:
        specs = self._specs.values()
        if status is not None:
            specs = (spec for spec in specs if spec.status == status)
        if capability is not None:
            specs = (spec for spec in specs if capability in spec.capabilities)
        if tag is not None:
            specs = (spec for spec in specs if tag in spec.tags)
        return sorted(specs, key=lambda spec: spec.id)

    @classmethod
    def discover(cls, roots: Iterable[str | Path]) -> "ComponentRegistry":
        paths: List[Path] = []
        for root in roots:
            path = Path(root)
            if path.is_file():
                paths.append(path)
            elif path.is_dir():
                paths.extend(sorted(path.rglob("component.yaml")))
            else:
                raise FileNotFoundError(path)
        specs = []
        for path in sorted(set(paths)):
            specs.append(load_manifest(path))
        return cls(specs)

    def check_pipeline(self, path: str | Path) -> List[str]:
        """检查 pipeline 节点和 edge 的端口合同，返回可读错误列表。"""
        pipeline = PipelineSpec.from_yaml(path)
        node_specs: Dict[str, ComponentSpec] = {}
        issues: List[str] = []
        for node in pipeline.nodes:
            node_id = node.id
            if node_id in node_specs:
                issues.append(f"Duplicate pipeline node id: {node_id}")
                continue
            try:
                node_specs[node_id] = self.get(node.component)
            except RegistryError as exc:
                issues.append(str(exc))
        for edge in pipeline.edges:
            source = _parse_port_ref(edge.source)
            target = _parse_port_ref(edge.target)
            if source is None or target is None:
                issues.append("Pipeline edge references must use 'node.port' strings.")
                continue
            source_node, source_port = source
            target_node, target_port = target
            source_spec = node_specs.get(source_node)
            target_spec = node_specs.get(target_node)
            if source_spec is None or target_spec is None:
                continue
            output = source_spec.outputs.get(source_port)
            input_port = target_spec.inputs.get(target_port)
            if output is None:
                issues.append(f"{source_node}.{source_port}: output port does not exist on {source_spec.id}")
            if input_port is None:
                issues.append(f"{target_node}.{target_port}: input port does not exist on {target_spec.id}")
            if output is not None and input_port is not None:
                for reason in compatible_ports(output, input_port):
                    issues.append(f"{source_node}.{source_port} -> {target_node}.{target_port}: {reason}")
        try:
            pipeline.topological_order()
        except ValueError as exc:
            issues.append(str(exc))
        return issues


def _parse_port_ref(value: object) -> Tuple[str, str] | None:
    if not isinstance(value, str) or value.count(".") != 1:
        return None
    node, port = value.split(".")
    if not node or not port:
        return None
    return node, port

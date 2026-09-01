"""组件发现、索引和 pipeline 合同检查。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import yaml

from .component import Component, ComponentSpec, compatible_ports, load_manifest, resolve_entrypoint
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

    def resolve_entrypoint(self, component_id: str):
        """显式解析组件入口；普通 discover/list 不导入组件代码。"""
        spec = self.get(component_id)
        try:
            entrypoint = resolve_entrypoint(spec.entrypoint)
        except ValueError as exc:
            raise RegistryError(str(exc)) from exc
        if spec.entrypoint_kind == "component":
            if not isinstance(entrypoint, type) or not issubclass(entrypoint, Component):
                raise RegistryError(
                    f"Component manifest {spec.id!r} declares entrypoint_kind=component, "
                    f"but {spec.entrypoint!r} is not a Component subclass."
                )
        return entrypoint

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
        visited_dirs: set[Path] = set()
        for root in roots:
            path = Path(root)
            if path.is_file():
                paths.append(path)
            elif path.is_dir():
                # ``Path.rglob`` does not descend through directory symlinks.
                # Compatibility roots intentionally contain symlinks to the
                # Task-local canonical component directories, so walk them
                # explicitly while de-duplicating resolved directories.
                paths.extend(_iter_manifest_paths(path, visited_dirs))
            else:
                raise FileNotFoundError(path)
        specs = []
        seen: set[Path] = set()
        for path in sorted(set(paths)):
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            specs.append(load_manifest(resolved))
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


def check_task_config(
    config: Mapping[str, object],
    *,
    registry_path: str | Path,
    repo_root: str | Path | None = None,
) -> List[str]:
    """Validate a Task's selected Components against its ``components.json``.

    ``components.json`` is the inventory/fact source, while the Task config
    selects one entry per sibling role.  Both flat selections and recursive
    ``children`` are supported.  The function is intentionally side-effect
    free apart from reading JSON/YAML manifests and never imports entrypoints.
    """
    root = Path(repo_root or Path.cwd()).resolve()
    registry_file = Path(registry_path)
    if not registry_file.is_absolute():
        registry_file = root / registry_file
    registry_file = registry_file.resolve()
    task_root = registry_file.parent.parent
    issues: List[str] = []
    try:
        raw = json.loads(registry_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Cannot read Task component registry {registry_file}: {exc}"]
    if not isinstance(raw, Mapping) or not isinstance(raw.get("components"), list):
        return [f"Task component registry {registry_file} must contain a components list."]

    inventory: dict[tuple[str, str, str], Mapping[str, object]] = {}
    for index, entry in enumerate(raw["components"]):
        if not isinstance(entry, Mapping):
            issues.append(f"Registry entry {index} must be a mapping.")
            continue
        key = tuple(str(entry.get(field, "")) for field in ("id", "version", "role"))
        if not all(key):
            issues.append(f"Registry entry {index} must declare id, version and role.")
            continue
        if key in inventory:
            issues.append(f"Duplicate Task registry entry id/version/role: {key!r}.")
            continue
        inventory[key] = entry

    declared_registry = config.get("component_registry")
    if declared_registry:
        declared_file = _resolve_task_path(str(declared_registry), (root, task_root, registry_file.parent))
        if declared_file != registry_file:
            issues.append(
                f"Config component_registry {declared_registry!r} resolves to {declared_file}, "
                f"not {registry_file}."
            )

    selected = config.get("components")
    if not isinstance(selected, (list, tuple)):
        return issues + ["Task config must declare components as a list."]

    def walk(entries: object, scope: str) -> None:
        if not isinstance(entries, (list, tuple)):
            issues.append(f"{scope}.children/components must be a list.")
            return
        seen_roles: set[str] = set()
        for index, entry in enumerate(entries):
            location = f"{scope}[{index}]"
            if not isinstance(entry, Mapping):
                issues.append(f"{location} must be a mapping.")
                continue
            role = str(entry.get("role", "")).strip()
            component_id = str(entry.get("id", "")).strip()
            version = str(entry.get("version", "")).strip()
            manifest = entry.get("manifest")
            if not role or not component_id or not version or not isinstance(manifest, str) or not manifest.strip():
                issues.append(f"{location} must declare non-empty id, version, role and manifest.")
                continue
            if role in seen_roles:
                issues.append(f"{scope} selects duplicate Component role {role!r}.")
            seen_roles.add(role)
            key = (component_id, version, role)
            inventory_entry = inventory.get(key)
            if inventory_entry is None:
                candidates = [candidate for candidate in inventory if candidate[:2] == key[:2]]
                if candidates:
                    issues.append(
                        f"{location} role {role!r} is not registered for {component_id}@{version}; "
                        f"registered roles: {[candidate[2] for candidate in candidates]!r}."
                    )
                else:
                    issues.append(f"{location} selects unregistered Component {key!r}.")
            selected_manifest = _resolve_task_path(manifest, (root, task_root, registry_file.parent))
            if not selected_manifest.is_file():
                issues.append(f"{location} manifest does not exist: {manifest!r} -> {selected_manifest}.")
            else:
                try:
                    selected_spec = load_manifest(selected_manifest)
                except (OSError, ValueError) as exc:
                    issues.append(f"{location} manifest is invalid: {exc}")
                else:
                    if selected_spec.id != component_id:
                        issues.append(
                            f"{location} id {component_id!r} disagrees with manifest id {selected_spec.id!r}."
                        )
                    if selected_spec.version != version:
                        issues.append(
                            f"{location} version {version!r} disagrees with manifest version {selected_spec.version!r}."
                        )
                    if inventory_entry is not None:
                        inventory_manifest = inventory_entry.get("manifest")
                        if not isinstance(inventory_manifest, str) or not inventory_manifest.strip():
                            issues.append(f"Registry entry {key!r} must declare a manifest path.")
                        else:
                            registered_manifest = _resolve_task_path(
                                inventory_manifest,
                                (registry_file.parent, task_root, root),
                            )
                            if registered_manifest != selected_manifest:
                                issues.append(
                                    f"{location} manifest {manifest!r} resolves to {selected_manifest}, "
                                    f"but registry entry resolves to {registered_manifest}."
                                )
                        registered_kind = inventory_entry.get("entrypoint_kind")
                        if registered_kind and str(registered_kind) != selected_spec.entrypoint_kind:
                            issues.append(
                                f"{location} entrypoint_kind {selected_spec.entrypoint_kind!r} disagrees "
                                f"with registry {registered_kind!r}."
                            )
                        registered_prefix = inventory_entry.get("parameter_prefix")
                        selected_prefix = entry.get("parameter_prefix")
                        if registered_prefix is not None and selected_prefix is not None:
                            if str(registered_prefix) != str(selected_prefix):
                                issues.append(
                                    f"{location} parameter_prefix {selected_prefix!r} disagrees "
                                    f"with registry {registered_prefix!r}."
                                )
            children = entry.get("children")
            if children is not None:
                walk(children, f"{location}.children")

    walk(selected, "components")
    return issues


def _resolve_task_path(value: str, anchors: tuple[Path, ...]) -> Path:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    candidates = [(anchor / candidate) for anchor in anchors]
    for path in candidates:
        if path.exists():
            return path.resolve()
    return candidates[0].resolve()


def _parse_port_ref(value: object) -> Tuple[str, str] | None:
    if not isinstance(value, str) or value.count(".") != 1:
        return None
    node, port = value.split(".")
    if not node or not port:
        return None
    return node, port


def _iter_manifest_paths(path: Path, visited_dirs: set[Path]) -> List[Path]:
    """Return component manifests below ``path``, following safe symlinks.

    The component roots are repository-controlled and may contain compatibility
    symlinks.  Resolved-directory tracking prevents accidental symlink cycles
    from making discovery recurse forever while retaining deterministic order.
    """
    resolved = path.resolve()
    if resolved in visited_dirs:
        return []
    visited_dirs.add(resolved)
    manifests: List[Path] = []
    for child in sorted(path.iterdir()):
        if child.is_file() and child.name == "component.yaml":
            manifests.append(child)
        elif child.is_dir():
            manifests.extend(_iter_manifest_paths(child, visited_dirs))
    return manifests

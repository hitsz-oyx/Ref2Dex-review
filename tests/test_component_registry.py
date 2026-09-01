from __future__ import annotations

from pathlib import Path

import pytest

from src.base.artifact import Artifact
from src.base.component import ManifestError, PortSpec, compatible_ports, load_manifest, resolve_entrypoint
from src.base.contract import Contract
from src.base.pipeline import PipelineError, PipelineSpec
from src.base.registry import ComponentRegistry, RegistryError


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_task_agnostic_and_discoverable(tmp_path: Path) -> None:
    for name in ("identity", "scale"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "component.yaml").write_text(
            "api_version: component/v1\n"
            f"id: example.{name}\nversion: 0.1.0\nentrypoint: json:JSONDecoder\n"
            "entrypoint_kind: task\ncapabilities: [execute]\n"
            "status: experimental\ntags: [transform]\n",
            encoding="utf-8",
        )
    registry = ComponentRegistry.discover([tmp_path])
    assert [spec.id for spec in registry.list(capability="execute")] == [
        "example.identity",
        "example.scale",
    ]
    spec = registry.get("example.identity")
    assert spec.status == "experimental"
    assert "transform" in spec.tags


def test_removed_task_manifests_are_not_discoverable() -> None:
    """仓库不再提供根级 Component 目录。"""
    assert not (ROOT / "components").exists()


def test_manifest_rejects_missing_identity(tmp_path: Path) -> None:
    manifest = tmp_path / "component.yaml"
    manifest.write_text("api_version: component/v1\n", encoding="utf-8")
    with pytest.raises(ManifestError, match="Missing required"):
        load_manifest(manifest)


def test_registry_rejects_duplicate_ids(tmp_path: Path) -> None:
    for name in ("a", "b"):
        directory = tmp_path / name
        directory.mkdir()
        (directory / "component.yaml").write_text(
            "api_version: component/v1\n"
            "id: duplicate\n"
            "version: 0.1.0\n"
            "entrypoint: example:Component\n",
            encoding="utf-8",
        )
    with pytest.raises(RegistryError, match="Duplicate component id"):
        ComponentRegistry.discover([tmp_path])


def test_pipeline_contract_check_accepts_compatible_ports(tmp_path: Path) -> None:
    component_root = tmp_path / "component_root"
    for name, ports in (
        ("a", "outputs:\n  out: {type: scalar}\n"),
        ("b", "inputs:\n  inp: {type: scalar}\n"),
    ):
        directory = component_root / name
        directory.mkdir(parents=True)
        (directory / "component.yaml").write_text(
            "api_version: component/v1\n"
            f"id: {name}\nversion: 0.1.0\nentrypoint: json:JSONDecoder\n"
            "entrypoint_kind: task\n"
            + ports,
            encoding="utf-8",
        )
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        "nodes:\n"
        "  - {id: a, component: a}\n"
        "  - {id: b, component: b}\n"
        "edges:\n"
        "  - {from: a.out, to: b.inp}\n",
        encoding="utf-8",
    )
    registry = ComponentRegistry.discover([component_root])
    assert registry.check_pipeline(pipeline) == []


def test_pipeline_contract_check_reports_type_mismatch(tmp_path: Path) -> None:
    component_root = tmp_path / "components"
    component_root.mkdir()
    (component_root / "a").mkdir()
    (component_root / "a" / "component.yaml").write_text(
        "api_version: component/v1\n"
        "id: a\nversion: 0.1.0\nentrypoint: a:Component\n"
        "outputs:\n  out:\n    type: image\n",
        encoding="utf-8",
    )
    (component_root / "b").mkdir()
    (component_root / "b" / "component.yaml").write_text(
        "api_version: component/v1\n"
        "id: b\nversion: 0.1.0\nentrypoint: b:Component\n"
        "inputs:\n  inp:\n    type: table\n",
        encoding="utf-8",
    )
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        "nodes:\n"
        "  - {id: a, component: a}\n"
        "  - {id: b, component: b}\n"
        "edges:\n"
        "  - {from: a.out, to: b.inp}\n",
        encoding="utf-8",
    )
    issues = ComponentRegistry.discover([component_root]).check_pipeline(pipeline)
    assert any("type 'image' != 'table'" in issue for issue in issues)


def test_port_compatibility_allows_unspecified_source_metadata() -> None:
    source = PortSpec(name="value", type="scalar")
    target = PortSpec(name="value", type="scalar", dtype="float32")
    assert compatible_ports(source, target) == ()


def test_artifact_contract_validates_type_and_metadata() -> None:
    contract = Contract(type="point_flow", schema_version="v1", constraints={"unit": "meter"})
    artifact = Artifact(
        type="point_flow",
        schema_version="v1",
        metadata={"unit": "meter"},
    )
    assert contract.validate(artifact) == ()
    artifact.metadata["unit"] = "radian"
    assert "constraint 'unit'" in contract.validate(artifact)[0]


def test_pipeline_topological_order_and_cycle_detection() -> None:
    pipeline = PipelineSpec.from_mapping({
        "id": "toy",
        "nodes": [
            {"id": "a", "component": "example.identity"},
            {"id": "b", "component": "example.scale"},
        ],
        "edges": [{"from": "a.value", "to": "b.value"}],
    })
    assert pipeline.topological_order() == ("a", "b")
    cyclic = PipelineSpec.from_mapping({
        "nodes": [{"id": "a", "component": "a"}, {"id": "b", "component": "b"}],
        "edges": [{"from": "a.out", "to": "b.in"}, {"from": "b.out", "to": "a.in"}],
    })
    with pytest.raises(PipelineError, match="cycle"):
        cyclic.topological_order()


def test_entrypoint_resolution_is_explicit() -> None:
    assert resolve_entrypoint("src.base.artifact:Artifact").__name__ == "Artifact"
    with pytest.raises(ManifestError, match="Cannot resolve"):
        resolve_entrypoint("src.base.artifact:Missing")

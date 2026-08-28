from __future__ import annotations

from pathlib import Path

import pytest

from src.base.artifact import Artifact
from src.base.component import ManifestError, compatible_ports, load_manifest, resolve_entrypoint
from src.base.contract import Contract
from src.base.pipeline import PipelineError, PipelineSpec
from src.base.registry import ComponentRegistry, RegistryError


ROOT = Path(__file__).resolve().parents[1]


def test_manifest_is_task_agnostic_and_discoverable() -> None:
    registry = ComponentRegistry.discover([ROOT / "components/examples"])
    assert [spec.id for spec in registry.list(capability="execute")] == [
        "example.identity",
        "example.scale",
    ]
    spec = registry.get("example.identity")
    assert spec.status == "experimental"
    assert "transform" in spec.tags


def test_ref2dex_tasks_are_read_only_registry_entries() -> None:
    registry = ComponentRegistry.discover([ROOT / "components/ref2dex"])
    assert [spec.id for spec in registry.list(status="active")] == [
        "ref2dex.cm.v1",
        "ref2dex.cmdecoder.pointflow.inspire_f1",
        "ref2dex.correspondence.ptv3_v2",
    ]
    decoder = registry.get("ref2dex.cmdecoder.pointflow.inspire_f1")
    assert decoder.inputs["representation"].shape == "[B,16,256]"
    assert decoder.outputs["hand_flow"].constraints["coordinate_frame"] == "current_wrist"
    pilot = registry.get("ref2dex.cm.inference.v1")
    assert pilot.status == "experimental"
    assert "predict" in pilot.capabilities


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


def test_pipeline_contract_check_accepts_compatible_ports() -> None:
    registry = ComponentRegistry.discover([ROOT / "components"])
    assert registry.check_pipeline(ROOT / "components/examples/example_pipeline.yaml") == []


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
    source = load_manifest(ROOT / "components/examples/identity/component.yaml").outputs["value"]
    target = load_manifest(ROOT / "components/examples/scale/component.yaml").inputs["value"]
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
    assert resolve_entrypoint("components.examples.identity:IdentityComponent").__name__ == "IdentityComponent"
    with pytest.raises(ManifestError, match="Cannot resolve"):
        resolve_entrypoint("src.base.artifact:Missing")


def test_example_component_executes_without_task_specific_runtime() -> None:
    from src.base.context import ExecutionContext

    cls = resolve_entrypoint("components.examples.scale:ScaleComponent")
    result = cls().execute({"value": Artifact(type="scalar", value=3.0)}, ExecutionContext(config={"factor": 2}))
    assert result["value"].value == 6.0


def test_cm_inference_adapter_maps_artifacts_to_model_outputs() -> None:
    import torch

    from components.ref2dex.cm.adapter import CmInferenceComponent
    from src.base.context import ExecutionContext

    class FakeCm(torch.nn.Module):
        def forward(self, batch):
            assert set(batch) == {
                "object_points", "object_normals", "hand_points", "hand_normals",
                "hand_flow", "obj_valid_mask",
            }
            size = batch["object_points"].shape[0]
            device = batch["object_points"].device
            return {
                "cm_tokens": torch.zeros(size, 16, 256, device=device),
                "pred_obj_flow": torch.zeros(size, 512, 3, device=device),
            }

    adapter = CmInferenceComponent(FakeCm())
    inputs = {
        "object_points": Artifact("point_cloud", torch.zeros(2, 512, 3)),
        "object_normals": Artifact("normal_field", torch.zeros(2, 512, 3)),
        "hand_points": Artifact("point_cloud", torch.zeros(2, 1538, 3)),
        "hand_normals": Artifact("normal_field", torch.zeros(2, 1538, 3)),
        "hand_flow": Artifact("point_flow", torch.zeros(2, 1538, 3)),
        "obj_valid_mask": Artifact("validity_mask", torch.ones(2, 512, dtype=torch.bool)),
    }
    outputs = adapter.execute(inputs, ExecutionContext())
    assert outputs["representation"].value.shape == (2, 16, 256)
    assert outputs["object_flow"].metadata["unit"] == "meter"


def test_cm_inference_adapter_rejects_incomplete_batch() -> None:
    import torch

    from components.ref2dex.cm.adapter import CmInferenceComponent
    from src.base.context import ExecutionContext

    with pytest.raises(ValueError, match="missing required inputs"):
        CmInferenceComponent(torch.nn.Identity()).execute({}, ExecutionContext())

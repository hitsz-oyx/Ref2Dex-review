from __future__ import annotations

from pathlib import Path

import pytest

from src.base.component import ManifestError, TrainableComponent, load_manifest
from src.base.artifact import Artifact
from src.base.contract import Contract
from src.base.registry import ComponentRegistry, RegistryError, check_task_config
from src.base.run_manifest import build_run_manifest
from src.task.Cm.dataset.object_v2 import _resolve_coordinate_frame


ROOT = Path(__file__).resolve().parents[1]


def test_removed_task_manifests_are_not_resolved() -> None:
    assert not (ROOT / "components").exists()


def test_registry_rejects_non_component_declared_as_component(tmp_path: Path) -> None:
    manifest = tmp_path / "component.yaml"
    manifest.write_text(
        "api_version: component/v1\n"
        "id: bad\nversion: 0.1.0\nentrypoint: json:JSONDecoder\n"
        "entrypoint_kind: component\n",
        encoding="utf-8",
    )
    registry = ComponentRegistry.discover([manifest])
    with pytest.raises(RegistryError, match="not a Component subclass"):
        registry.resolve_entrypoint("bad")


def test_contract_requires_declared_metadata_by_default() -> None:
    contract = Contract(type="point_flow", constraints={"coordinate_frame": "object_pose_t"})
    assert "is missing" in contract.validate(Artifact(type="point_flow"))[0]
    assert contract.validate(Artifact(type="point_flow"), require_metadata=False) == ()


def test_contract_validates_materialized_shape_and_dtype() -> None:
    import torch

    contract = Contract(
        type="point_flow",
        constraints={"shape": "[B,N,3]", "dtype": "float32"},
    )
    assert contract.validate(
        Artifact(type="point_flow", value=torch.zeros(2, 5, 3, dtype=torch.float32)),
        require_metadata=False,
    ) == ()
    shape_issues = contract.validate(
        Artifact(type="point_flow", value=torch.zeros(2, 5, 2, dtype=torch.float32)),
        require_metadata=False,
    )
    assert any("dimension 2" in issue for issue in shape_issues)
    dtype_issues = contract.validate(
        Artifact(type="point_flow", value=torch.zeros(2, 5, 3, dtype=torch.float64)),
        require_metadata=False,
    )
    assert any("dtype" in issue for issue in dtype_issues)


def test_task_config_registry_check_catches_manifest_drift(tmp_path: Path) -> None:
    registry = tmp_path / "components.json"
    manifest = tmp_path / "component.yaml"
    manifest.write_text(
        "api_version: component/v1\n"
        "id: example.encoder\nversion: 1.0.0\nentrypoint: json:JSONDecoder\n"
        "entrypoint_kind: task\n",
        encoding="utf-8",
    )
    registry.write_text(
        '{"components": [{"id": "example.encoder", "version": "1.0.0", '
        '"role": "encoder", "manifest": "component.yaml", "entrypoint_kind": "task"}]}',
        encoding="utf-8",
    )
    config = {
        "component_registry": str(registry),
        "components": [{
            "id": "example.encoder",
            "version": "1.0.0",
            "role": "decoder",
            "manifest": str(manifest),
        }],
    }
    issues = check_task_config(config, registry_path=registry, repo_root=tmp_path)
    assert any("not registered" in issue for issue in issues)


def test_trainable_component_checkpoint_namespace_is_explicit(tmp_path: Path) -> None:
    import torch

    manifest = tmp_path / "component.yaml"
    manifest.write_text(
        "api_version: component/v1\n"
        "id: example.identity\nversion: 0.1.0\nentrypoint: json:JSONDecoder\n"
        "entrypoint_kind: task\n",
        encoding="utf-8",
    )

    class Tiny(torch.nn.Module, TrainableComponent):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.weight = torch.nn.Parameter(torch.ones(1))

        def spec(self):
            return load_manifest(manifest)

        def execute(self, inputs, context=None):
            return inputs

    first = Tiny()
    payload = first.checkpoint_state_dict()
    assert payload["namespace"] == "example.identity@0.1.0"
    assert first.parameter_groups(prefix="pipeline.encoder")[0]["parameter_prefix"] == "pipeline.encoder"
    second = Tiny()
    second.load_checkpoint_state_dict(payload)
    payload["namespace"] = "other@0.1.0"
    with pytest.raises(ValueError, match="namespace"):
        second.load_checkpoint_state_dict(payload)


def test_object_v2_rejects_mixed_coordinate_root(tmp_path: Path) -> None:
    object_pose = tmp_path / "object"
    legacy = tmp_path / "legacy"
    (object_pose / "shared").mkdir(parents=True)
    (legacy / "shared").mkdir(parents=True)
    (object_pose / "shared" / "obj_pose_world.npy").write_bytes(b"placeholder")
    with pytest.raises(ValueError, match="mixes coordinate frames"):
        _resolve_coordinate_frame([object_pose, legacy])


def test_run_manifest_ignores_its_own_snapshot_pointers(tmp_path: Path) -> None:
    output = tmp_path / "run"
    payload = build_run_manifest(
        task="cm",
        run_name="unit",
        output_dir=output,
        mode="eval",
        config={
            "run_manifest_path": str(output / "run_manifest.json"),
            "config_snapshot_path": str(output / "config.json"),
            "data": {"split_json_path": str(tmp_path / "split.json")},
        },
        metadata={"metadata_snapshot_path": str(output / "metadata.json")},
    )
    assert all(
        "run_manifest_path" not in record["sources"]
        and "config_snapshot_path" not in record["sources"]
        and "metadata_snapshot_path" not in record["sources"]
        for record in payload["input_references"]
    )

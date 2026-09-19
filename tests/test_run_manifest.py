import json
from pathlib import Path

from src.base.run_manifest import build_run_manifest, write_run_manifest, write_run_summary


def test_run_manifest_records_contract_and_input_metadata_without_inlining_metadata(tmp_path: Path) -> None:
    input_manifest = tmp_path / "split.json"
    input_manifest.write_text('{"train": ["a"]}\n', encoding="utf-8")
    checkpoint = tmp_path / "init.pt"
    checkpoint.write_bytes(b"checkpoint")
    payload = build_run_manifest(
        task="Example",
        run_name="example_20260830",
        output_dir=tmp_path / "run",
        mode="train",
        config={"train": {"seed": 7}, "data": {"split_json_path": str(input_manifest)}},
        metadata={
            "schema_name": "example/v1",
            "coordinate_frame": "object_pose_t",
            "num_obj_points": 512,
        },
        config_source=tmp_path / "config.yaml",
        initial_checkpoint=checkpoint,
    )
    assert payload["manifest_schema"] == "ref2dex.run.v1"
    assert payload["seed"] == 7
    assert payload["contract"]["coordinate_frame"] == "object_pose_t"
    assert "dataset_split" not in payload["contract"]
    assert "dataset_metadata" not in payload
    assert "components" not in payload
    assert "component_tree" not in payload
    refs = {item["resolved_path"]: item for item in payload["input_references"]}
    record = refs[str(input_manifest.resolve())]
    assert record["kind"] == "file"
    assert record["size_bytes"] > 0
    assert record["mtime_ns"] > 0
    assert refs[str(checkpoint.resolve())]["sources"] == ["run:initial_checkpoint"]
    assert "sha256" not in record
    assert "config_sha256" not in payload
    assert payload["config_snapshot"].endswith("/config.json")

    output = tmp_path / "run_manifest.json"
    write_run_manifest(output, payload)
    assert json.loads(output.read_text(encoding="utf-8"))["task"] == "Example"


def test_run_summary_is_terminal_user_readable_snapshot(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    (output / "config_eval_20260901_120000.json").write_text("{}\n", encoding="utf-8")
    summary = output / "summary.json"
    write_run_summary(
        summary,
        task="Example",
        run_name="example_20260901_120000",
        output_dir=output,
        mode="eval",
        run_status="COMPLETED",
        work_version="V1.2.3",
        metrics={"test/loss": 0.25},
        global_step=12,
        artifact_paths={"config": output / "config_eval_20260901_120000.json"},
    )
    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert payload["summary_schema"] == "ref2dex.run_summary.v1"
    assert payload["run_id"] == "example_20260901_120000"
    assert payload["run_status"] == "COMPLETED"
    assert payload["work_version"] == "V1.2.3"
    assert payload["metrics"]["test/loss"] == 0.25
    assert payload["artifacts"]["config"].endswith("config_eval_20260901_120000.json")


def test_run_manifest_records_version_and_component_selection(tmp_path: Path) -> None:
    components = [
        {
            "id": "ref2dex.example.encoder",
            "version": "1.2.0",
            "role": "encoder",
            "manifest": "components/example/component.yaml",
            "parameter_paths": ["model", "data"],
        }
    ]
    payload = build_run_manifest(
        task="Example",
        run_name="versioned",
        output_dir=tmp_path / "run",
        mode="train",
        config={
            "work_version": "V1.2.3",
            "operation_category": ["experiment", "diagnostic"],
            "component_registry": "src/task/Example/components/components.json",
            "components": components,
        },
        metadata={},
    )
    assert payload["work_version"] == "V1.2.3"
    assert "guide_version" not in payload
    assert "plan_version" not in payload
    assert "operation_version" not in payload
    assert payload["operation_category"] == ["experiment", "diagnostic"]
    assert payload["component_registry"].endswith("components.json")
    assert payload["components"] == components


def test_run_manifest_does_not_treat_component_ids_as_paths(tmp_path: Path) -> None:
    registry = tmp_path / "components.json"
    registry.write_text("{}\n", encoding="utf-8")
    payload = build_run_manifest(
        task="Example",
        run_name="component-provenance",
        output_dir=tmp_path / "run",
        mode="train",
        config={
            "component_registry": str(registry),
            "components": [{
                "id": "ref2dex.example.coordinate.hand_root",
                "version": "1.0.0",
                "role": "coordinate_transform",
                "manifest": str(tmp_path / "component.yaml"),
            }],
        },
        metadata={
            "coordinate_component_id": "ref2dex.example.coordinate.hand_root",
            "coordinate_component_version": "1.0.0",
        },
    )
    paths = {record["path"] for record in payload["input_references"]}
    assert str(registry) in paths
    assert str(tmp_path / "component.yaml") in paths
    assert "ref2dex.example.coordinate.hand_root" not in paths
    assert "1.0.0" not in paths


def test_run_manifest_builds_recursive_component_tree(tmp_path: Path) -> None:
    components = [
        {
            "id": "example.encoder",
            "version": "1.0.0",
            "role": "encoder",
            "manifest": "encoder/component.yaml",
        },
        {
            "id": "example.pipeline",
            "version": "1.0.0",
            "role": "pipeline",
            "manifest": "pipeline/component.yaml",
            "children": [{
                "id": "example.encoder",
                "version": "1.0.0",
                "role": "encoder",
                "manifest": "encoder/component.yaml",
            }],
        },
    ]
    payload = build_run_manifest(
        task="Example",
        run_name="nested",
        output_dir=tmp_path / "run",
        mode="train",
        config={"components": components},
        metadata={},
    )
    tree = payload["component_tree"]
    assert [node["role"] for node in tree] == ["pipeline"]
    assert tree[0]["children"][0]["id"] == "example.encoder"
    assert "children" not in tree[0]["children"][0]
    assert [item["role"] for item in payload["components"]] == ["encoder", "pipeline"]
    assert all("children" not in item for item in payload["components"])

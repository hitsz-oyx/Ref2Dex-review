from __future__ import annotations

import json
from pathlib import Path

import torch
import yaml

from src.task.ObjectInteractionCmv2.compact_endpoint import (
    CACHE_SCHEMA,
    CompactEndpointDataset,
    CompactEndpointShardWriter,
    pack_compact_endpoint,
    unpack_compact_endpoint,
)
from src.task.ObjectInteractionCmv2.part_se3 import collate_part_se3, part_se3_v112_loss
from src.task.ObjectInteractionCmv2.part_se3_training import (
    build_part_se3_dataset,
    load_part_se3_training_config,
)
from src.task.ObjectInteractionCmv2.tests.test_v1_12_part_se3 import _model, _sample


def _identified_sample(parts: int = 2) -> dict:
    sample = _sample(parts, interacting_parts=(0,))
    count = sample["obj_points"].shape[0]
    sample.update({
        "obj_point_id": torch.arange(count, dtype=torch.int64),
        "stride": torch.tensor(1, dtype=torch.int64),
        "source_frame_id": torch.tensor(100, dtype=torch.int64),
        "next_source_frame_id": torch.tensor(101, dtype=torch.int64),
    })
    return sample


def test_compact_record_removes_full_hand_stream_and_preserves_endpoint_model_path():
    torch.manual_seed(13)
    sample = _identified_sample()
    record = pack_compact_endpoint(sample)
    compact = unpack_compact_endpoint(record)
    assert record["schema_name"] == CACHE_SCHEMA
    assert record["unique_hand_points"] <= record["source_hand_points"]
    assert not any(key in compact for key in ("hand_points", "hand_normals", "hand_flow"))
    assert compact["edge_hand_points"].shape == (64, 32, 3)
    assert record["tensors"]["edge_lookup"].dtype == torch.int16
    assert compact["edge_source_id"].dtype == torch.int16
    assert torch.equal(compact["obj_point_id"], sample["obj_point_id"])

    model = _model().eval()
    with torch.no_grad():
        reference_batch = collate_part_se3([sample])
        compact_batch = collate_part_se3([compact])
        reference = model(reference_batch)
        accelerated = model(compact_batch)
    for key in ("edge_indices", "edge_valid_mask"):
        assert torch.equal(accelerated[key], reference[key])
    for key in ("edge_distances", "contact_features", "delta_xi_part", "obj_flow_pred"):
        assert torch.allclose(accelerated[key], reference[key], atol=1e-6, rtol=1e-6)
    reference_loss = part_se3_v112_loss(reference, reference_batch)
    compact_loss = part_se3_v112_loss(accelerated, compact_batch)
    for key in reference_loss:
        assert torch.allclose(compact_loss[key], reference_loss[key], atol=1e-6, rtol=1e-6)


def test_sharded_dataset_filters_group_split_stride_and_uses_fixed_index(tmp_path):
    root = tmp_path / "cache"
    writer = CompactEndpointShardWriter(
        root, run_id="pilot", git_commit="a" * 40, source_sha256={"source": "b" * 64},
        shard_target_bytes=1 << 20)
    first = _identified_sample(1)
    second = _identified_sample(2)
    second["stride"] = torch.tensor(2)
    second["next_source_frame_id"] = torch.tensor(102)
    writer.add(first, group="grab/mano", split="train")
    writer.add(second, group="grab/mano", split="val")
    writer.finalize()

    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["record_count"] == 2
    assert manifest["validation"]["bad_count"] == 0
    train = CompactEndpointDataset(root, "grab/mano", "train")
    val = CompactEndpointDataset(root, "grab/mano", "val", fixed_stride=2)
    assert len(train) == len(val) == 1
    assert int(train[0]["stride"]) == 1
    assert int(val[0]["stride"]) == 2
    assert not (root / "records").exists()

    configured = {"data_backend": "compact", "compact_cache_root": str(root)}
    selected = build_part_se3_dataset(configured, "grab/mano", "val", fixed_stride=2)
    assert len(selected) == 1


def test_compact_manifest_rejects_unvalidated_cache(tmp_path):
    root = tmp_path / "cache"
    writer = CompactEndpointShardWriter(
        root, run_id="bad", git_commit="a" * 40, source_sha256={},
        shard_target_bytes=1 << 20)
    writer.add(_identified_sample(), group="arctic/mano", split="train")
    writer.finalize(validation_bad_count=1)
    try:
        CompactEndpointDataset(root, "arctic/mano", "train")
    except ValueError as error:
        assert "not validated" in str(error)
    else:
        raise AssertionError("unvalidated compact cache was accepted")


def test_v113_formal_config_requires_explicit_approval_and_compact_root(tmp_path):
    source = Path(__file__).parents[1] / "configs/active/mixed_part_se3_v1_12_ddp.yaml"
    config = yaml.safe_load(source.read_text())
    config.update({
        "schema_name": "object_interaction_cmv2_part_se3_ddp_v1_13",
        "work_version": "V1.13",
        "data_backend": "compact",
        "compact_cache_root": str(tmp_path / "cache"),
        "run_authorization": "not_approved",
    })
    path = tmp_path / "v113.yaml"
    path.write_text(yaml.safe_dump(config))
    try:
        load_part_se3_training_config(path)
    except ValueError as error:
        assert "not approved" in str(error)
    else:
        raise AssertionError("unapproved V1.13 formal config was accepted")
    config["run_authorization"] = "approved"
    path.write_text(yaml.safe_dump(config))
    loaded = load_part_se3_training_config(path)
    assert loaded["work_version"] == "V1.13"
    assert loaded["data_backend"] == "compact"


def test_v113_plan_remains_declared():
    repo = Path(__file__).parents[4]
    assert (repo / "src/task/ObjectInteractionCmv2/docs/plan/V1.13.md").is_file()

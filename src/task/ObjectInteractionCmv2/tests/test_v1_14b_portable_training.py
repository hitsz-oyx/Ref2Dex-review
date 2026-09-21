from __future__ import annotations

from pathlib import Path
import json
import pickle

import numpy as np
import yaml

from src.task.ObjectInteractionCmv2.part_se3_training import SUPPORTED_PART_SE3_GROUPS
from src.task.ObjectInteractionCmv2.part_se3_v114_training import (
    load_v114_config,
    normalized_batch_counts,
)
from src.task.ObjectInteractionCmv2.tools.data.materialize_oakink2_stage3_v114b import (
    build_sequence,
    hand_to_object_distances,
    object_frame_points,
    sample_object_surface,
)
from src.task.ObjectInteractionCmv2.tools.data.build_active_group_split_v114b import build as build_active_split


TASK_ROOT = Path(__file__).parents[1]


def test_v114b_configs_cover_local_two_group_and_portable_six_group_contracts():
    local = load_v114_config(TASK_ROOT / "configs/active/mixed_part_se3_v1_14b_local.yaml")
    assert local["active_groups"] == ["grab/mano", "grab/inspire_f1"]
    assert local["group_weights"] == {"grab/mano": 0.5, "grab/inspire_f1": 0.5}
    full = load_v114_config(TASK_ROOT / "configs/active/mixed_part_se3_v1_14b_six_source.yaml")
    assert tuple(full["active_groups"]) == SUPPORTED_PART_SE3_GROUPS
    assert abs(sum(full["group_weights"].values()) - 1.0) < 1e-12
    assert full["hand_sampling"]["points_per_side"] == 2048


def test_v114b_batch_rounding_is_exact_positive_and_deterministic():
    groups = ["grab/mano", "grab/inspire_f1", "arctic/mano"]
    weights = {"grab/mano": 0.5, "grab/inspire_f1": 0.3, "arctic/mano": 0.2}
    first = normalized_batch_counts(groups, weights, 17, 0)
    second = normalized_batch_counts(groups, weights, 17, 0)
    rotated = normalized_batch_counts(groups, weights, 17, 1)
    assert first == second
    assert sum(first.values()) == sum(rotated.values()) == 17
    assert all(value >= 1 for value in first.values())


def test_v114b_active_split_accepts_matching_grab_variants(tmp_path):
    config = yaml.safe_load((TASK_ROOT / "configs/active/mixed_part_se3_v1_14b_local.yaml").read_text())
    for group in config["active_groups"]:
        variant = group.split("/")[1]
        source = tmp_path / variant
        rows = {split: [] for split in ("train", "val", "test")}
        for split, sequence_id in (("train", "grab/s1/demo_train"), ("val", "grab/s2/demo_val")):
            sequence = source / split / sequence_id.replace("/", "__")
            (sequence / "geometry").mkdir(parents=True)
            (sequence / "geometry/manifest.json").write_text("{}", encoding="utf-8")
            rows[split].append({"id": sequence_id, "path": str(sequence), "dataset": "grab",
                                "hand_variant": variant, "split": split})
        index = source / "index.json"; manifest = source / "cache_manifest.json"
        index.write_text(json.dumps({"sequences": rows}), encoding="utf-8")
        manifest.write_text(json.dumps({"validation": {"bad_count": 0}}), encoding="utf-8")
        config["sources"][group] = {"index": str(index), "manifest": str(manifest)}
    config["split_root"] = str(tmp_path / "unused")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "split"
    counts = build_active_split(config_path, output, "fixture_split")
    assert counts["train"] == {"grab/mano": 1, "grab/inspire_f1": 1}
    index = json.loads((output / "index.json").read_text())
    assert index["active_groups"] == config["active_groups"]


def test_oakink2_portable_stage3_geometry_is_deterministic_and_object_centered(tmp_path):
    mesh = tmp_path / "O02@test" / "model.obj"
    mesh.parent.mkdir()
    mesh.write_text(
        "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\n"
        "f 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n",
        encoding="utf-8",
    )
    points_a, normals_a = sample_object_surface(mesh, 64)
    points_b, normals_b = sample_object_surface(mesh, 64)
    assert points_a.shape == normals_a.shape == (64, 3)
    assert np.array_equal(points_a, points_b)
    assert np.array_equal(normals_a, normals_b)
    assert np.allclose(np.linalg.norm(normals_a, axis=1), 1.0, atol=1e-6)

    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (2, 4, 4)).copy()
    poses[:, :3, 3] = np.asarray([[1, 2, 3], [-1, 0, 2]], dtype=np.float32)
    local = np.asarray([[[0.1, 0.2, 0.3]], [[0.2, 0.1, 0.4]]], dtype=np.float32)
    world = local + poses[:, None, :3, 3]
    assert np.allclose(object_frame_points(world, poses), local, atol=1e-6)
    distances = hand_to_object_distances(np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32), local)
    assert distances.shape == (2, 1)
    assert np.isfinite(distances).all()


def test_oakink2_portable_stage3_record_matches_existing_consumer_contract(tmp_path):
    object_id = "O02@test"
    object_root = tmp_path / "objects"
    mesh = object_root / "object_repair/align_ds" / object_id / "model.obj"
    mesh.parent.mkdir(parents=True)
    mesh.write_text(
        "v 0 0 0\nv .01 0 0\nv 0 .01 0\nv 0 0 .01\n"
        "f 1 2 3\nf 1 4 2\nf 1 3 4\nf 2 4 3\n",
        encoding="utf-8",
    )
    identity = np.eye(4, dtype=np.float32)
    annotation = {
        "mocap_frame_id_list": [10, 11],
        "raw_mano": {10: {}, 11: {}},
        "obj_list": [object_id],
        "obj_transf": {object_id: {10: identity, 11: identity}},
    }
    annotation_path = tmp_path / "sequence.pkl"
    with annotation_path.open("wb") as stream:
        pickle.dump(annotation, stream)

    class FakeReconstructor:
        faces = {"left": np.asarray([[0, 1, 2]]), "right": np.asarray([[0, 1, 2]])}

        def reconstruct(self, raw_mano, frame_ids, side):
            return np.zeros((len(frame_ids), 3, 3), dtype=np.float32), np.zeros((len(frame_ids), 1, 3), dtype=np.float32)

    output = tmp_path / "stage3"
    output.mkdir()
    rows, skipped = build_sequence(
        annotation_path, object_root, output, FakeReconstructor(), sequence_ordinal=0,
        frame_offset=1)
    assert not skipped
    assert {row["side"] for row in rows} == {"left", "right"}
    for row in rows:
        with np.load(row["file"], allow_pickle=False) as record:
            assert record["obj_points"].shape == record["obj_normals"].shape == (4096, 3)
            assert record["raw_frame_id"].tolist() == [11]
            assert record["obj_root_pose_world"].shape == (1, 4, 4)
            assert record["hand_to_obj_min_dist"].shape == (1, 1)
            assert record["hand_normals"].shape == (1, 1, 3)
            assert str(record["coordinate_frame"]) == "object"


def test_v114b_plan_is_final_and_raw_subset_has_canonical_location():
    repo = TASK_ROOT.parents[2]
    plan = TASK_ROOT / "docs/plan/V1.14b.md"
    assert "状态：`FINAL`" in plan.read_text(encoding="utf-8")
    payload = repo / "data/raw_data/ObjectInteractionCmv2/oicmv2_mano_raw_subset_5x5/run_manifest.json"
    assert payload.is_file()

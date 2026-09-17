import json

import numpy as np
import pytest

from src.task.ObjectInteractionCm.visualize_grab import (
    FUTURE_DELTA_MAX,
    TrainingSequence,
    _future_frame,
    _flow_summary,
    build_parser,
    discover_sequences,
)


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, value)


def test_v1_4_bilateral_cache_is_loaded_without_rewriting_source_schema(tmp_path):
    sequence = tmp_path / "sequence"
    geometry = sequence / "geometry"
    geometry.mkdir(parents=True)
    manifest = {
        "schema_name": "ref2dex_object_interaction_cm_bilateral_geometry_v1",
        "sequence_id": "s1/airplane_fly_1",
        "split": "train",
        "source": "inspire_f1",
        "source_dataset": "grab",
        "coordinate_frame": "object_pose_t",
        "hand_side": "bilateral_merged_left_then_right",
        "merged_hand_sides": True,
        "hand_points": 6,
        "effective_fps": 30.0,
    }
    (geometry / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    _save(geometry / "obj_points_pool_world.npy", np.zeros((12, 4096, 3), np.float32))
    _save(geometry / "obj_normals_pool_world.npy", np.zeros((12, 4096, 3), np.float32))
    _save(geometry / "obj_pose_world.npy", np.tile(np.eye(4, dtype=np.float32), (12, 1, 1)))
    _save(geometry / "source_frame_id.npy", np.arange(0, 48, 4, dtype=np.int32))
    _save(geometry / "hand_points_world.npy", np.zeros((12, 6, 3), np.float32))
    _save(geometry / "hand_normals_world.npy", np.zeros((12, 6, 3), np.float32))
    candidate = np.zeros((12, 4096), dtype=bool)
    candidate[0, 3] = True
    _save(geometry / "obj_candidate_mask_5cm.npy", candidate)
    index = {
        "schema_name": "ref2dex_object_interaction_cm_index_v1_2",
        "sequences": {
            "train": [{
                "id": "grab/s1/airplane_fly_1",
                "path": str(sequence),
                "source": "inspire_f1",
                "dataset": "grab",
            }],
            "val": [],
            "test": [],
        },
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    payload, records = discover_sequences(index_path)
    assert payload["schema_name"].endswith("v1_2")
    assert records[0].variant == "inspire_geometric"
    loaded = TrainingSequence(records[0])
    assert loaded.world_frame == "grab_native_world"
    assert loaded.hand_points_count == 6
    assert loaded.active_5cm(0)
    assert not loaded.active_5cm(1)


def test_flow_summary_reports_corresponding_point_displacement_mm():
    current = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
    future = np.asarray([[0.001, 0.0, 0.0], [0.003, 0.0, 0.0]], dtype=np.float32)
    result = _flow_summary(current, future)
    np.testing.assert_allclose(result["median_mm"], 2.0)
    np.testing.assert_allclose(result["p95_mm"], 2.9)
    np.testing.assert_allclose(result["max_mm"], 3.0)


def test_future_visualization_accepts_30_frames_and_rejects_31():
    assert FUTURE_DELTA_MAX == 30
    assert build_parser().parse_args(["--future-delta", "30"]).future_delta == 30
    assert _future_frame(5, 30, 20) == 19
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--future-delta", "31"])


def test_raw_bilateral_mano_triplet_is_adapted_in_memory(tmp_path):
    sequence = tmp_path / "s1" / "airplane_fly_1"
    sequence.mkdir(parents=True)
    frames = 2
    object_points = np.zeros((frames, 4096, 3), dtype=np.float32)
    object_points[:, 0, 0] = 0.01
    shared = {
        "schema_name": np.asarray("ref2dex_cm_sequence_shared"),
        "schema_version": np.asarray("3.0.0"),
        "seq_id": np.asarray("s1/airplane_fly_1"),
        "source_fps": np.asarray(120.0, dtype=np.float32),
        "ds_rate": np.asarray(4, dtype=np.int32),
        "coordinate_frame": np.asarray("world"),
        "raw_frame_id": np.asarray([0, 4], dtype=np.int32),
        "obj_points_world": object_points,
        "obj_normals_world": np.zeros_like(object_points),
        "context_points_world": np.ones((frames, 5, 3), dtype=np.float32),
        "obj_pose_world": np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1)),
    }
    np.savez(sequence / "shared.npz", **shared)
    for side, offset in (("left", -0.02), ("right", 0.02)):
        hand_points = np.zeros((frames, 3, 3), dtype=np.float32)
        hand_points[..., 0] = offset
        candidate = np.zeros((frames, 4096), dtype=bool)
        candidate[0, 0] = side == "left"
        np.savez(
            sequence / f"{side}.npz",
            schema_name=np.asarray("ref2dex_cm_sequence_hand"),
            schema_version=np.asarray("3.0.0"),
            side=np.asarray(side),
            hand_points_world=hand_points,
            hand_normals_world=np.zeros_like(hand_points),
            hand_mesh_vertices_world=np.zeros((frames, 4, 3), dtype=np.float32),
            hand_mesh_faces=np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
            obj_candidate_mask_5cm=candidate,
        )
    index = {
        "schema_name": "ref2dex_object_interaction_cm_index_v1_2",
        "sequences": {
            "train": [{
                "id": "grab/s1/airplane_fly_1",
                "path": str(sequence),
                "source": "mano",
                "variant": "mano_bilateral_raw",
                "dataset": "grab",
            }],
            "val": [],
            "test": [],
        },
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    _, records = discover_sequences(index_path)
    loaded = TrainingSequence(records[0])
    assert loaded.hand_points_all.shape == (frames, 6, 3)
    assert loaded.context_points(0).shape == (5, 3)
    assert loaded.manifest["hand_side"] == "bilateral_merged_left_then_right"
    assert loaded.manifest["viewer_only_adapter"] is True
    assert loaded.effective_fps == 30.0
    assert loaded.source_frame(1) == 4
    assert loaded.active_5cm(0)
    assert not loaded.active_5cm(1)
    np.testing.assert_array_equal(loaded.object_points(0), object_points[0])


def test_raw_arctic_mano_without_rigid_pose_or_mesh_supports_point_viewer(tmp_path):
    sequence = tmp_path / "s01" / "box_use_01"
    sequence.mkdir(parents=True)
    frames = 2
    object_points = np.zeros((frames, 4096, 3), dtype=np.float32)
    np.savez(
        sequence / "shared.npz",
        schema_name=np.asarray("ref2dex_cm_sequence_shared"),
        schema_version=np.asarray("3.0.0"),
        seq_id=np.asarray("s01/box_use_01"),
        source_fps=np.asarray(30.0, dtype=np.float32),
        ds_rate=np.asarray(1, dtype=np.int32),
        coordinate_frame=np.asarray("world"),
        raw_frame_id=np.asarray([0, 1], dtype=np.int32),
        obj_points_world=object_points,
        obj_normals_world=np.zeros_like(object_points),
    )
    for side in ("left", "right"):
        hand_points = np.zeros((frames, 3, 3), dtype=np.float32)
        np.savez(
            sequence / f"{side}.npz",
            schema_name=np.asarray("ref2dex_cm_sequence_hand"),
            schema_version=np.asarray("3.0.0"),
            side=np.asarray(side),
            hand_points_world=hand_points,
            hand_normals_world=np.zeros_like(hand_points),
            obj_candidate_mask_5cm=np.zeros((frames, 4096), dtype=bool),
        )
    index = {
        "schema_name": "ref2dex_object_interaction_cm_index_v1_2",
        "sequences": {
            "train": [{
                "id": "arctic/s01/box_use_01",
                "path": str(sequence),
                "source": "arctic_mano",
                "variant": "mano_bilateral_raw",
                "dataset": "arctic",
            }],
            "val": [],
            "test": [],
        },
    }
    index_path = tmp_path / "index.json"
    index_path.write_text(json.dumps(index), encoding="utf-8")

    _, records = discover_sequences(index_path)
    loaded = TrainingSequence(records[0])
    assert loaded.hand_points_all.shape == (frames, 6, 3)
    assert loaded.manifest["object_pose_available"] is False
    assert loaded.manifest["hand_mesh_available"] is False
    np.testing.assert_array_equal(
        loaded.object_pose_all,
        np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1)),
    )

import json
from pathlib import Path

import numpy as np
import pytest

from src.task.ObjectInteractionCmv2.grab import GrabManoTransitions


def make_cache(root: Path):
    sequence = root / "sequences/train/mano/grab/s1/demo/geometry"
    sequence.mkdir(parents=True)
    metadata = {"schema_name": "ref2dex_object_interaction_cm_bilateral_mano_v1_4",
                "dataset": "grab", "source": "mano", "split": "train",
                "object_representation": "rigid_se3", "coordinate_frame": "object_pose_t",
                "hand_side": "bilateral_merged_left_then_right", "knn_points_per_side": 2048,
                "knn_hand_points": 4096, "object_pool_points": 4096,
                "effective_fps": 30.0, "surface_sampling": {"cross_frame_fixed": True,
                "left_sha256": "left", "right_sha256": "right"}}
    (sequence / "manifest.json").write_text(json.dumps(metadata))
    np.save(sequence / "frame_time.npy", np.array([0, 1/30, 3/30, 4/30], np.float32))
    np.save(sequence / "source_frame_id.npy", np.array([0, 4, 12, 16], np.int32))
    pose = np.tile(np.eye(4, dtype=np.float32), (4, 1, 1))
    pose[:, 0, 3] = np.arange(4) * 0.1
    np.save(sequence / "obj_pose_world.npy", pose)
    object_points = np.zeros((4, 4096, 3), np.float32)
    hand_points = np.zeros_like(object_points)
    for t in range(4):
        object_points[t, :, 0] = pose[t, 0, 3] + t * 0.001
        hand_points[t, :, 0] = pose[t, 0, 3] + t * 0.002
    np.save(sequence / "obj_points_pool_world.npy", object_points)
    np.save(sequence / "obj_normals_pool_world.npy", np.broadcast_to([0, 0, 1], object_points.shape).astype(np.float32))
    np.save(sequence / "knn_hand_points_world.npy", hand_points)
    np.save(sequence / "knn_hand_normals_world.npy", np.broadcast_to([0, 0, 1], hand_points.shape).astype(np.float32))
    index = {"knn_hand_points_per_stream": {"mano": 4096}, "model_object_points": 1024,
             "object_pool_points": 4096,
             "sequences": {"train": [
                 {"dataset": "grab", "source": "mano", "id": "grab/s1/demo", "frame_count": 4},
                 {"dataset": "arctic", "source": "mano", "id": "arctic/ignored", "frame_count": 4},
                 {"dataset": "grab", "source": "inspire", "id": "grab/ignored", "frame_count": 4}],
                 "val": [], "test": []}}
    (root / "index.json").write_text(json.dumps(index))
    (root / "run_manifest.json").write_text(json.dumps({"run_status": "COMPLETED", "result": "SUPPORTED"}))
    return sequence


def test_grab_only_contiguous_pairs_and_current_object_frame(tmp_path):
    sequence = make_cache(tmp_path)
    dataset = GrabManoTransitions(tmp_path / "index.json", tmp_path / "run_manifest.json", "train")
    assert len(dataset) == 2
    assert dataset.dropped_pairs == 1
    assert [dataset[i]["frame_index"] for i in range(2)] == [0, 2]
    assert dataset[0]["sequence_id"] == "grab/s1/demo"
    assert np.isclose(dataset[0]["obj_flow_gt"][0, 0], 0.101, atol=1e-6)
    assert np.isclose(dataset[0]["hand_flow"][0, 0], 0.102, atol=1e-6)
    metadata = json.loads((sequence / "manifest.json").read_text())
    metadata["hand_side"] = "right_then_left"
    (sequence / "manifest.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Incompatible GRAB cache"):
        GrabManoTransitions(tmp_path / "index.json", tmp_path / "run_manifest.json", "train")

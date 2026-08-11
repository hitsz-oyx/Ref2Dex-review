from pathlib import Path

import numpy as np

from src.task.InteractionDynamics.dataset import InteractionDynamicsDataset


def _write_sequence(root: Path, t: int = 10) -> Path:
    sequence = root / "s1" / "synthetic"
    sequence.mkdir(parents=True)
    obj0 = np.zeros((4096, 3), np.float32)
    obj0[:, 0] = np.arange(4096) / 4096
    hand0 = np.zeros((1538, 3), np.float32)
    shared = {
        "schema_name": np.asarray("ref2dex_interaction_dynamics_shared"),
        "schema_version": np.asarray("1.0.0"), "coordinate_frame": np.asarray("world"),
        "seq_id": np.asarray("s1/synthetic"), "raw_frame_id": np.arange(t) * 4,
        "ds_rate": np.asarray(4), "source_fps": np.asarray(120.0),
        "obj_points_world": np.stack([obj0 + [0, k * .001, 0] for k in range(t)]).astype(np.float32),
        "obj_normals_world": np.tile([[[0, 0, 1]]], (t, 4096, 1)).astype(np.float32),
        "obj_root_pose_world": np.tile(np.eye(4, dtype=np.float32), (t, 1, 1)),
        "obj_point_id": np.arange(4096),
    }
    hand = {
        "schema_name": np.asarray("ref2dex_interaction_dynamics_hand"),
        "schema_version": np.asarray("1.0.0"), "side": np.asarray("right"),
        "hand_points_world": np.stack([hand0 + [k * .002, 0, 0] for k in range(t)]).astype(np.float32),
        "hand_normals_world": np.tile([[[0, 0, 1]]], (t, 1538, 1)).astype(np.float32),
        "hand_root_pose_world": np.tile(np.eye(4, dtype=np.float32), (t, 1, 1)),
        "hand_cano_points": hand0,
        "obj_candidate_mask_5cm": np.ones((t, 4096), bool),
    }
    np.savez(sequence / "shared.npz", **shared)
    np.savez(sequence / "right.npz", **hand)
    return sequence / "right.npz"


def test_chunk_shapes_and_correspondence(tmp_path):
    path = _write_sequence(tmp_path)
    dataset = InteractionDynamicsDataset(tmp_path, file_list=[path])
    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["world_obj_points_object"].shape == (4096, 3)
    assert sample["hand_disp_chunk"].shape == (8, 1538, 3)
    assert sample["hand_articulation_increment_gt"].shape == (8, 1538, 3)
    assert sample["hand_root_increment_pose_gt"].shape == (8, 4, 4)
    assert sample["action_hand_points_local_sequence"].shape == (9, 1538, 3)
    assert sample["action_hand_points_object_sequence"].shape == (9, 1538, 3)
    assert sample["action_hand_root_increment_pose"].shape == (8, 4, 4)
    assert sample["action_hand_cano_points"].shape == (1538, 3)
    assert sample["action_patch_knn_idx"].shape == (64, 32)
    assert sample["hand_disp_chunk_object_gt"].shape == (8, 1538, 3)
    assert sample["action_hand_disp_chunk_object"].shape == (8, 1538, 3)
    assert sample["obj_normals_chunk_object_gt"].shape == (8, 4096, 3)
    assert sample["obj_increment_pose_gt"].shape == (8, 4, 4)
    assert sample["effect_obj_disp_gt"].shape == (8, 512, 3)
    np.testing.assert_allclose(sample["hand_disp_chunk"][0, :, 0], .002, atol=1e-7)
    np.testing.assert_allclose(sample["hand_disp_chunk"][7, :, 0], .016, atol=1e-7)
    np.testing.assert_allclose(sample["hand_articulation_increment_gt"][:, :, 0], .002, atol=1e-7)
    np.testing.assert_allclose(sample["effect_obj_disp_gt"][7, :, 1], .008, atol=1e-7)
    assert sample["future_raw_frame_ids"].tolist() == list(range(4, 36, 4))


def test_dynamic_object_frame_sequence_removes_joint_translation(tmp_path):
    path = _write_sequence(tmp_path)
    shared_path = path.parent / "shared.npz"
    with np.load(shared_path, allow_pickle=False) as payload:
        shared = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(path, allow_pickle=False) as payload:
        hand = {key: np.asarray(payload[key]) for key in payload.files}
    translation = np.arange(10, dtype=np.float32)[:, None] * np.array([[.01, 0, 0]], np.float32)
    shared["obj_root_pose_world"][:, :3, 3] = translation
    shared["obj_points_world"] += translation[:, None]
    hand["hand_points_world"] += translation[:, None]
    np.savez(shared_path, **shared)
    np.savez(path, **hand)
    sequence = InteractionDynamicsDataset(tmp_path, file_list=[path])[0][
        "action_hand_points_object_sequence"]
    expected = np.broadcast_to(
        np.arange(9, dtype=np.float32)[:, None] * .002, sequence[:, :, 0].shape)
    np.testing.assert_allclose(sequence[:, :, 0], expected, atol=1e-6)


def test_max_samples_per_sequence_balances_files(tmp_path):
    first = _write_sequence(tmp_path / "first", t=14)
    second = _write_sequence(tmp_path / "second", t=14)
    dataset = InteractionDynamicsDataset(
        tmp_path, file_list=[first, second], max_samples_per_sequence=3,
    )
    assert len(dataset) == 6
    assert {dataset.sample_location(i)[0].parent for i in range(6)} == {
        first.parent, second.parent,
    }
    for path in (first, second):
        currents = [current for hand_path, current in dataset._samples if hand_path == path]
        assert currents == [0, 2, 5]

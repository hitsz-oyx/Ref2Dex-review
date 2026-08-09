from pathlib import Path

import numpy as np

from src.task.InteractionDynamics.dataset import InteractionDynamicsDataset


def _write_sequence(root: Path) -> Path:
    sequence = root / "s1" / "synthetic"
    sequence.mkdir(parents=True)
    t = 10
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
    assert sample["effect_obj_disp_gt"].shape == (8, 512, 3)
    np.testing.assert_allclose(sample["hand_disp_chunk"][0, :, 0], .002, atol=1e-7)
    np.testing.assert_allclose(sample["hand_disp_chunk"][7, :, 0], .016, atol=1e-7)
    np.testing.assert_allclose(sample["effect_obj_disp_gt"][7, :, 1], .008, atol=1e-7)
    assert sample["future_raw_frame_ids"].tolist() == list(range(4, 36, 4))


def test_max_samples_per_sequence_balances_files(tmp_path):
    first = _write_sequence(tmp_path / "first")
    second = _write_sequence(tmp_path / "second")
    dataset = InteractionDynamicsDataset(
        tmp_path, file_list=[first, second], max_samples_per_sequence=1,
    )
    assert len(dataset) == 2
    assert {dataset.sample_location(i)[0].parent for i in range(2)} == {
        first.parent, second.parent,
    }

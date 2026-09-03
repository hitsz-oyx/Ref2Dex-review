from __future__ import annotations

import numpy as np

from src.task.Cm.dataset.hrdexdb import HrdexdbGeometryDataset


def test_hrdexdb_resamples_hand_surface_and_preserves_flow_correspondence(tmp_path):
    geometry = tmp_path / "episodes" / "inspire_f1" / "unit" / "0" / "geometry"
    geometry.mkdir(parents=True)
    frames, pool = 4, 4096
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    mesh = np.stack([vertices, vertices + np.array([0.1, 0, 0], np.float32)])
    faces = np.array([[0, 1, 2], [0, 1, 3]], np.int32)
    hand = np.zeros((frames, 1538, 3), np.float32)
    obj = np.zeros((frames, pool, 3), np.float32)
    obj[:, :4, 0] = 0.03
    candidate = np.zeros((frames, pool), bool)
    candidate[:, :4] = True
    np.save(geometry / "hand_points_world.npy", hand)
    np.save(geometry / "hand_normals_world.npy", np.zeros_like(hand))
    np.save(geometry / "hand_mesh_vertices_world.npy", mesh)
    np.save(geometry / "hand_mesh_faces.npy", faces)
    np.save(geometry / "obj_points_pool_world.npy", obj)
    np.save(geometry / "obj_normals_pool_world.npy", np.zeros_like(obj))
    np.save(geometry / "obj_candidate_mask_5cm.npy", candidate)
    np.save(geometry / "wrist_pose_world.npy", np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1)))
    obj_pose = np.tile(np.eye(4, dtype=np.float32), (frames, 1, 1))
    obj_pose[0, :3, 3] = [1.0, 2.0, 3.0]
    np.save(geometry / "obj_pose_world.npy", obj_pose)
    np.save(geometry / "frame_time.npy", np.arange(frames, dtype=np.float64) / 30.0)

    ds = HrdexdbGeometryDataset([("inspire_f1/unit/0", geometry.parent)], min_stride=1, max_stride=1, base_seed=5)
    first = ds[0]
    ds.set_epoch(1)
    second = ds[0]
    assert not np.array_equal(first["hand_points"].numpy(), second["hand_points"].numpy())
    assert np.isfinite(first["hand_flow"].numpy()).all()
    # Object coordinates are expressed relative to the current object pose.
    np.testing.assert_allclose(first["obj_points"].numpy()[0], [-0.97, -2.0, -3.0])

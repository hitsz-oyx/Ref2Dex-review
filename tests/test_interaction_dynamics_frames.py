import numpy as np

from src.task.InteractionDynamics.dataset import inverse_transform_points, transform_normals, transform_points


def test_frame_round_trip_and_normal_translation_invariance():
    angle = .4
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    pose[:3, 3] = [1, 2, 3]
    points = np.asarray([[1.2, 2.4, 3.8], [-1, 0, 2]], np.float32)
    np.testing.assert_allclose(inverse_transform_points(transform_points(points, pose), pose), points, atol=1e-6)
    normals = np.asarray([[1, 0, 0]], np.float32)
    moved_pose = pose.copy(); moved_pose[:3, 3] += 100
    np.testing.assert_allclose(transform_normals(normals, pose), transform_normals(normals, moved_pose))

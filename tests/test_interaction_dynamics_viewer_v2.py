import numpy as np

from src.task.InteractionDynamics.viewer_v2.backends import (
    HandFrame, ManoBackend, ObjectTrajectory)


def test_object_trajectory_world_vertices() -> None:
    vertices = np.array([[1., 0., 0.]], np.float32)
    poses = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    poses[1, :3, 3] = (2., 3., 4.)
    trajectory = ObjectTrajectory(vertices, np.zeros((0, 3), np.int32), poses)
    np.testing.assert_allclose(trajectory.world_vertices(1), [[3., 3., 4.]])


def test_mano_backend_returns_parameterization_free_frames() -> None:
    vertices = np.zeros((9, 5, 3), np.float32); faces = np.array([[0, 1, 2]], np.int32)
    backend = ManoBackend()
    prediction = backend.decode_prediction({}, (vertices, faces))
    gt = backend.decode_gt({"gt_mesh": (vertices + 1, faces)})
    assert len(prediction) == 9 and len(gt) == 9
    assert isinstance(prediction[0], HandFrame)
    assert prediction[0].vertices.shape == (5, 3)

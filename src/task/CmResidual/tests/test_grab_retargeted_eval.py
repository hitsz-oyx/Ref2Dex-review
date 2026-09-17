import torch

from src.task.CmResidual.tools.eval_grab_retargeted_base import (
    pose_matrix_xyzw,
    relative_pose,
    rotation_error_rad,
)


def test_pose_matrix_xyzw_and_relative_translation():
    current = pose_matrix_xyzw(torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]]))
    following = pose_matrix_xyzw(torch.tensor([[1.4, 1.5, 3.2, 0.0, 0.0, 0.0, 1.0]]))
    relative = relative_pose(current, following)
    torch.testing.assert_close(relative[0, :3, 3], torch.tensor([0.4, -0.5, 0.2]))
    torch.testing.assert_close(rotation_error_rad(current[:, :3, :3], following[:, :3, :3]), torch.zeros(1))


def test_rotation_error_is_geodesic():
    identity = pose_matrix_xyzw(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]))
    half_turn_z = pose_matrix_xyzw(torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]]))
    torch.testing.assert_close(
        rotation_error_rad(identity[:, :3, :3], half_turn_z[:, :3, :3]),
        torch.tensor([torch.pi]), atol=1e-6, rtol=0,
    )

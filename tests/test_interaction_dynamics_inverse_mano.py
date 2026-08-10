import torch

from src.task.InteractionDynamics.inverse_mano import face_centers, world_to_frame


def test_world_to_frame_and_face_centers_are_differentiable():
    pose = torch.eye(4)
    pose[:3, 3] = torch.tensor([1., 2., 3.])
    vertices = torch.tensor([[[1., 2., 3.], [3., 2., 3.], [1., 4., 3.]]],
                            requires_grad=True)
    faces = torch.tensor([[0, 1, 2]])
    centers = world_to_frame(face_centers(vertices, faces), pose)
    assert torch.allclose(centers, torch.tensor([[[2 / 3, 2 / 3, 0.]]]))
    centers.square().sum().backward()
    assert vertices.grad is not None and vertices.grad.abs().sum() > 0

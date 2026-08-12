import torch
from pytorch3d.transforms import axis_angle_to_matrix

from src.task.InteractionDynamics.mano_hand_transition import ManoHandTransition
from src.task.InteractionDynamics.research.v18_4.build_mano_h_cache import delta_h, hand_state


def test_mano_h_object_frame_translation_and_rotation() -> None:
    orient = torch.tensor([[.1, -.2, .3], [.2, -.1, .4]])
    pose = torch.randn(2, 24); transl = torch.randn(2, 3)
    q = axis_angle_to_matrix(torch.tensor([[.3, 0., 0.], [0., -.2, 0.]]))
    t = torch.randn(2, 1, 3)
    state, rotations = hand_state(orient, pose, transl, q, t)
    recovered_translation = (state[:, None, :3] / 100 - t) @ q.transpose(-1, -2)
    recovered_rotation = q @ rotations
    assert torch.allclose(recovered_translation[:, 0], transl, atol=1e-5)
    assert torch.allclose(recovered_rotation, axis_angle_to_matrix(orient), atol=1e-5)
    assert delta_h(state, rotations).shape == (1, 30)


def test_mano_h_transition_shape_and_gradient() -> None:
    model = ManoHandTransition(dim=64, heads=4, layers=2)
    output = model(torch.randn(2, 128, 4), torch.randn(2, 128, 3),
                   torch.randn(2, 128, 32, 6), torch.randn(2, 33))
    assert output.shape == (2, 8, 30)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

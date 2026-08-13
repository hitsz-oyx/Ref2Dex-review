import torch
from pytorch3d.transforms import axis_angle_to_matrix

from src.task.InteractionDynamics.mano_hand_transition import ManoHandTransition
from src.task.InteractionDynamics.research.v18_4.build_mano_h_cache import delta_h, hand_state
from src.task.InteractionDynamics.interaction_field import build_interaction_y
from src.task.InteractionDynamics.mano_y_decoder_v18_5 import build_interaction_y_batched, pack_batched_future
from src.task.InteractionDynamics.state_interaction_diffusion import pack_state_future
from src.task.InteractionDynamics.train_mano_y_v18_5 import residual_std


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


def test_batched_chunked_y_matches_frozen_teacher_and_has_gradient() -> None:
    torch.manual_seed(3)
    hand = (torch.randn(2, 9, 40, 3) * .03).requires_grad_()
    anchors = torch.randn(2, 13, 3) * .02
    batched = build_interaction_y_batched(hand, anchors, anchor_chunk=5)
    future = pack_batched_future(batched)
    expected = []
    for index in range(2):
        y = {key: 100 * value for key, value in
             build_interaction_y(hand[index], anchors[index], .015).items()}
        expected.append(pack_state_future(y, 8)[1])
    assert torch.allclose(future, torch.stack(expected), atol=2e-5)
    future.square().mean().backward()
    assert hand.grad is not None and torch.isfinite(hand.grad).all()


def test_v18_6_residual_std_is_per_channel() -> None:
    class TinyDataset:
        def __len__(self): return 3
        def __getitem__(self, index):
            scale = torch.arange(1, 57).float()
            return {"residual": torch.randn(5, 56) * scale + index}
    std = residual_std(TinyDataset())
    assert std.shape == (1, 1, 56)
    assert not torch.allclose(std[..., 0], std[..., -1])

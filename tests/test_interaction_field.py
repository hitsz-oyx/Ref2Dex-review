import torch

from src.task.InteractionDynamics.interaction_field import build_interaction_y


def test_interaction_field_shape_and_gradient():
    hand = torch.randn(4, 11, 3, requires_grad=True)
    anchors = torch.randn(7, 3)
    output = build_interaction_y(hand, anchors)
    assert output["interaction_y"].shape == (3, 7, 6)
    assert output["weights"].shape == (4, 7, 11)
    assert output["relative_geometry"].shape == (4, 7, 3)
    assert output["relative_distance"].shape == (4, 7)
    assert output["relative_motion"].shape == (3, 7, 3)
    (output["interaction_y"].square().mean()
     + output["relative_distance"].square().mean()).backward()
    assert hand.grad is not None
    assert torch.isfinite(hand.grad).all()
    assert hand.grad.abs().sum() > 0


def test_interaction_field_is_invariant_to_joint_translation():
    hand = torch.randn(3, 9, 3)
    anchors = torch.randn(5, 3)
    translation = torch.tensor([.3, -.2, .7])
    original = build_interaction_y(hand, anchors)["interaction_y"]
    translated = build_interaction_y(hand + translation, anchors + translation)["interaction_y"]
    torch.testing.assert_close(original, translated)


def test_soft_distance_does_not_cancel_opposite_vectors():
    hand = torch.tensor([[[-1.0, 0, 0], [1.0, 0, 0]]])
    anchors = torch.zeros(1, 3)
    output = build_interaction_y(torch.cat([hand, hand], dim=0), anchors, tau_m=1.0)
    torch.testing.assert_close(output["relative_geometry"][0], torch.zeros(1, 3))
    torch.testing.assert_close(output["relative_distance"][0], torch.ones(1))

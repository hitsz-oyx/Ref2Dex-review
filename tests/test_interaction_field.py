import torch

from src.task.InteractionDynamics.interaction_field import build_interaction_y


def test_interaction_field_shape_and_gradient():
    hand = torch.randn(4, 11, 3, requires_grad=True)
    anchors = torch.randn(7, 3)
    output = build_interaction_y(hand, anchors)
    assert output["interaction_y"].shape == (3, 7, 6)
    output["interaction_y"].square().mean().backward()
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

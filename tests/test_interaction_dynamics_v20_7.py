import torch

from src.task.InteractionDynamics.research.gendex_gt_baseline.contact_map import (
    build_gendex_contact_value, build_gendex_gt_map,
)


def test_gendex_gt_map_shape_range_and_direction_alignment() -> None:
    object_points = torch.tensor([[0., 0., 0.], [0., 0., 0.]])
    normals = torch.tensor([[1., 0., 0.], [-1., 0., 0.]])
    hand = torch.tensor([[.001, 0., 0.]])
    value = build_gendex_contact_value(object_points, normals, hand)
    assert value.shape == (2,) and bool(((0 <= value) & (value <= 1)).all())
    assert value[0] > value[1]
    assert build_gendex_gt_map(object_points, normals, hand).shape == (2, 7)


def test_gendex_contact_value_has_hand_gradient() -> None:
    hand = torch.tensor([[.01, .002, 0.]], requires_grad=True)
    value = build_gendex_contact_value(torch.zeros(1, 3), torch.tensor([[1., 0., 0.]]), hand)
    value.sum().backward()
    assert hand.grad is not None and torch.isfinite(hand.grad).all() and float(hand.grad.norm()) > 0

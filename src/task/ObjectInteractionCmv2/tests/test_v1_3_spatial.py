from types import SimpleNamespace

import torch

from src.task.ObjectInteractionCmv2.model import (
    CompetitiveContactTokens, ObjectInteractionCmv2V13Model,
    _axis_angle_matrix_stable, object_interaction_v13_loss,
)
from src.task.ObjectInteractionCmv2.synthetic import make_synthetic_batch


def model():
    return ObjectInteractionCmv2V13Model(SimpleNamespace(
        hidden_width=128, num_tokens=16, use_residual=False,
        knn_k=32, interaction_radius_m=0.02, interaction_mode="swept",
        feature_scale_m=0.02, frame_dt_s=1 / 30))


def batch():
    value = make_synthetic_batch(batch_size=2, num_object=8, num_hand=10)
    value["obj_points"] = torch.randn(2, 8, 3) * 0.01
    value["delta_time_s"] = torch.full((2,), 1 / 30)
    value["delta_translation_gt"] = torch.zeros(2, 3)
    value["delta_rotation_gt"] = torch.eye(3).expand(2, 3, 3).clone()
    return value


def test_empty_contact_finite_and_no_gt_leakage():
    value = batch()
    value["hand_valid_mask"][:] = False
    net = model()
    first = net(value)
    assert "p_effect" not in first
    assert not any("effect" in name for name, _ in net.named_parameters())
    assert first["cm_tokens"].shape == (2, 16, 32)
    assert not first["token_mask"].any()
    assert torch.count_nonzero(first["token_mass"]) == 0
    assert torch.count_nonzero(first["cm_tokens"]) == 0
    assert torch.count_nonzero(first["obj_flow_residual"]) == 0
    assert not any("residual_head" in name for name, _ in net.named_parameters())
    value["delta_translation_gt"] += 10
    value["delta_rotation_gt"] *= 0
    assert torch.equal(first["obj_flow_pred"], net(value)["obj_flow_pred"])
    loss = object_interaction_v13_loss(first, batch())["total"]
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_competitive_mass_conservation_and_permutation():
    torch.manual_seed(1)
    tokens = CompetitiveContactTokens(8, 4)
    features = torch.randn(1, 7, 8)
    points = torch.randn(1, 7, 3)
    normals = torch.nn.functional.normalize(torch.randn(1, 7, 3), dim=-1)
    active = torch.tensor([[True, False, True, True, False, True, True]])
    first = tokens(features, points, normals, active, torch.ones(1))
    assert torch.allclose(first[3].sum(), active.sum().float(), atol=1e-6)
    order = torch.tensor([6, 2, 0, 5, 1, 3, 4])
    second = tokens(features[:, order], points[:, order], normals[:, order],
                    active[:, order], torch.ones(1))
    for a, b in zip(first[:4], second[:4]):
        assert torch.allclose(a, b, atol=1e-6)


def test_rotation_map_and_direct_loss_at_zero_and_pi():
    for angle in (0.0, 1e-8, 3.1415925):
        axis_angle = torch.tensor([[angle, 0., 0.]], requires_grad=True)
        rotation = _axis_angle_matrix_stable(axis_angle)
        assert torch.allclose(rotation.transpose(1, 2) @ rotation,
                              torch.eye(3)[None], atol=1e-5)
        rotation.sum().backward()
        assert torch.isfinite(axis_angle.grad).all()
    value = batch()
    predicted = {"delta_xi_root": torch.zeros(2, 6, requires_grad=True),
                 "obj_flow_pred": torch.zeros_like(value["obj_points"])}
    losses = object_interaction_v13_loss(predicted, value)
    assert set(losses) == {"total", "translation", "rotation", "flow"}
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert torch.isfinite(predicted["delta_xi_root"].grad).all()

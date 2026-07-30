from __future__ import annotations

import torch
from torch import nn

from src.task.Cm.model import CmFlowHead, CmFlowModel
from src.task.Cm.runner import internal_flow_smooth_l1


def test_cm_flow_head_uses_full_hand_motion_inputs_and_slot_bottleneck() -> None:
    torch.manual_seed(7)
    head = CmFlowHead(
        dense_token_dim=8,
        cm_dim=16,
        num_cm_tokens=4,
        num_slot_iters=2,
    )
    batch_size, num_obj, num_hand = 2, 5, 7
    output = head(
        z_obj=torch.randn(batch_size, num_obj, 8),
        z_hand=torch.randn(batch_size, num_hand, 8),
        dense_hand_contact=torch.rand(batch_size, num_hand),
        obj_points=torch.randn(batch_size, num_obj, 3),
        obj_normals=torch.randn(batch_size, num_obj, 3),
        hand_points=torch.randn(batch_size, num_hand, 3),
        hand_normals=torch.randn(batch_size, num_hand, 3),
        hand_flow=torch.randn(batch_size, num_hand, 3),
        obj_valid_mask=torch.tensor([[True, True, False, True, False], [True] * num_obj]),
    )

    assert head.hand_motion_encoder[0].in_features == 8 + 10
    assert output["cm_tokens"].shape == (batch_size, 4, 16)
    assert output["cm_assignment"].shape == (batch_size, 4, num_hand)
    assert output["cm_slot_weights"].shape == (batch_size, 4, num_hand)
    assert output["decoder_slot_usage"].shape == (batch_size, 4)
    assert output["cm_anchor_pos"].shape == (batch_size, 4, 3)
    assert output["cm_anchor_normal"].shape == (batch_size, 4, 3)
    assert output["pred_obj_flow"].shape == (batch_size, num_obj, 3)
    torch.testing.assert_close(output["cm_assignment"].sum(dim=1), torch.ones(batch_size, num_hand))
    torch.testing.assert_close(output["cm_slot_weights"].sum(dim=-1), torch.ones(batch_size, 4))
    torch.testing.assert_close(output["decoder_slot_usage"].sum(dim=-1), torch.ones(batch_size))
    # The zero-initialized edge head starts with equal decoder attention over
    # slots, including when an object-valid mask excludes some points.
    torch.testing.assert_close(output["decoder_slot_usage"], torch.full((batch_size, 4), 0.25))
    torch.testing.assert_close(
        torch.linalg.vector_norm(output["cm_anchor_normal"], dim=-1),
        torch.ones(batch_size, 4),
    )
    torch.testing.assert_close(output["pred_obj_flow"][0, ~torch.tensor([True, True, False, True, False])], torch.zeros(2, 3))
    assert torch.isfinite(output["cm_tokens"]).all()

    repeated_inputs = {
        "z_obj": torch.randn(batch_size, num_obj, 8),
        "z_hand": torch.randn(batch_size, num_hand, 8),
        "dense_hand_contact": torch.rand(batch_size, num_hand),
        "obj_points": torch.randn(batch_size, num_obj, 3),
        "obj_normals": torch.randn(batch_size, num_obj, 3),
        "hand_points": torch.randn(batch_size, num_hand, 3),
        "hand_normals": torch.randn(batch_size, num_hand, 3),
        "hand_flow": torch.randn(batch_size, num_hand, 3),
        "obj_valid_mask": torch.ones(batch_size, num_obj, dtype=torch.bool),
    }
    repeated_output = head(**repeated_inputs)
    repeated_output_again = head(**repeated_inputs)
    for key in ("cm_tokens", "cm_assignment", "cm_slot_weights", "decoder_slot_usage", "pred_obj_flow"):
        torch.testing.assert_close(repeated_output_again[key], repeated_output[key])


def test_cm_flow_head_restores_metric_anchor_coordinates_after_internal_scaling() -> None:
    torch.manual_seed(11)
    head = CmFlowHead(
        dense_token_dim=4,
        cm_dim=8,
        num_cm_tokens=2,
        num_slot_iters=1,
        internal_point_flow_scale=100.0,
    )
    hand_points = torch.randn(1, 3, 3)
    output = head(
        z_obj=torch.randn(1, 2, 4),
        z_hand=torch.randn(1, 3, 4),
        dense_hand_contact=torch.rand(1, 3),
        obj_points=torch.randn(1, 2, 3),
        obj_normals=torch.randn(1, 2, 3),
        hand_points=hand_points,
        hand_normals=torch.randn(1, 3, 3),
        hand_flow=torch.randn(1, 3, 3),
        obj_valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    expected_anchor_m = torch.einsum("bkh,bhd->bkd", output["cm_slot_weights"], hand_points)
    torch.testing.assert_close(output["cm_anchor_pos"], expected_anchor_m)
    # A nonzero internal-centimetre decoder output must be restored to metres.
    with torch.no_grad():
        head.flow_edge[-1].bias[1:] = torch.tensor([1.0, 2.0, 3.0])
    output = head(
        z_obj=torch.randn(1, 2, 4), z_hand=torch.randn(1, 3, 4),
        dense_hand_contact=torch.rand(1, 3), obj_points=torch.randn(1, 2, 3),
        obj_normals=torch.randn(1, 2, 3), hand_points=hand_points,
        hand_normals=torch.randn(1, 3, 3), hand_flow=torch.randn(1, 3, 3),
        obj_valid_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    expected_m = torch.tensor([0.01, 0.02, 0.03]).expand(1, 2, 3)
    torch.testing.assert_close(output["pred_obj_flow"], expected_m)


def test_dense_token_input_stays_in_metres_and_internal_loss_scales_gradient() -> None:
    class CapturingDense(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_dim = 4
            self.received: dict[str, torch.Tensor] = {}

        def forward(self, **kwargs):
            self.received = kwargs
            batch_size = kwargs["obj_points"].shape[0]
            return (
                torch.zeros(batch_size, 2, 4),
                torch.zeros(batch_size, 3, 4),
                torch.zeros(batch_size, 3),
            )

    class PassthroughHead(nn.Module):
        def forward(self, **kwargs):
            return {"pred_obj_flow": kwargs["obj_points"]}

    model = object.__new__(CmFlowModel)
    nn.Module.__init__(model)
    dense = CapturingDense()
    model.dense_encoder = dense
    model.head = PassthroughHead()
    obj_points_m = torch.tensor([[[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]]])
    batch = {
        "obj_points": obj_points_m,
        "obj_normals": torch.zeros_like(obj_points_m),
        "hand_points": torch.zeros(1, 3, 3),
        "hand_normals": torch.zeros(1, 3, 3),
        "hand_flow": torch.zeros(1, 3, 3),
        "obj_valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }
    model(batch)
    torch.testing.assert_close(dense.received["obj_points"], obj_points_m)

    pred_m = torch.tensor([[[0.01, 0.0, 0.0]]], requires_grad=True)
    gt_m = torch.zeros_like(pred_m)
    valid = torch.ones(1, 1, dtype=torch.bool)
    loss_m = internal_flow_smooth_l1(pred_m, gt_m, valid, beta_m=0.01, internal_scale=1.0)
    loss_cm = internal_flow_smooth_l1(pred_m, gt_m, valid, beta_m=0.01, internal_scale=100.0)
    grad_m = torch.autograd.grad(loss_m, pred_m, retain_graph=True)[0]
    grad_cm = torch.autograd.grad(loss_cm, pred_m)[0]
    torch.testing.assert_close(grad_cm, grad_m * 100.0)

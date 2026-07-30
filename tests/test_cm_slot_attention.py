from __future__ import annotations

import torch

from src.task.Cm.model import CmFlowHead


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
    # The zero-initialized decoder predicts internal zero, which restores to
    # metric zero independently of the internal scale.
    torch.testing.assert_close(output["pred_obj_flow"], torch.zeros_like(output["pred_obj_flow"]))

from __future__ import annotations

import torch

from src.task.Cm.model import CmFlowHead
from src.task.Cm.runner import _wrist_targets


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
        wrist_delta=torch.eye(4).expand(batch_size, -1, -1).clone(),
        obj_valid_mask=torch.tensor([[True, True, False, True, False], [True] * num_obj]),
    )

    assert head.hand_motion_encoder[0].in_features == 8 + 22
    assert output["cm_tokens"].shape == (batch_size, 4, 16)
    assert output["cm_assignment"].shape == (batch_size, 4, num_hand)
    assert output["cm_slot_weights"].shape == (batch_size, 4, num_hand)
    assert output["decoder_slot_usage"].shape == (batch_size, 4)
    assert output["cm_anchor_pos"].shape == (batch_size, 4, 3)
    assert output["cm_anchor_normal"].shape == (batch_size, 4, 3)
    assert output["cm_hand_flow"].shape == (batch_size, 4, 3)
    assert output["pred_obj_flow"].shape == (batch_size, num_obj, 3)
    assert output["pred_hand_flow"].shape == (batch_size, num_hand, 3)
    assert output["pred_rigid_hand_flow"].shape == (batch_size, num_hand, 3)
    assert output["pred_hand_articulation_flow"].shape == (batch_size, num_hand, 3)
    assert output["pred_wrist_rotation"].shape == (batch_size, 3, 3)
    assert output["pred_wrist_translation"].shape == (batch_size, 3)
    assert output["hand_decoder_attention"].shape == (batch_size, num_hand, 4)
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
    torch.testing.assert_close(
        output["pred_hand_flow"], output["pred_rigid_hand_flow"] + output["pred_hand_articulation_flow"]
    )
    torch.testing.assert_close(
        output["hand_decoder_attention"].sum(dim=-1), torch.ones(batch_size, num_hand)
    )
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
        "wrist_delta": torch.eye(4).expand(batch_size, -1, -1).clone(),
        "obj_valid_mask": torch.ones(batch_size, num_obj, dtype=torch.bool),
    }
    repeated_output = head(**repeated_inputs)
    repeated_output_again = head(**repeated_inputs)
    for key in (
        "cm_tokens", "cm_assignment", "cm_slot_weights", "decoder_slot_usage", "pred_obj_flow",
        "pred_hand_flow", "pred_wrist_rotation", "pred_wrist_translation",
    ):
        torch.testing.assert_close(repeated_output_again[key], repeated_output[key])


def test_wrist_target_split_uses_inverse_stage4_transform() -> None:
    hand_points = torch.tensor([[[1.0, 0.0, 0.0]]])
    # Stage 4 stores T_current<-next.  Its inverse maps (1, 0, 0) to (0, 0, 0).
    wrist_delta = torch.eye(4).reshape(1, 4, 4)
    wrist_delta[:, 0, 3] = 1.0
    total_flow = torch.tensor([[[-1.0, 0.0, 0.0]]])

    rotation, translation, rigid_flow, articulation_flow = _wrist_targets(hand_points, total_flow, wrist_delta)

    torch.testing.assert_close(rotation, torch.eye(3).reshape(1, 3, 3))
    torch.testing.assert_close(translation, torch.tensor([[-1.0, 0.0, 0.0]]))
    torch.testing.assert_close(rigid_flow, total_flow)
    torch.testing.assert_close(articulation_flow, torch.zeros_like(total_flow))

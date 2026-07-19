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
        wrist_delta=torch.eye(4).expand(batch_size, -1, -1).clone(),
        obj_valid_mask=torch.tensor([[True, True, False, True, False], [True] * num_obj]),
    )

    assert head.hand_motion_encoder[0].in_features == 8 + 22
    assert output["cm_tokens"].shape == (batch_size, 4, 16)
    assert output["cm_assignment"].shape == (batch_size, 4, num_hand)
    assert output["cm_anchor_pos"].shape == (batch_size, 4, 3)
    assert output["cm_anchor_normal"].shape == (batch_size, 4, 3)
    assert output["cm_hand_flow"].shape == (batch_size, 4, 3)
    assert output["pred_obj_flow"].shape == (batch_size, num_obj, 3)
    torch.testing.assert_close(output["cm_assignment"].sum(dim=-1), torch.ones(batch_size, 4))
    torch.testing.assert_close(
        torch.linalg.vector_norm(output["cm_anchor_normal"], dim=-1),
        torch.ones(batch_size, 4),
    )
    torch.testing.assert_close(output["pred_obj_flow"][0, ~torch.tensor([True, True, False, True, False])], torch.zeros(2, 3))
    assert torch.isfinite(output["cm_tokens"]).all()

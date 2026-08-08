from types import SimpleNamespace

import torch
from torch import nn

from src.task.InteractionDynamics.model import InteractionDynamicsModel


class FakeDense(nn.Module):
    token_dim = 16

    def forward(self, *, obj_points, hand_points, **kwargs):
        del kwargs
        # Parameter-free stand-in preserving the frozen encoder contract.
        z_obj = torch.cat([obj_points, obj_points, obj_points, obj_points, obj_points, obj_points[..., :1]], -1)
        z_hand = torch.cat([hand_points, hand_points, hand_points, hand_points, hand_points, hand_points[..., :1]], -1)
        return z_obj, z_hand, hand_points[..., 0].sigmoid()


def test_forward_backward_without_future_object_input():
    meta = SimpleNamespace(model_dim=48, attention_heads=6, num_hand_patches=4,
        num_obj_patches=4, patch_size=8, chunk_len=8, action_temporal_layers=2,
        action_world_layers=2, motion_scale=100., dense_checkpoint="unused", uni3d_checkpoint="missing")
    model = InteractionDynamicsModel(SimpleNamespace(meta=meta), dense_encoder=FakeDense(),
                                     load_uni3d=False, world_depth=1)
    batch, hand_count, object_count, effect_count = 2, 32, 48, 12
    hand_object = torch.randn(batch, hand_count, 3)
    hand_local = torch.randn(batch, hand_count, 3)
    obj = torch.randn(batch, object_count, 3)
    effect = torch.randn(batch, effect_count, 3)
    normals = lambda *shape: torch.nn.functional.normalize(torch.randn(*shape, 3), dim=-1)
    output = model(
        world_hand_points_object=hand_object, world_hand_normals_object=normals(batch, hand_count),
        world_obj_points_object=obj, world_obj_normals_object=normals(batch, object_count),
        action_hand_points_hand=hand_local, action_hand_normals_hand=normals(batch, hand_count),
        hand_disp_chunk=torch.randn(batch, 8, hand_count, 3) * .01,
        dense_obj_points_hand=effect, dense_obj_normals_hand=normals(batch, effect_count),
        dense_hand_points_hand=hand_local, dense_hand_normals_hand=normals(batch, hand_count),
        effect_obj_points_object=effect, effect_obj_normals_object=normals(batch, effect_count),
        effect_obj_valid_mask=torch.ones(batch, effect_count, dtype=torch.bool))
    assert output["world_tokens"].shape == (batch, 8, 48)
    assert output["action_tokens"].shape == (batch, 4, 48)
    assert output["interaction_tokens"].shape == (batch, 4, 48)
    assert output["pred_obj_disp_chunk"].shape == (batch, 8, effect_count, 3)
    output["pred_obj_disp_internal"].square().mean().backward()
    assert model.action.spatial[0].weight.grad is not None
    assert model.canonicalizer.attn.in_proj_weight.grad is not None
    assert model.effect.flow[0].weight.grad is not None
    assert model.world.patch_encoder.local[0].weight.grad is not None
    assert not any(parameter.requires_grad for parameter in model.dense_encoder.parameters())
    assert not any(key.startswith("dense_encoder.") for key in model.state_dict())

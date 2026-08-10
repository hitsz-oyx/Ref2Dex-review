from types import SimpleNamespace

import torch
from torch import nn

from src.task.InteractionDynamics.model import (
    DenseEdgeInteractionModel, EffectDecoder, InteractionDynamicsModel, SE3DynamicsHead,
    PosePairActionEncoder, SpatiotemporalInteractionField, axis_angle_to_matrix,
)


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
        action_world_layers=2, motion_scale=100., dense_checkpoint="unused", uni3d_checkpoint="missing",
        interaction_reconstruction=True, action_decomposition_reconstruction=True)
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
        action_hand_disp_chunk_object=torch.randn(batch, 8, hand_count, 3) * .01,
        dense_obj_points_hand=effect, dense_obj_normals_hand=normals(batch, effect_count),
        dense_hand_points_hand=hand_local, dense_hand_normals_hand=normals(batch, hand_count),
        effect_obj_points_object=effect, effect_obj_normals_object=normals(batch, effect_count),
        effect_obj_valid_mask=torch.ones(batch, effect_count, dtype=torch.bool))
    assert output["world_tokens"].shape == (batch, 8, 48)
    assert output["action_tokens"].shape == (batch, 4, 48)
    assert output["interaction_tokens"].shape == (batch, 4, 48)
    assert output["pred_obj_disp_chunk"].shape == (batch, 8, effect_count, 3)
    assert output["pred_hand_patch_disp_internal"].shape == (batch, 8, 4, 3)
    assert output["pred_obj_patch_disp_internal"].shape == (batch, 8, 4, 3)
    assert output["relative_interaction_tokens"].shape == (batch, 4, 48)
    assert output["pred_relative_motion_internal"].shape == (batch, 8, 4, 4)
    assert output["action_tokens_temporal"].shape == (batch, 8, 4, 48)
    assert output["interaction_field"].shape == (batch, 8, 4, 128)
    assert output["interaction_field_descriptor"].shape == (batch, 8, 4, 5)
    assert output["se3_interaction_field"].shape == (batch, 8, 4, 128)
    assert output["global_interaction_code"].shape == (batch, 8, 128)
    assert output["pred_object_centric_action_field"].shape == (batch, 8, 4, 6)
    assert output["pred_hand_articulation_increment_internal"].shape == (batch, 8, 4, 3)
    assert output["pred_hand_root_increment_translation_internal"].shape == (batch, 8, 3)
    assert output["pred_hand_root_increment_rotation_matrix"].shape == (batch, 8, 3, 3)
    assert output["pred_obj_increment_rotation_matrix"].shape == (batch, 8, 3, 3)
    assert output["effect_patch_index"].shape == (batch, effect_count)
    output["pred_obj_disp_internal"].square().mean().backward()
    assert model.action.spatial[0].weight.grad is not None
    assert model.canonicalizer.attn.in_proj_weight.grad is not None
    assert model.effect.flow[0].weight.grad is not None
    assert model.world.patch_encoder.local[0].weight.grad is not None
    assert not any(parameter.requires_grad for parameter in model.dense_encoder.parameters())
    assert not any(key.startswith("dense_encoder.") for key in model.state_dict())


def test_global_action_ablation_removes_object_indexed_field_and_descriptor():
    meta = SimpleNamespace(model_dim=48, attention_heads=6, num_hand_patches=4,
        num_obj_patches=4, patch_size=8, chunk_len=8, action_temporal_layers=2,
        action_world_layers=2, motion_scale=100., dense_checkpoint="unused",
        uni3d_checkpoint="missing", use_v7_field=True, field_ablation="global_action_only")
    model = InteractionDynamicsModel(SimpleNamespace(meta=meta), dense_encoder=FakeDense(),
                                     load_uni3d=False, world_depth=1)
    batch, hand_count, object_count, effect_count = 2, 32, 48, 12
    normals = lambda *shape: torch.nn.functional.normalize(torch.randn(*shape, 3), dim=-1)
    output = model(
        world_hand_points_object=torch.randn(batch, hand_count, 3),
        world_hand_normals_object=normals(batch, hand_count),
        world_obj_points_object=torch.randn(batch, object_count, 3),
        world_obj_normals_object=normals(batch, object_count),
        action_hand_points_hand=torch.randn(batch, hand_count, 3),
        action_hand_normals_hand=normals(batch, hand_count),
        hand_disp_chunk=torch.randn(batch, 8, hand_count, 3) * .01,
        action_hand_disp_chunk_object=torch.randn(batch, 8, hand_count, 3) * .01,
        dense_obj_points_hand=torch.randn(batch, effect_count, 3),
        dense_obj_normals_hand=normals(batch, effect_count),
        dense_hand_points_hand=torch.randn(batch, hand_count, 3),
        dense_hand_normals_hand=normals(batch, hand_count),
        effect_obj_points_object=torch.randn(batch, effect_count, 3),
        effect_obj_normals_object=normals(batch, effect_count),
        effect_obj_valid_mask=torch.ones(batch, effect_count, dtype=torch.bool))
    field = output["se3_interaction_field"]
    assert torch.allclose(field, field[:, :, :1].expand_as(field))
    assert torch.count_nonzero(output["se3_interaction_field_descriptor"]) == 0


def test_pose_pair_encoder_builds_static_and_incremental_tokens():
    encoder = PosePairActionEncoder(dim=24, heads=4, chunk_len=8, temporal_layers=1,
                                    patch_size=4)
    points = torch.randn(2, 9, 16, 3)
    root = torch.eye(4).reshape(1, 1, 4, 4).repeat(2, 8, 1, 1)
    knn = torch.arange(16).reshape(1, 4, 4).repeat(2, 1, 1)
    output = encoder(points, root, torch.randn(2, 16, 3), knn, 100.)
    assert output["pose_tokens"].shape == (2, 9, 4, 24)
    assert output["global_pose_code"].shape == (2, 9, 24)
    assert output["action_tokens_temporal"].shape == (2, 8, 4, 24)
    assert output["pred_hand_pose_patch_center_internal"].shape == (2, 9, 4, 3)


def test_effect_decoder_routes_patch_motion_by_nearest_center():
    decoder = EffectDecoder(dim=12, heads=3, chunk_len=2, dense_dim=4)
    torch.nn.init.zeros_(decoder.flow[0].weight)
    torch.nn.init.zeros_(decoder.flow[0].bias)
    torch.nn.init.zeros_(decoder.flow[2].weight)
    torch.nn.init.zeros_(decoder.flow[2].bias)
    points = torch.tensor([[[-.9, 0., 0.], [.8, 0., 0.]]])
    centers = torch.tensor([[[-1., 0., 0.], [1., 0., 0.]]])
    patch_motion = torch.tensor([[[[1., 0., 0.], [2., 0., 0.]],
                                  [[3., 0., 0.], [4., 0., 0.]]]])
    output = decoder(points, torch.zeros_like(points), torch.empty(1, 2, 4),
                     torch.zeros(1, 2, 12), centers, patch_motion,
                     torch.ones(1, 2, dtype=torch.bool), 100.)
    assert torch.equal(output["effect_patch_index"], torch.tensor([[0, 1]]))
    assert torch.allclose(output["pred_obj_disp_internal"][0, :, :, 0],
                          torch.tensor([[1., 2.], [3., 4.]]))


def test_axis_angle_and_se3_translation_are_analytic():
    rotation = axis_angle_to_matrix(torch.zeros(1, 2, 3))
    assert torch.allclose(rotation, torch.eye(3).expand(1, 2, 3, 3))
    head = SE3DynamicsHead(model_dim=6, field_dim=4, motion_scale=100.)
    with torch.no_grad():
        head.head[-1].bias[0] = 1.0
    output = head(torch.zeros(1, 2, 3, 4), torch.zeros(1, 2, 3, 5),
                  torch.zeros(1, 3, 6), torch.zeros(1, 1, 3))
    assert torch.allclose(output["pred_obj_disp_chunk_se3"][0, :, 0, 0],
                          torch.tensor([.01, .02]))


def test_soft_field_is_normalized_and_differentiable_to_hand_trajectory():
    module = SpatiotemporalInteractionField(model_dim=6, field_dim=4, sigma_m=.05)
    action = torch.randn(1, 2, 3, 6)
    displacement = torch.randn(1, 2, 3, 3, requires_grad=True) * .01
    output = module(action, torch.randn(1, 2, 6), torch.randn(1, 3, 3) * .01,
                    torch.randn(1, 2, 3) * .01, torch.randn(1, 3, 3),
                    torch.randn(1, 2, 3), displacement)
    weights = output["hand_to_object_soft_weights"]
    assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6)
    gradient = torch.autograd.grad(output["interaction_field_descriptor"].square().sum(), displacement)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def test_dense_edge_model_preserves_all_hand_points_and_local_neighbors():
    meta = SimpleNamespace(motion_scale=100., dense_edge_knn=2,
                           dense_edge_dim=8, dense_edge_sigma_m=.02)
    model = DenseEdgeInteractionModel(SimpleNamespace(meta=meta))
    hand = torch.tensor([[[0., 0., 0.], [.1, 0., 0.], [.2, 0., 0.]]])
    obj = torch.tensor([[[.01, 0., 0.], [.11, 0., 0.], [.21, 0., 0.], [.3, 0., 0.]]])
    normals_hand = torch.tensor([[[1., 0., 0.]]]).expand_as(hand)
    normals_obj = torch.tensor([[[1., 0., 0.]]]).expand_as(obj)
    output = model(world_hand_points_object=hand, world_hand_normals_object=normals_hand,
                   world_obj_points_object=obj, world_obj_normals_object=normals_obj,
                   action_hand_disp_chunk_object=torch.zeros(1, 2, 3, 3))
    assert output["dense_interaction_tokens"].shape == (1, 2, 3, 8)
    assert output["pred_dense_relative_motion_internal"].shape == (1, 2, 3, 4)
    assert torch.equal(output["dense_nearest_obj_point"], torch.tensor([[0, 1, 2]]))

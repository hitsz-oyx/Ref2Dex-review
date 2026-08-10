import torch

from src.task.InteractionDynamics.runner import (
    action_decomposition_losses, dense_relative_motion_target, field_statistics,
    masked_effect_mse, masked_relative_mse,
    patch_motion_target, relative_motion_target,
    trajectory_statistics,
)


def test_action_decomposition_separates_root_and_articulation():
    pose = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 2, 1, 1)
    pose[..., :3, 3] = torch.tensor([[[.01, 0., 0.], [.02, 0., 0.]]])
    batch = {
        "hand_articulation_increment_gt": torch.tensor([
            [[[.001, 0., 0.], [.003, 0., 0.]],
             [[.002, 0., 0.], [.004, 0., 0.]]]]),
        "hand_root_increment_pose_gt": pose,
    }
    prediction = {
        "hand_knn_idx": torch.tensor([[[0, 1]]]),
        "pred_hand_articulation_increment_internal": torch.tensor([[[[.2, 0., 0.]], [[.3, 0., 0.]]]]),
        "pred_hand_root_increment_translation_internal": torch.tensor([[[1., 0., 0.], [2., 0., 0.]]]),
        "pred_hand_root_increment_rotation_matrix": torch.eye(3).reshape(1, 1, 3, 3).repeat(1, 2, 1, 1),
    }
    articulation, translation, rotation, metrics = action_decomposition_losses(
        batch, prediction, 100.)
    assert torch.allclose(articulation, torch.tensor(0.))
    assert torch.allclose(translation, torch.tensor(0.))
    assert torch.allclose(rotation, torch.tensor(0.))
    assert metrics["action/zero_articulation_rmse_cm"] > 0


def test_internal_centimeter_mse_and_metrics():
    gt = torch.zeros(1, 8, 3, 3)
    pred = gt.clone()
    pred[:, :, 0, 0] = .001
    valid = torch.tensor([[True, False, False]])
    # One non-zero xyz component at every valid timestep: (0.1 cm)^2.
    assert torch.allclose(masked_effect_mse(pred, gt, valid), torch.tensor(0.01 / 3), atol=1e-7)
    stats = trajectory_statistics(pred, gt, valid)
    assert torch.allclose(stats["object/ade_mm"], torch.tensor(1.0))
    assert torch.allclose(stats["object/fde_mm"], torch.tensor(1.0))


def test_patch_motion_target_uses_stable_patch_indices_and_centimeters():
    displacement = torch.zeros(1, 2, 4, 3)
    displacement[:, :, 0, 0] = .01
    displacement[:, :, 1, 0] = .03
    displacement[:, :, 2, 1] = .02
    displacement[:, :, 3, 1] = .04
    knn = torch.tensor([[[0, 1], [2, 3]]])
    target = patch_motion_target(displacement, knn)
    assert target.shape == (1, 2, 2, 3)
    assert torch.allclose(target[0, :, 0, 0], torch.tensor([2., 2.]))
    assert torch.allclose(target[0, :, 1, 1], torch.tensor([3., 3.]))


def test_relative_motion_target_removes_shared_motion_and_splits_normal_tangent():
    batch = {
        "hand_disp_chunk_object_gt": torch.tensor([[[[.02, .03, 0.], [.02, .03, 0.]]]]),
        "obj_disp_chunk_gt": torch.tensor([[[[.01, .01, 0.], [.01, .01, 0.]]]]),
        "world_obj_normals_object": torch.tensor([[[1., 0., 0.], [1., 0., 0.]]]),
    }
    prediction = {
        "hand_knn_idx": torch.tensor([[[0, 1]]]),
        "obj_knn_idx": torch.tensor([[[0, 1]]]),
        "relative_nearest_obj_patch": torch.tensor([[0]]),
    }
    target = relative_motion_target(batch, prediction)
    assert target.shape == (1, 1, 1, 4)
    assert torch.allclose(target[0, 0, 0], torch.tensor([1., 0., 2., 0.]))


def test_increment_relative_target_differences_cumulative_motion():
    batch = {
        "hand_disp_chunk_object_gt": torch.tensor([
            [[[.02, .03, 0.], [.02, .03, 0.]], [[.05, .07, 0.], [.05, .07, 0.]]]]),
        "obj_disp_chunk_gt": torch.tensor([
            [[[.01, .01, 0.], [.01, .01, 0.]], [[.02, .02, 0.], [.02, .02, 0.]]]]),
        "world_obj_normals_object": torch.tensor([[[1., 0., 0.], [1., 0., 0.]]]),
        "obj_normals_chunk_object_gt": torch.tensor([
            [[[1., 0., 0.], [1., 0., 0.]], [[1., 0., 0.], [1., 0., 0.]]]]),
    }
    prediction = {
        "hand_knn_idx": torch.tensor([[[0, 1]]]), "obj_knn_idx": torch.tensor([[[0, 1]]]),
        "relative_nearest_obj_patch": torch.tensor([[0]]),
    }
    target = relative_motion_target(batch, prediction, mode="increment_fixed")
    assert torch.allclose(target[0, :, 0], torch.tensor([[1., 0., 2., 0.], [2., 0., 3., 0.]]))


def test_masked_relative_mse_ignores_far_edges():
    target = torch.tensor([[[[1., 0., 0., 0.], [100., 0., 0., 0.]]]])
    distance = torch.tensor([[.01, .10]])
    assert torch.allclose(masked_relative_mse(torch.zeros_like(target), target, distance, 5.),
                          torch.tensor(.25))


def test_field_statistics_measure_v7_spatial_temporal_structure():
    field = torch.zeros(1, 2, 2, 1)
    field[0, 0, 1, 0] = 2.
    field[0, 1, :, 0] = torch.tensor([2., 4.])
    descriptor = torch.zeros(1, 2, 2, 5)
    descriptor[..., 0] = .25
    weights = torch.tensor([[[[.5, .5], [.5, .5]], [[.5, .5], [.5, .5]]]])
    stats = field_statistics(field, descriptor, weights)
    assert stats["field/spatial_feature_variance"] > 0
    assert stats["field/temporal_feature_variance"] > 0
    assert torch.allclose(stats["field/proximity_density_mean"], torch.tensor(.25))
    assert torch.allclose(stats["field/hand_weight_entropy"], torch.log(torch.tensor(2.)))


def test_dense_relative_target_uses_point_level_fixed_edge_increment():
    batch = {
        "hand_disp_chunk_object_gt": torch.tensor([[[[.02, .03, 0.]], [[.05, .07, 0.]]]]),
        "obj_disp_chunk_gt": torch.tensor([
            [[[.01, .01, 0.], [0., 0., 0.]], [[.02, .02, 0.], [0., 0., 0.]]]]),
        "obj_normals_chunk_object_gt": torch.tensor([
            [[[1., 0., 0.], [1., 0., 0.]], [[1., 0., 0.], [1., 0., 0.]]]]),
    }
    prediction = {"dense_nearest_obj_point": torch.tensor([[0]])}
    target, hand_only = dense_relative_motion_target(batch, prediction)
    assert torch.allclose(target[0, :, 0], torch.tensor([[1., 0., 2., 0.], [2., 0., 3., 0.]]))
    assert torch.allclose(hand_only[0, :, 0], torch.tensor([[2., 0., 3., 0.], [3., 0., 4., 0.]]))

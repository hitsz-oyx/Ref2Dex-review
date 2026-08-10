import torch

from src.task.InteractionDynamics.runner import (
    masked_effect_mse, masked_relative_mse, patch_motion_target, relative_motion_target,
    trajectory_statistics,
)


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


def test_masked_relative_mse_ignores_far_edges():
    target = torch.tensor([[[[1., 0., 0., 0.], [100., 0., 0., 0.]]]])
    distance = torch.tensor([[.01, .10]])
    assert torch.allclose(masked_relative_mse(torch.zeros_like(target), target, distance, 5.),
                          torch.tensor(.25))

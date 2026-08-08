import torch

from src.task.InteractionDynamics.runner import masked_effect_mse, trajectory_statistics


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

import torch

from src.task.InteractionDynamics.action_probe import (
    ProbeData, decompose_patch_motion, evaluate_probe, fit_ridge, probe_shapes,
)


def test_patch_motion_decomposition_is_exact_and_local_zero_mean():
    target = torch.randn(2, 3, 4, 3)
    global_motion, local = decompose_patch_motion(target)
    assert torch.allclose(target, global_motion[:, :, None] + local)
    assert torch.allclose(local.mean(2), torch.zeros_like(global_motion), atol=1e-6)


def test_ridge_probe_recovers_linear_local_signal():
    tokens = torch.randn(3, 4, 5)
    linear = torch.randn(5, 6)
    local_flat = tokens @ linear
    local = local_flat.reshape(3, 4, 2, 3).permute(0, 2, 1, 3)
    data = ProbeData(tokens=tokens, full=local, global_motion=torch.zeros(3, 2, 3), local=local)
    x, y = probe_shapes(data, "local")
    weight = fit_ridge(x, y, 1e-5)
    metrics = evaluate_probe(data, "local", weight)
    assert metrics["rmse_cm"] < 1e-3
    assert metrics["relative_improvement"] > .99

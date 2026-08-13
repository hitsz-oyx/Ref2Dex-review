import torch

from src.task.InteractionDynamics.eval_parallel_fusion_v20_4 import (
    complementarity, fuse_channels, interpolate_velocity,
)
from src.task.InteractionDynamics.fusion_gate_v20_4 import (
    VelocityFusionGate, fuse_velocity, fusion_features,
)


def test_channel_fusion_keeps_free_p() -> None:
    free = torch.zeros(2, 4, 3, 8); hand = torch.ones(2, 4, 3, 7)
    free[..., 7] = 9
    mixed = fuse_channels(free, hand, "YHH")
    torch.testing.assert_close(mixed[..., :3], free[..., :3])
    torch.testing.assert_close(mixed[..., 3:7], hand[..., 3:7])
    torch.testing.assert_close(mixed[..., 7], free[..., 7])


def test_velocity_interpolation_changes_only_v() -> None:
    free = torch.zeros(1, 2, 3, 8); hand = torch.ones(1, 2, 3, 7)
    mixed = interpolate_velocity(free, hand, .25)
    torch.testing.assert_close(mixed[..., :4], free[..., :4])
    torch.testing.assert_close(mixed[..., 4:7], torch.full_like(mixed[..., 4:7], .25))
    torch.testing.assert_close(mixed[..., 7], free[..., 7])


def test_complementarity_reports_per_sample_oracle() -> None:
    target = torch.zeros(2, 1, 1, 7)
    free = torch.zeros_like(target); hand = torch.zeros_like(target)
    free[0, ..., :3] = 1; hand[1, ..., :3] = 1
    result = complementarity(free, hand, target)["r"]
    assert result["h_win_rate"] == .5
    assert result["oracle_sample_rmse_mean_cm"] == 0
    assert result["oracle_improvement_vs_best_endpoint"] == 1


def test_gate_features_and_endpoints() -> None:
    free = torch.randn(2, 4, 3, 8); hand = torch.randn(2, 4, 3, 7)
    features = fusion_features(free, hand)
    assert features.shape == (2, 4, 3, 8)
    gate = VelocityFusionGate(hidden=16)(features)
    assert gate.shape == (2, 4, 3, 1)
    torch.testing.assert_close(fuse_velocity(free, hand, torch.zeros_like(gate)), free)
    force_h = fuse_velocity(free, hand, torch.ones_like(gate))
    torch.testing.assert_close(force_h[..., :4], free[..., :4])
    torch.testing.assert_close(force_h[..., 4:7], hand[..., 4:7])
    torch.testing.assert_close(force_h[..., 7], free[..., 7])

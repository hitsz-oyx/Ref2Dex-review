import torch

from src.task.InteractionDynamics.state_interaction_diffusion import (
    StateInteractionDiffusion, pack_state_future, sample_state_ddim)


def test_pack_state_future_order():
    y = {"relative_geometry": torch.randn(5, 3, 3),
         "relative_distance": torch.randn(5, 3),
         "relative_motion": torch.randn(4, 3, 3)}
    state, future = pack_state_future(y, 4)
    assert state.shape == (3, 4) and future.shape == (3, 28)
    assert torch.equal(future[:, :3], y["relative_motion"][0])
    assert torch.equal(future[:, 3:6], y["relative_geometry"][1])
    assert torch.equal(future[:, 6], y["relative_distance"][1])


def test_state_diffusion_is_anchor_permutation_equivariant():
    torch.manual_seed(3)
    model = StateInteractionDiffusion(horizon=2, dim=32, heads=4, layers=2).eval()
    noisy, state = torch.randn(2, 7, 14), torch.randn(2, 7, 4)
    anchors, patches = torch.randn(2, 7, 3), torch.randn(2, 7, 5, 6)
    effect, timestep = torch.randn(2, 6), torch.tensor([2, 7])
    order = torch.tensor([3, 0, 6, 2, 5, 1, 4])
    expected = model(noisy, state, anchors, patches, effect, timestep)[:, order]
    actual = model(noisy[:, order], state[:, order], anchors[:, order], patches[:, order],
                   effect, timestep)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_state_ddim_is_finite():
    model = StateInteractionDiffusion(horizon=2, dim=32, heads=4, layers=1).eval()
    state, anchors = torch.randn(1, 6, 4), torch.randn(1, 6, 3)
    patches, effect = torch.randn(1, 6, 4, 6), torch.randn(1, 6)
    result = sample_state_ddim(model, state, anchors, patches, effect, 4,
                               initial_noise=torch.randn(1, 6, 14))
    assert result.shape == (1, 6, 14)
    assert torch.isfinite(result).all() and result.abs().max() <= 5

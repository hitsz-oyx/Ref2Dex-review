import torch

from src.task.InteractionDynamics.interaction_diffusion import (
    InteractionFieldDiffusion, pack_anchor_y, sample_ddpm, unpack_anchor_y)


def test_anchor_y_pack_roundtrip():
    y = {"relative_geometry": torch.randn(2, 9, 7, 3),
         "relative_distance": torch.randn(2, 9, 7),
         "relative_motion": torch.randn(2, 8, 7, 3)}
    restored = unpack_anchor_y(pack_anchor_y(y))
    for name in y:
        torch.testing.assert_close(restored[name], y[name])


def test_diffusion_is_anchor_permutation_equivariant():
    torch.manual_seed(3)
    model = InteractionFieldDiffusion(dim=32, heads=4, layers=2).eval()
    noisy = torch.randn(2, 7, 60)
    anchors = torch.randn(2, 7, 3)
    normals = torch.randn(2, 7, 3)
    effect = torch.randn(2, 8, 6)
    timestep = torch.tensor([2, 5])
    permutation = torch.randperm(7)
    original = model(noisy, anchors, normals, effect, timestep)
    permuted = model(noisy[:, permutation], anchors[:, permutation],
                     normals[:, permutation], effect, timestep)
    torch.testing.assert_close(permuted, original[:, permutation], atol=1e-5, rtol=1e-5)


def test_ddim_sampling_stays_finite():
    class ZeroModel(torch.nn.Module):
        def forward(self, noisy_y, anchors_cm, anchor_normals, effect, timestep):
            return torch.zeros_like(noisy_y)

    anchors = torch.randn(2, 7, 3)
    result = sample_ddpm(
        ZeroModel(), anchors, torch.randn_like(anchors), torch.randn(2, 8, 6),
        steps=10, generator=torch.Generator().manual_seed(5))
    assert result.shape == (2, 7, 60)
    assert torch.isfinite(result).all()
    assert result.abs().max() <= 5

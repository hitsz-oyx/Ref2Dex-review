import torch

from src.task.InteractionDynamics.residual_interaction_diffusion import (
    ResidualInteractionDiffusion, make_v_target, recover_x0_noise, sample_residual_v)


def test_v_transform_roundtrip():
    torch.manual_seed(0)
    x0, noise = torch.randn(3, 4, 5), torch.randn(3, 4, 5)
    alpha_bar = torch.rand(3, 1, 1)
    xt, velocity = make_v_target(x0, noise, alpha_bar)
    recovered_x0, recovered_noise = recover_x0_noise(xt, velocity, alpha_bar)
    assert (recovered_x0 - x0).abs().max() < 1e-6
    assert (recovered_noise - noise).abs().max() < 1e-6


def test_shape_and_permutation_equivariance():
    torch.manual_seed(0)
    model = ResidualInteractionDiffusion(4, 4, 32, 4, 2).eval()
    noisy, state = torch.randn(2, 128, 28), torch.randn(2, 128, 4)
    anchors, patches = torch.randn(2, 128, 3), torch.randn(2, 128, 32, 6)
    goal, timestep = torch.randn(2, 25), torch.tensor([2, 7])
    permutation = torch.randperm(128)
    prediction = model(noisy, state, anchors, patches, goal, timestep)
    permuted = model(noisy[:, permutation], state[:, permutation], anchors[:, permutation],
                     patches[:, permutation], goal, timestep)
    assert prediction.shape == (2, 128, 28)
    assert torch.allclose(permuted, prediction[:, permutation], atol=1e-5)


def test_oracle_velocity_sampler_recovers_clean():
    class Oracle(torch.nn.Module):
        residual_dim = 28
        def __init__(self, clean, steps):
            super().__init__()
            self.clean, self.steps = clean, steps
        def forward(self, value, state, anchors, patches, goal, timestep):
            from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule
            _, schedule = cosine_schedule(self.steps, value.device)
            alpha = schedule[timestep][:, None, None]
            noise = (value - alpha.sqrt() * self.clean) / (1 - alpha).sqrt()
            return alpha.sqrt() * noise - (1 - alpha).sqrt() * self.clean
    clean = torch.randn(1, 128, 28)
    model = Oracle(clean, 10)
    empty = torch.zeros(1, 128, 4)
    result, trace = sample_residual_v(model, empty, torch.zeros(1, 128, 3),
                                      torch.zeros(1, 128, 32, 6), torch.zeros(1, 25), 10,
                                      initial_noise=torch.randn_like(clean), trace_every=1)
    assert torch.allclose(result, clean, atol=2e-5)
    assert all(torch.isfinite(torch.tensor(list(row.values()))).all() for row in trace)


def test_oracle_velocity_sampler_recovers_clean_with_skipping():
    torch.manual_seed(3)
    clean = torch.randn(1, 128, 28)
    class Oracle(torch.nn.Module):
        residual_dim = 28
        def forward(self, value, state, anchors, patches, goal, timestep):
            from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule
            _, schedule = cosine_schedule(100, value.device)
            alpha = schedule[timestep][:, None, None]
            noise = (value - alpha.sqrt() * clean) / (1 - alpha).sqrt()
            return alpha.sqrt() * noise - (1 - alpha).sqrt() * clean
    result, _ = sample_residual_v(
        Oracle(), torch.zeros(1, 128, 4), torch.zeros(1, 128, 3),
        torch.zeros(1, 128, 32, 6), torch.zeros(1, 25), 100,
        initial_noise=torch.randn_like(clean), sampling_steps=10)
    assert torch.allclose(result, clean, atol=2e-5)

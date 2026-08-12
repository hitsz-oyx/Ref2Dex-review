"""V16.8 conditional residual diffusion with v-prediction。"""
from __future__ import annotations

import torch
from torch import nn

from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule, sinusoidal_embedding
from src.task.InteractionDynamics.state_interaction_diffusion import AdaLNBlock


def make_v_target(x0: torch.Tensor, noise: torch.Tensor,
                  alpha_bar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """由 clean/noise 构造 x_t 与 v target。"""
    alpha = alpha_bar.sqrt()
    sigma = (1 - alpha_bar).sqrt()
    return alpha * x0 + sigma * noise, alpha * noise - sigma * x0


def recover_x0_noise(xt: torch.Tensor, velocity: torch.Tensor,
                     alpha_bar: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """从 x_t 与 v 恢复 x0/noise，不除以 sqrt(alpha_bar)。"""
    alpha = alpha_bar.sqrt()
    sigma = (1 - alpha_bar).sqrt()
    return alpha * xt - sigma * velocity, sigma * xt + alpha * velocity


class ResidualInteractionDiffusion(nn.Module):
    def __init__(self, horizon: int = 4, goal_segment: int = 4, dim: int = 256,
                 heads: int = 8, layers: int = 6) -> None:
        super().__init__()
        self.horizon, self.goal_segment = horizon, goal_segment
        self.residual_dim, self.dim = 7 * horizon, dim
        self.noisy_residual = nn.Sequential(
            nn.Linear(self.residual_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.object_point = nn.Sequential(
            nn.Linear(6, 128), nn.GELU(), nn.Linear(128, dim), nn.GELU())
        self.state = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.goal = nn.Sequential(
            nn.Linear(1 + 6 * goal_segment, 128), nn.GELU(), nn.Linear(128, dim))
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.condition = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(layers)])
        self.output_norm, self.output = nn.LayerNorm(dim), nn.Linear(dim, self.residual_dim)

    def forward(self, noisy_residual: torch.Tensor, current_state: torch.Tensor,
                anchors_cm: torch.Tensor, object_patches: torch.Tensor,
                goal: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        tokens = (self.noisy_residual(noisy_residual) + self.state(current_state)
                  + self.anchor(anchors_cm) + self.object_point(object_patches).amax(2))
        condition = self.condition(torch.cat([
            self.goal(goal), self.time(sinusoidal_embedding(timestep, self.dim))], -1))
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.output(self.output_norm(tokens))


@torch.no_grad()
def sample_residual_v(model: ResidualInteractionDiffusion, state: torch.Tensor,
                      anchors_cm: torch.Tensor, patches: torch.Tensor, goal: torch.Tensor,
                      steps: int, initial_noise: torch.Tensor | None = None,
                      generator: torch.Generator | None = None,
                      trace_every: int = 0, sampling_steps: int | None = None
                      ) -> tuple[torch.Tensor, list[dict]]:
    """无 clamp、eta=0 的 deterministic DDIM-like v sampler。"""
    _, alpha_bar = cosine_schedule(steps, anchors_cm.device)
    value = (torch.randn((*anchors_cm.shape[:2], model.residual_dim), device=anchors_cm.device,
                         generator=generator) if initial_noise is None else initial_noise.clone())
    trace = []
    clean = value
    sampling_steps = steps if sampling_steps is None else sampling_steps
    if not 1 <= sampling_steps <= steps:
        raise ValueError("sampling_steps must be in [1, diffusion_steps]")
    indices = torch.linspace(steps - 1, 0, sampling_steps, device=anchors_cm.device).round().long()
    indices = torch.unique_consecutive(indices).tolist()
    for position, index in enumerate(indices):
        timestep = torch.full((len(value),), index, device=value.device, dtype=torch.long)
        velocity = model(value, state, anchors_cm, patches, goal, timestep)
        scale = alpha_bar[index].reshape(1, 1, 1)
        clean, noise = recover_x0_noise(value, velocity, scale)
        if not torch.isfinite(clean).all():
            raise FloatingPointError(f"non-finite x0 prediction at timestep {index}")
        if trace_every and (index % trace_every == 0 or position == 0 or position == len(indices) - 1):
            trace.append({"timestep": index, "xt_rms": float(value.square().mean().sqrt()),
                          "xt_abs_max": float(value.abs().max()),
                          "x0_rms": float(clean.square().mean().sqrt()),
                          "x0_abs_max": float(clean.abs().max())})
        if position + 1 < len(indices):
            next_index = indices[position + 1]
            value = (alpha_bar[next_index].sqrt() * clean
                     + (1 - alpha_bar[next_index]).sqrt() * noise)
        else:
            value = clean
    return value, trace

"""V18 无 Goal 的 stable-grasp residual v-diffusion。"""
from __future__ import annotations

import torch
from torch import nn

from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule, sinusoidal_embedding
from src.task.InteractionDynamics.residual_interaction_diffusion import recover_x0_noise
from src.task.InteractionDynamics.state_interaction_diffusion import AdaLNBlock


class GraspInteractionDiffusion(nn.Module):
    def __init__(self, horizon: int = 8, dim: int = 256, heads: int = 8,
                 layers: int = 6) -> None:
        super().__init__()
        self.residual_dim, self.dim = 7 * horizon, dim
        self.noisy_residual = nn.Sequential(
            nn.Linear(self.residual_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.object_point = nn.Sequential(
            nn.Linear(6, 128), nn.GELU(), nn.Linear(128, dim), nn.GELU())
        self.state = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(layers)])
        self.output_norm, self.output = nn.LayerNorm(dim), nn.Linear(dim, self.residual_dim)

    def forward(self, noisy_residual: torch.Tensor, state: torch.Tensor,
                anchors_cm: torch.Tensor, patches: torch.Tensor,
                timestep: torch.Tensor) -> torch.Tensor:
        tokens = (self.noisy_residual(noisy_residual) + self.state(state)
                  + self.anchor(anchors_cm) + self.object_point(patches).amax(2))
        condition = self.time(sinusoidal_embedding(timestep, self.dim))
        for block in self.blocks: tokens = block(tokens, condition)
        return self.output(self.output_norm(tokens))


@torch.no_grad()
def sample_grasp_v(model: GraspInteractionDiffusion, state: torch.Tensor,
                   anchors_cm: torch.Tensor, patches: torch.Tensor, steps: int = 100,
                   initial_noise: torch.Tensor | None = None,
                   sampling_steps: int = 10) -> torch.Tensor:
    _, alpha_bar = cosine_schedule(steps, anchors_cm.device)
    value = torch.randn((*anchors_cm.shape[:2], model.residual_dim),
                        device=anchors_cm.device) if initial_noise is None else initial_noise.clone()
    indices = torch.unique_consecutive(
        torch.linspace(steps - 1, 0, sampling_steps, device=value.device).round().long()).tolist()
    clean = value
    for position, index in enumerate(indices):
        timestep = torch.full((len(value),), index, device=value.device, dtype=torch.long)
        velocity = model(value, state, anchors_cm, patches, timestep)
        clean, noise = recover_x0_noise(value, velocity, alpha_bar[index].reshape(1, 1, 1))
        if not torch.isfinite(clean).all():
            raise FloatingPointError(f"non-finite V18 sample at timestep {index}")
        if position + 1 < len(indices):
            next_index = indices[position + 1]
            value = alpha_bar[next_index].sqrt() * clean + (1 - alpha_bar[next_index]).sqrt() * noise
    return clean

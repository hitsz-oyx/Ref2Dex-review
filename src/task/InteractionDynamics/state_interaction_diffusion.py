"""V16.5 当前状态条件的短时 interaction diffusion。"""
from __future__ import annotations

import torch
from torch import nn

from src.task.InteractionDynamics.interaction_diffusion import cosine_schedule, sinusoidal_embedding


def pack_state_future(y: dict[str, torch.Tensor], horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
    """把解析 Teacher 拆成 clean S_t[M,4] 与 F_t^H[M,7H]。"""
    r, d, u = y["relative_geometry"], y["relative_distance"], y["relative_motion"]
    if horizon < 1 or r.shape[0] < horizon + 1 or u.shape[0] < horizon:
        raise ValueError("Teacher trajectory is shorter than the requested horizon")
    state = torch.cat([r[0], d[0, :, None]], -1)
    future = torch.cat([
        torch.cat([u[step], r[step + 1], d[step + 1, :, None]], -1)
        for step in range(horizon)
    ], -1)
    return state, future


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 4 * dim))

    @staticmethod
    def modulate(value: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return (1 + scale[:, None]) * value + shift[:, None]

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, shift_f, scale_f = self.modulation(condition).chunk(4, -1)
        normalized = self.modulate(self.attn_norm(tokens), shift_a, scale_a)
        tokens = tokens + self.attn(normalized, normalized, normalized, need_weights=False)[0]
        return tokens + self.ff(self.modulate(self.ff_norm(tokens), shift_f, scale_f))


class StateInteractionDiffusion(nn.Module):
    """对 anchor permutation 等价的 state-conditioned epsilon predictor。"""
    def __init__(self, horizon: int = 4, dim: int = 256, heads: int = 8,
                 layers: int = 6) -> None:
        super().__init__()
        self.horizon = horizon
        self.future_dim = 7 * horizon
        self.dim = dim
        self.object_point = nn.Sequential(
            nn.Linear(6, 128), nn.GELU(), nn.Linear(128, dim), nn.GELU())
        self.state = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.future = nn.Sequential(
            nn.Linear(self.future_dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.effect = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.time = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.condition = nn.Sequential(nn.Linear(2 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, self.future_dim)

    def forward(self, noisy_future: torch.Tensor, current_state: torch.Tensor,
                anchors_cm: torch.Tensor, object_patches: torch.Tensor,
                endpoint_effect: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        object_feature = self.object_point(object_patches).amax(2)
        tokens = (self.future(noisy_future) + self.state(current_state)
                  + self.anchor(anchors_cm) + object_feature)
        condition = self.condition(torch.cat([
            self.effect(endpoint_effect),
            self.time(sinusoidal_embedding(timestep, self.dim))], -1))
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.output(self.output_norm(tokens))


@torch.no_grad()
def sample_state_ddim(model: StateInteractionDiffusion, current_state: torch.Tensor,
                      anchors_cm: torch.Tensor, object_patches: torch.Tensor,
                      endpoint_effect: torch.Tensor, steps: int,
                      initial_noise: torch.Tensor | None = None,
                      generator: torch.Generator | None = None) -> torch.Tensor:
    _, alpha_bar = cosine_schedule(steps, anchors_cm.device)
    value = (torch.randn((*anchors_cm.shape[:2], model.future_dim), device=anchors_cm.device,
                         generator=generator) if initial_noise is None else initial_noise.clone())
    for index in reversed(range(steps)):
        timestep = torch.full((len(value),), index, device=value.device, dtype=torch.long)
        epsilon = model(value, current_state, anchors_cm, object_patches,
                        endpoint_effect, timestep)
        clean = ((value - (1 - alpha_bar[index]).sqrt() * epsilon)
                 / alpha_bar[index].sqrt()).clamp(-5, 5)
        value = (alpha_bar[index - 1].sqrt() * clean
                 + (1 - alpha_bar[index - 1]).sqrt() * epsilon) if index else clean
    return value

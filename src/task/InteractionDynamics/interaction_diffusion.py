"""V16.4 Object-Effect conditional analytic interaction-field diffusion。"""
from __future__ import annotations

import math

import torch
from torch import nn


def pack_anchor_y(y: dict[str, torch.Tensor]) -> torch.Tensor:
    """把 Y-Teacher V1 从时间主序整理为 anchor-major [B,M,60]。"""
    prefix = y["relative_geometry"].shape[:-3]
    anchors = y["relative_geometry"].shape[-2]
    r = y["relative_geometry"].transpose(-3, -2).reshape(*prefix, anchors, 27)
    d = y["relative_distance"].transpose(-2, -1)
    u = y["relative_motion"].transpose(-3, -2).reshape(*prefix, anchors, 24)
    return torch.cat([r, d, u], -1)


def unpack_anchor_y(value: torch.Tensor) -> dict[str, torch.Tensor]:
    prefix, anchors = value.shape[:-2], value.shape[-2]
    return {
        "relative_geometry": value[..., :27].reshape(*prefix, anchors, 9, 3).transpose(-3, -2),
        "relative_distance": value[..., 27:36].transpose(-2, -1),
        "relative_motion": value[..., 36:].reshape(*prefix, anchors, 8, 3).transpose(-3, -2),
    }


def sinusoidal_embedding(timestep: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    frequency = torch.exp(
        -math.log(10000) * torch.arange(half, device=timestep.device) / max(half - 1, 1))
    phase = timestep.float()[:, None] * frequency[None]
    embedding = torch.cat([phase.sin(), phase.cos()], -1)
    return torch.nn.functional.pad(embedding, (0, dim - embedding.shape[-1]))


class ConditionedInteractionBlock(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(dim)
        self.self_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ff_norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, interaction: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        normalized = self.self_norm(interaction)
        interaction = interaction + self.self_attention(
            normalized, normalized, normalized, need_weights=False)[0]
        interaction = interaction + self.cross_attention(
            self.cross_norm(interaction), condition, condition, need_weights=False)[0]
        return interaction + self.ff(self.ff_norm(interaction))


class InteractionFieldDiffusion(nn.Module):
    """无 anchor index embedding 的 permutation-equivariant epsilon predictor。"""
    def __init__(self, dim: int = 256, heads: int = 8, layers: int = 6) -> None:
        super().__init__()
        self.dim = dim
        self.object_point = nn.Sequential(
            nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim), nn.LayerNorm(dim))
        self.effect = nn.Sequential(nn.Linear(6, dim), nn.GELU(), nn.Linear(dim, dim))
        self.effect_time = nn.Embedding(8, dim)
        self.r_embedding = nn.Sequential(nn.Linear(27, dim), nn.GELU(), nn.Linear(dim, dim))
        self.d_embedding = nn.Sequential(nn.Linear(9, dim), nn.GELU(), nn.Linear(dim, dim))
        self.u_embedding = nn.Sequential(nn.Linear(24, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor_embedding = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.diffusion_time = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList([ConditionedInteractionBlock(dim, heads) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(dim)
        self.r_head = nn.Linear(dim, 27)
        self.d_head = nn.Linear(dim, 9)
        self.u_head = nn.Linear(dim, 24)

    def forward(self, noisy_y: torch.Tensor, anchors_cm: torch.Tensor,
                anchor_normals: torch.Tensor, effect: torch.Tensor,
                timestep: torch.Tensor) -> torch.Tensor:
        object_feature = self.object_point(torch.cat([anchors_cm, anchor_normals], -1))
        global_object = object_feature.amax(1, keepdim=True)
        effect_time = torch.arange(8, device=effect.device)[None].expand(effect.shape[0], -1)
        effect_tokens = self.effect(effect) + self.effect_time(effect_time)
        condition = torch.cat([global_object, effect_tokens], 1)
        interaction = (self.r_embedding(noisy_y[..., :27])
                       + self.d_embedding(noisy_y[..., 27:36])
                       + self.u_embedding(noisy_y[..., 36:])
                       + self.anchor_embedding(anchors_cm)
                       + object_feature
                       + self.diffusion_time(sinusoidal_embedding(timestep, self.dim))[:, None])
        for block in self.blocks:
            interaction = block(interaction, condition)
        interaction = self.output_norm(interaction)
        return torch.cat([self.r_head(interaction), self.d_head(interaction),
                          self.u_head(interaction)], -1)


def cosine_schedule(steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    time = torch.linspace(0, steps, steps + 1, device=device) / steps
    alpha_bar = torch.cos((time + .008) / 1.008 * math.pi / 2).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    beta = (1 - alpha_bar[1:] / alpha_bar[:-1]).clamp(1e-5, .999)
    alpha = 1 - beta
    return alpha, torch.cumprod(alpha, 0)


@torch.no_grad()
def sample_ddpm(model: InteractionFieldDiffusion, anchors_cm: torch.Tensor,
                anchor_normals: torch.Tensor, effect: torch.Tensor, steps: int,
                generator: torch.Generator | None = None,
                initial_noise: torch.Tensor | None = None) -> torch.Tensor:
    _, alpha_bar = cosine_schedule(steps, anchors_cm.device)
    value = (torch.randn((*anchors_cm.shape[:2], 60), device=anchors_cm.device,
                         generator=generator) if initial_noise is None else initial_noise.clone())
    for index in reversed(range(steps)):
        timestep = torch.full((value.shape[0],), index, device=value.device, dtype=torch.long)
        epsilon = model(value, anchors_cm, anchor_normals, effect, timestep)
        clean = ((value - (1 - alpha_bar[index]).sqrt() * epsilon)
                 / alpha_bar[index].sqrt()).clamp(-5, 5)
        if index:
            value = (alpha_bar[index - 1].sqrt() * clean
                     + (1 - alpha_bar[index - 1]).sqrt() * epsilon)
        else:
            value = clean
    return value

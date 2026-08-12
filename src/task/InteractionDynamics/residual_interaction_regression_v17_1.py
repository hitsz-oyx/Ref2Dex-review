"""V17.1 可选 time-to-effect 条件的确定性 residual 回归器。"""
from __future__ import annotations

import torch
from torch import nn

from src.task.InteractionDynamics.state_interaction_diffusion import AdaLNBlock


def temporal_goal(goal: torch.Tensor, time_to_effect: torch.Tensor,
                  mean: torch.Tensor, std: torch.Tensor,
                  enabled: bool) -> torch.Tensor:
    """追加标准化 log(1+tau)；缺失值和 baseline 均为零。"""
    value = torch.log1p(time_to_effect.clamp_min(0).float()).unsqueeze(-1)
    value = (value - mean) / std
    valid = (time_to_effect >= 0).unsqueeze(-1)
    value = torch.where(valid & enabled, value, torch.zeros_like(value))
    return torch.cat([goal, value], -1)


class TemporalResidualInteractionRegressor(nn.Module):
    def __init__(self, horizon: int = 4, goal_segment: int = 4, dim: int = 256,
                 heads: int = 8, layers: int = 6) -> None:
        super().__init__()
        self.residual_dim = 7 * horizon
        self.object_point = nn.Sequential(
            nn.Linear(6, 128), nn.GELU(), nn.Linear(128, dim), nn.GELU())
        self.state = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.goal = nn.Sequential(
            nn.Linear(2 + 6 * goal_segment, 128), nn.GELU(), nn.Linear(128, dim))
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(layers)])
        self.output_norm, self.output = nn.LayerNorm(dim), nn.Linear(dim, self.residual_dim)

    def forward(self, state: torch.Tensor, anchors_cm: torch.Tensor,
                object_patches: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        tokens = (self.state(state) + self.anchor(anchors_cm)
                  + self.object_point(object_patches).amax(2))
        condition = self.goal(goal)
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.output(self.output_norm(tokens))

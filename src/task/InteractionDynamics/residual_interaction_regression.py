"""V16.7 clean persistence-residual interaction regression。"""
from __future__ import annotations

import torch
from torch import nn

from src.task.InteractionDynamics.state_interaction_diffusion import AdaLNBlock


def persistence_future(state: torch.Tensor, horizon: int) -> torch.Tensor:
    """构造 [u=0, S_t] 重复 H 次的 persistence future。"""
    if state.shape[-1] != 4 or horizon < 1:
        raise ValueError("state must end in 4 channels and horizon must be positive")
    step = torch.cat([torch.zeros_like(state[..., :3]), state], -1)
    return torch.cat([step for _ in range(horizon)], -1)


def residual_target(state: torch.Tensor, future: torch.Tensor, horizon: int) -> torch.Tensor:
    persistence = persistence_future(state, horizon)
    if persistence.shape != future.shape:
        raise ValueError(f"future shape {future.shape} does not match {persistence.shape}")
    return future - persistence


class ResidualInteractionRegressor(nn.Module):
    """只读取当前状态、当前物体几何和 Goal 的 anchor-equivariant 回归器。"""

    def __init__(self, horizon: int = 4, goal_segment: int = 4, dim: int = 256,
                 heads: int = 8, layers: int = 6) -> None:
        super().__init__()
        self.horizon = horizon
        self.goal_segment = goal_segment
        self.residual_dim = 7 * horizon
        self.object_point = nn.Sequential(
            nn.Linear(6, 128), nn.GELU(), nn.Linear(128, dim), nn.GELU())
        self.state = nn.Sequential(nn.Linear(4, dim), nn.GELU(), nn.Linear(dim, dim))
        self.anchor = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.goal = nn.Sequential(
            nn.Linear(1 + 6 * goal_segment, 128), nn.GELU(), nn.Linear(128, dim))
        self.blocks = nn.ModuleList([AdaLNBlock(dim, heads) for _ in range(layers)])
        self.output_norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, self.residual_dim)

    def forward(self, current_state: torch.Tensor, anchors_cm: torch.Tensor,
                object_patches: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        tokens = (self.state(current_state) + self.anchor(anchors_cm)
                  + self.object_point(object_patches).amax(2))
        condition = self.goal(goal)
        for block in self.blocks:
            tokens = block(tokens, condition)
        return self.output(self.output_norm(tokens))

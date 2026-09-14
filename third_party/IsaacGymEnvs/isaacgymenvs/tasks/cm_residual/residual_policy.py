"""Residual actor composition: frozen DExplore action plus Cm-conditioned 18D delta."""
from __future__ import annotations

import torch
from torch import nn

from .base_policy import ACTION_DIM, OBSERVATION_DIM, InspireDExplorePolicy
from .cm_adapter import CmOnlineTarget


class CmResidualActor(nn.Module):
    """Reference actor used by integrations that train outside IsaacGym's VecTask."""

    def __init__(self, checkpoint: str, cm_feature_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.teacher = InspireDExplorePolicy(checkpoint)
        self.cm = CmOnlineTarget(OBSERVATION_DIM, cm_feature_dim)
        self.delta = nn.Sequential(
            nn.Linear(OBSERVATION_DIM + cm_feature_dim, hidden), nn.ELU(),
            nn.Linear(hidden, ACTION_DIM), nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        base_action = self.teacher(obs)
        online, _ = self.cm.features(obs)
        residual = self.delta(torch.cat((obs, online), dim=-1))
        return (base_action + residual).clamp(-1.0, 1.0)

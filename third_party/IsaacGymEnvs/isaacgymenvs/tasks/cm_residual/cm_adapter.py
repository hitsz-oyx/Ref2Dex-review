"""Cm online/target lifecycle for residual-policy training."""
from __future__ import annotations

import copy
from typing import Optional
import torch
from torch import nn


class CmOnlineTarget(nn.Module):
    """Small, explicit online/target interface; target is frozen per PPO block."""

    def __init__(self, input_dim: int, feature_dim: int = 128, ema_decay: float = 0.995):
        super().__init__()
        self.online = nn.Sequential(nn.Linear(input_dim, feature_dim), nn.ELU(), nn.Linear(feature_dim, feature_dim))
        self.target = copy.deepcopy(self.online)
        self.ema_decay = float(ema_decay)
        self._freeze_target()

    def _freeze_target(self):
        self.target.eval()
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def begin_ppo_block(self):
        """Freeze target predictions for one PPO block and sync from online by EMA."""
        for target, online in zip(self.target.parameters(), self.online.parameters()):
            target.mul_(self.ema_decay).add_(online, alpha=1.0 - self.ema_decay)
        self._freeze_target()

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.online(x), self.target(x).detach()

    def supervised_loss(self, x: torch.Tensor) -> torch.Tensor:
        online, target = self.features(x)
        return torch.nn.functional.mse_loss(online, target)

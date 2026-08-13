"""V20 causal object-indexed interaction field Y=[r,d,v,p]。"""
from __future__ import annotations

import torch

from src.task.InteractionDynamics.interaction_field import (
    build_relative_distance, build_relative_geometry, build_soft_correspondence)


def build_causal_field(hand_points: torch.Tensor, anchors: torch.Tensor,
                       penetration: torch.Tensor, tau_m: float = .015) -> torch.Tensor:
    """返回 `[T-1,N,8]`；第 t 项只使用 hand[t-1:t+1]。"""
    weights = build_soft_correspondence(hand_points, anchors, tau_m)
    r = build_relative_geometry(hand_points, anchors, weights)
    d = build_relative_distance(hand_points, anchors, weights)
    velocity = (weights[1:, ..., None]
                * (hand_points[1:] - hand_points[:-1])[:, None]).sum(-2)
    p = (weights * penetration[:, None]).sum(-1)
    return torch.cat([r[1:], d[1:, :, None], velocity, p[1:, :, None]], -1)


def residual_target(current_y: torch.Tensor, future_y: torch.Tensor) -> torch.Tensor:
    if current_y.ndim != 2 or future_y.ndim != 3:
        raise ValueError("Expected current [N,8] and future [N,H,8]")
    return future_y - current_y[:, None]

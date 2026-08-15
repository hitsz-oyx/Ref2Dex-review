from __future__ import annotations

import torch


def build_edges(object_points: torch.Tensor, hand_points: torch.Tensor, k: int = 16,
                radius: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """Return K nearest hand indices and validity (distance <= radius)."""
    if object_points.ndim != 3 or hand_points.ndim != 3:
        raise ValueError("points must have shape [B,N,3]")
    k = min(int(k), hand_points.shape[1])
    distances = torch.cdist(object_points, hand_points)
    values, indices = torch.topk(distances, k=k, dim=-1, largest=False)
    valid = values <= float(radius)
    return indices, valid

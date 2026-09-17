"""Deterministic synthetic rigid contract for V1.0 smoke tests."""
from __future__ import annotations

import torch

from .model import transform_points


def make_synthetic_batch(batch_size: int = 4, num_object: int = 32, num_hand: int = 64, seed: int = 42):
    generator = torch.Generator().manual_seed(seed)
    points = torch.rand(batch_size, num_object, 3, generator=generator) * 0.08 - 0.04
    normals = torch.nn.functional.normalize(points + 1e-3, dim=-1)
    hand_points = torch.rand(batch_size, num_hand, 3, generator=generator) * 0.08 - 0.04
    hand_normals = torch.nn.functional.normalize(hand_points + 1e-3, dim=-1)
    hand_flow = torch.zeros_like(hand_points)
    hand_flow[:, : num_hand // 2, 0] = 0.002
    xi = torch.zeros(batch_size, 6)
    xi[:, 0] = 0.001
    obj_flow = transform_points(points, xi) - points
    return {"obj_points": points, "obj_normals": normals, "hand_points": hand_points,
            "hand_normals": hand_normals, "hand_flow": hand_flow,
            "hand_valid_mask": torch.ones(batch_size, num_hand, dtype=torch.bool),
            "obj_flow_gt": obj_flow, "obj_link_id": torch.zeros(batch_size, num_object, dtype=torch.long),
            "num_links": torch.ones(batch_size, dtype=torch.long)}

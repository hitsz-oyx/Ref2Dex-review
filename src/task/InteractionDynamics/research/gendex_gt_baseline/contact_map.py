"""与 GenDexGrasp CMapAdam align_dist 一致的 contact-map 定义。"""
from __future__ import annotations

import torch


def build_gendex_contact_value(object_points: torch.Tensor, object_normals: torch.Tensor,
                               hand_points: torch.Tensor) -> torch.Tensor:
    """输入 meter 坐标，返回每个 object point 的官方口径 contact value。"""
    delta = hand_points[None] - object_points[:, None]
    distance = torch.linalg.norm(delta, dim=-1)
    alignment = (delta * object_normals[:, None]).sum(-1) / (distance + 1e-5)
    aligned_distance = distance * torch.exp(2 * (1 - alignment))
    contact_distance = torch.sqrt(aligned_distance.min(dim=1).values)
    return 1 - 2 * (torch.sigmoid(10 * contact_distance) - .5)


def build_gendex_gt_map(object_points: torch.Tensor, object_normals: torch.Tensor,
                        hand_points: torch.Tensor) -> torch.Tensor:
    contact = build_gendex_contact_value(object_points, object_normals, hand_points)
    return torch.cat((object_points, object_normals, contact[:, None]), dim=-1)

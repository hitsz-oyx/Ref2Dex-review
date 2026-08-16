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


def build_edges_bimanual(object_points: torch.Tensor, left_points: torch.Tensor,
                         right_points: torch.Tensor, k_l: int = 8, k_r: int = 8,
                         radius: float = 0.05) -> tuple[torch.Tensor, torch.Tensor]:
    """V1.0 双手 split KNN：每个 object point 分别取左手 K_L、右手 K_R 近邻。

    返回的 edge_idx 指向 concat([left, right]) 的手点序列（右手索引带
    left_points.shape[1] 偏移），保证双手 interaction 都有机会进入 C_obj；
    总 edge 数 K_L+K_R 与 V0.x 的 K=16 计算量一致。
    """
    if object_points.ndim != 3 or left_points.ndim != 3 or right_points.ndim != 3:
        raise ValueError("points must have shape [B,N,3]")
    n_left = left_points.shape[1]
    idx_l, valid_l = build_edges(object_points, left_points, k_l, radius)
    idx_r, valid_r = build_edges(object_points, right_points, k_r, radius)
    return torch.cat([idx_l, idx_r + n_left], dim=-1), torch.cat([valid_l, valid_r], dim=-1)

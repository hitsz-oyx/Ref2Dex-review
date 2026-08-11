"""Y-Teacher V1：解析、可微的 object-centric interaction field。"""
from __future__ import annotations

import torch


def build_soft_correspondence(
    hand_points_object: torch.Tensor,
    object_anchors: torch.Tensor,
    tau_m: float = 0.015,
) -> torch.Tensor:
    """返回每个时刻、每个 object anchor 对 hand surface 的 soft correspondence。"""
    if hand_points_object.ndim != 3 or object_anchors.ndim != 2:
        raise ValueError("Expected hand [frames, points, 3] and anchors [anchors, 3]")
    if hand_points_object.shape[-1] != 3 or object_anchors.shape[-1] != 3:
        raise ValueError("Interaction geometry must be 3D")
    if tau_m <= 0:
        raise ValueError("tau_m must be positive")
    squared_distance = (
        hand_points_object[:, None] - object_anchors[None, :, None]
    ).square().sum(-1)
    return torch.softmax(-squared_distance / float(tau_m) ** 2, dim=-1)


def build_relative_geometry(
    hand_points_object: torch.Tensor,
    object_anchors: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    relative = hand_points_object[:, None] - object_anchors[None, :, None]
    return (weights[..., None] * relative).sum(-2)


def build_relative_distance(
    hand_points_object: torch.Tensor,
    object_anchors: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """返回 soft correspondence 下的真实期望距离，避免平均向量相互抵消。"""
    relative = hand_points_object[:, None] - object_anchors[None, :, None]
    return (weights * relative.norm(dim=-1)).sum(-1)


def build_relative_motion(
    hand_points_object: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """用 t 时刻的权重跟踪同一批 surface points 的 t→t+1 位移。"""
    displacement = torch.diff(hand_points_object, dim=0)
    return (weights[:-1, ..., None] * displacement[:, None]).sum(-2)


def build_interaction_y(
    hand_points_object: torch.Tensor,
    object_anchors: torch.Tensor,
    tau_m: float = 0.015,
) -> dict[str, torch.Tensor]:
    """构造冻结定义 Y={r[0:T], d[0:T], u[0:T-1]}。

    ``interaction_y`` 仅是兼容旧诊断的 transition-level ``[r[0:T-1],u]``；
    完整 teacher 必须读取三个具名字段，不能由该拼接 tensor 代替。
    """
    if hand_points_object.shape[0] < 2:
        raise ValueError("At least two hand frames are required")
    weights = build_soft_correspondence(hand_points_object, object_anchors, tau_m)
    relative_geometry = build_relative_geometry(
        hand_points_object, object_anchors, weights)
    relative_distance = build_relative_distance(
        hand_points_object, object_anchors, weights)
    relative_motion = build_relative_motion(hand_points_object, weights)
    return {
        "weights": weights,
        "relative_geometry": relative_geometry,
        "relative_distance": relative_distance,
        "relative_motion": relative_motion,
        "interaction_y": torch.cat([relative_geometry[:-1], relative_motion], dim=-1),
    }

from __future__ import annotations

import torch


@torch.no_grad()
def epe(pred: torch.Tensor, target: torch.Tensor) -> float:
    """V0.8 唯一指标：EPE = 每点 L2 误差均值（米）。"""
    error = torch.linalg.vector_norm(pred - target, dim=-1)
    return float(error.mean())


@torch.no_grad()
def flow_metrics(pred: torch.Tensor, target: torch.Tensor,
                 points: torch.Tensor | None = None) -> dict[str, float]:
    error = torch.linalg.vector_norm(pred - target, dim=-1)
    return {"point_error_m": float(error.mean()),
            "translation_error_m": float(torch.linalg.vector_norm(pred.mean(1) - target.mean(1), dim=-1).mean()),
            "rotation_error_deg": rigid_rotation_error_deg(points, pred, target) if points is not None else float("nan")}


@torch.no_grad()
def rigid_rotation_error_deg(points: torch.Tensor, pred_flow: torch.Tensor, target_flow: torch.Tensor) -> float:
    """Kabsch rotation disagreement between predicted and target displaced clouds."""
    pred = points + pred_flow
    target = points + target_flow
    pred = pred - pred.mean(1, keepdim=True)
    target = target - target.mean(1, keepdim=True)
    cov = pred.transpose(1, 2) @ target
    u, _, vh = torch.linalg.svd(cov)
    det = torch.det(vh.transpose(1, 2) @ u.transpose(1, 2))
    correction = torch.eye(3, device=pred.device).unsqueeze(0).repeat(pred.shape[0], 1, 1)
    correction[:, -1, -1] = torch.where(det < 0, -1.0, 1.0)
    rot = vh.transpose(1, 2) @ correction @ u.transpose(1, 2)
    cos = ((torch.diagonal(rot, dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0).clamp(-1.0, 1.0)
    return float(torch.rad2deg(torch.acos(cos)).mean())

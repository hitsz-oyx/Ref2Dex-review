from __future__ import annotations

import torch
import torch.nn.functional as F


def contact_target_from_distance(
    distance: torch.Tensor,
    contact_radius: float,
) -> torch.Tensor:
    if contact_radius <= 0:
        raise ValueError("contact_radius must be positive.")
    return (1.0 - distance.float() / float(contact_radius)).clamp(0.0, 1.0)


def quality_focal_loss_map(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    beta: float = 2.0,
) -> torch.Tensor:
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    logits = logits.float()
    target = target.float()
    pred_prob = torch.sigmoid(logits)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    modulating_factor = torch.abs(target - pred_prob).pow(beta)
    return modulating_factor * bce


def reduce_loss_map(
    loss_map: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum()
    if float(denom.item()) <= 0.0:
        return loss_map.sum() * 0.0
    return (loss_map.float() * mask_f).sum() / denom


def reduce_loss_map_per_object(
    loss_map: torch.Tensor,
    edge_mask: torch.Tensor,
    obj_valid_mask: torch.Tensor,
) -> torch.Tensor:
    edge_mask_f = edge_mask.float()
    per_obj_weight = edge_mask_f.sum(dim=-1)
    per_obj_loss = (loss_map.float() * edge_mask_f).sum(dim=-1) / per_obj_weight.clamp_min(
        torch.finfo(loss_map.dtype).eps
    )
    valid_obj_mask = obj_valid_mask.bool() & (per_obj_weight > 0)
    if bool(valid_obj_mask.any()):
        return per_obj_loss[valid_obj_mask].mean()
    return loss_map.sum() * 0.0

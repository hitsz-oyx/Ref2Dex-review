from __future__ import annotations

import torch
import torch.nn.functional as F


_EPS = 1e-6


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


def binary_cross_entropy_with_logits_map(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), target.float(), reduction="none")


def reduce_loss_map(
    loss_map: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask_f = mask.float()
    weighted = (loss_map.float() * mask_f).sum()
    denom = mask_f.sum()
    safe = weighted / denom.clamp_min(torch.finfo(loss_map.dtype).eps)
    return torch.where(denom > 0, safe, weighted * 0.0)


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
    valid_obj_f = valid_obj_mask.float()
    total = (per_obj_loss * valid_obj_f).sum()
    count = valid_obj_f.sum()
    safe = total / count.clamp_min(torch.finfo(loss_map.dtype).eps)
    return torch.where(count > 0, safe, loss_map.sum() * 0.0)


def binary_entropy_floor_map(target: torch.Tensor) -> torch.Tensor:
    """Per-element binary entropy ``H(y) = -[y log y + (1-y) log (1-y)]``.

    Acts as the theoretical lower bound of the BCE between a probabilistic
    prediction and a soft target ``y in (0, 1)``.
    """
    target = target.float().clamp(min=_EPS, max=1.0 - _EPS)
    return -(target * torch.log(target) + (1.0 - target) * torch.log(1.0 - target))


def zero_predictor_qfl_map(
    target: torch.Tensor,
    *,
    zero_logit: float = -20.0,
    beta: float = 2.0,
) -> torch.Tensor:
    """QFL map for a fixed ``zero_logit`` baseline (default logits=-20, p≈0)."""
    zero_logits = torch.full_like(target, float(zero_logit))
    return quality_focal_loss_map(zero_logits, target, beta=beta)


def zero_predictor_bce_map(
    target: torch.Tensor,
    *,
    zero_logit: float = -20.0,
) -> torch.Tensor:
    """BCE map for a fixed ``zero_logit`` baseline."""
    zero_logits = torch.full_like(target, float(zero_logit))
    return binary_cross_entropy_with_logits_map(zero_logits, target)


def zero_predictor_mae_map(target: torch.Tensor) -> torch.Tensor:
    """MAE map for a ``p=0`` zero predictor (assumes p≈0)."""
    return target.float().abs()


def target_strength_bin_mask(
    target: torch.Tensor,
    *,
    lower: float,
    upper: float,
    upper_inclusive: bool = True,
) -> torch.Tensor:
    """Boolean mask selecting targets in ``(lower, upper]`` (or ``(lower, upper)``)."""
    if lower < 0 or upper > 1 or lower >= upper:
        raise ValueError(f"Invalid target bin: lower={lower}, upper={upper}.")
    if upper_inclusive:
        return (target > lower) & (target <= upper)
    return (target > lower) & (target < upper)

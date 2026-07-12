from __future__ import annotations

import torch


def reduce_loss_map_per_object(
    loss_map: torch.Tensor,
    edge_weight: torch.Tensor,
    obj_valid_mask: torch.Tensor,
) -> torch.Tensor:
    edge_weight = edge_weight.float()
    weighted_loss = loss_map.float() * edge_weight
    per_obj_weight = edge_weight.sum(dim=-1)
    per_obj_loss = weighted_loss.sum(dim=-1) / per_obj_weight.clamp_min(
        torch.finfo(weighted_loss.dtype).eps
    )
    valid_obj_mask = obj_valid_mask.bool() & (per_obj_weight > 0)
    if bool(valid_obj_mask.any()):
        return per_obj_loss[valid_obj_mask].mean()
    return weighted_loss.sum() * 0.0


__all__ = ["reduce_loss_map_per_object"]

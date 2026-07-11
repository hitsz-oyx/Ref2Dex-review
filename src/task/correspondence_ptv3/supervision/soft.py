from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from src.task.correspondence_ptv3.contracts import ContactLossResult


def flatten_binary_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if logits.shape == target.shape:
        return logits
    if logits.shape[:-1] == target.shape and logits.shape[-1] == 1:
        return logits.squeeze(-1)
    raise ValueError(
        "Binary contact logits must have shape matching target or trailing singleton "
        f"dimension, got logits={tuple(logits.shape)} target={tuple(target.shape)}."
    )


def binary_entropy_floor_map(
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    with torch.autocast(device_type=target.device.type, enabled=False):
        target = target.float()
        clamped = target.clamp(min=eps, max=1.0 - eps)
        entropy = -(
            clamped * torch.log(clamped)
            + (1.0 - clamped) * torch.log(1.0 - clamped)
        )
        return entropy.masked_fill((target <= 0.0) | (target >= 1.0), 0.0)


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


def masked_bce_with_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = flatten_binary_logits(logits, target).float()
        target = target.float()
        mask = mask.float()
        loss_map = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
        return (loss_map * mask).sum() / mask.sum().clamp(min=1.0)


def masked_bce_with_logits_per_object(
    logits: torch.Tensor,
    target: torch.Tensor,
    edge_weight: torch.Tensor,
    obj_valid_mask: torch.Tensor,
) -> torch.Tensor:
    with torch.autocast(device_type=logits.device.type, enabled=False):
        logits = flatten_binary_logits(logits, target).float()
        target = target.float()
        loss_map = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
        )
    return reduce_loss_map_per_object(
        loss_map,
        edge_weight,
        obj_valid_mask,
    )


@dataclass(frozen=True)
class SoftContactSupervision:
    name: str = "soft"

    @property
    def output_dim(self) -> int:
        return 1

    def decode(self, logits: torch.Tensor) -> torch.Tensor:
        if logits.shape[-1] != 1:
            raise ValueError(
                "Soft contact supervision expects last logit dimension to be 1, got "
                f"{tuple(logits.shape)}."
            )
        return torch.sigmoid(logits.squeeze(-1))

    def export_compat(
        self,
        logits: torch.Tensor,
        probability: torch.Tensor,
    ) -> torch.Tensor:
        del probability
        if logits.shape[-1] != 1:
            raise ValueError(
                "Soft contact supervision expects last logit dimension to be 1, got "
                f"{tuple(logits.shape)}."
            )
        return logits.squeeze(-1)

    def legacy_output_aliases(
        self,
        *,
        pred_obj_contact_logits: torch.Tensor,
        pred_cross_contact_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del pred_obj_contact_logits, pred_cross_contact_logits
        return {}

    def compute_object_loss(
        self,
        *,
        logits: torch.Tensor,
        target_probability: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> ContactLossResult:
        raw_bce = masked_bce_with_logits(
            logits,
            target_probability,
            valid_mask,
        )
        oracle_bce = (
            binary_entropy_floor_map(target_probability) * valid_mask.float()
        ).sum() / valid_mask.float().sum().clamp(min=1.0)
        excess_bce = raw_bce - oracle_bce.detach()
        return ContactLossResult(
            loss=excess_bce,
            metrics={
                "raw_bce": raw_bce.detach(),
                "oracle_bce": oracle_bce.detach(),
                "excess_bce": excess_bce.detach(),
            },
        )

    def compute_edge_loss(
        self,
        *,
        logits: torch.Tensor,
        target_probability: torch.Tensor,
        edge_weight: torch.Tensor,
        obj_valid_mask: torch.Tensor,
    ) -> ContactLossResult:
        raw_bce = masked_bce_with_logits_per_object(
            logits,
            target_probability,
            edge_weight,
            obj_valid_mask,
        )
        oracle_bce = reduce_loss_map_per_object(
            binary_entropy_floor_map(target_probability),
            edge_weight,
            obj_valid_mask,
        )
        excess_bce = raw_bce - oracle_bce.detach()
        return ContactLossResult(
            loss=excess_bce,
            metrics={
                "raw_bce": raw_bce.detach(),
                "oracle_bce": oracle_bce.detach(),
                "excess_bce": excess_bce.detach(),
            },
        )

from __future__ import annotations

import torch

from src.utils.correspondence import soft_contact_label


def build_dynamic_edge_contact_targets(
    batch: dict[str, torch.Tensor],
    *,
    num_obj_points: int,
    num_hand_points: int,
    d_pos: float,
    d_neg: float,
    gamma: float,
) -> torch.Tensor:
    points = batch.get("gt_points", batch["points"])
    obj_points = points[:, :num_obj_points]
    hand_points = points[:, num_obj_points : num_obj_points + num_hand_points]
    knn_idx = batch.get("input_obj_to_hand_logit_idx")
    if knn_idx is None:
        knn_idx = batch["input_obj_to_hand_knn_idx"]
    edge_valid = batch.get("input_obj_to_hand_logit_valid_mask")
    if edge_valid is None:
        edge_valid = batch["input_obj_to_hand_knn_valid_mask"]

    knn_idx = knn_idx.long()
    edge_valid = edge_valid.bool()
    safe_idx = knn_idx.clamp(min=0)
    batch_idx = torch.arange(points.shape[0], device=points.device).view(-1, 1, 1)
    neighbor_hand = hand_points[batch_idx, safe_idx]
    distance = torch.norm(neighbor_hand - obj_points.unsqueeze(2), dim=-1)
    labels = soft_contact_label(
        distance,
        d_pos=float(d_pos),
        d_neg=float(d_neg),
        gamma=float(gamma),
    )
    return labels * edge_valid.float()


def build_clean_correspondence_targets(
    batch: dict[str, torch.Tensor],
    *,
    corr_contact_label_min: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    clean_knn = batch["gt_obj_to_hand_knn_idx"].long()
    nearest_idx = clean_knn[..., 0]
    safe_idx = nearest_idx.clamp(min=0)

    hand_cano = batch["hand_cano_points"].float()
    hand_finger = batch["hand_finger_id"].long()
    hand_region = batch["hand_region_id"].long()
    if hand_cano.dim() == 2:
        hand_cano = hand_cano.unsqueeze(0)
        hand_finger = hand_finger.unsqueeze(0)
        hand_region = hand_region.unsqueeze(0)

    target_cano = _batch_gather(hand_cano, safe_idx)
    target_finger = _batch_gather(hand_finger, safe_idx)
    target_region = _batch_gather(hand_region, safe_idx)
    corr_valid = (
        batch["runtime_obj_valid_mask"].bool()
        & (nearest_idx >= 0)
        & (batch["obj_contact_label"] > float(corr_contact_label_min))
    ).float()
    return target_cano, corr_valid, target_finger, target_region


def _batch_gather(
    values: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    batch_idx = torch.arange(values.shape[0], device=values.device).view(-1, 1)
    return values[batch_idx, indices]


__all__ = [
    "build_clean_correspondence_targets",
    "build_dynamic_edge_contact_targets",
]
